---
date: '2026-01-19T22:12:33-08:00'
title: 'Visualizing Parallelism in Transformer'
tags: ["Distributed Training", "Transformer", "Parallelism", "ML Systems"]
---

## **Simplicity buried in Abstractions**

I've always loved the ["Transformer Accounting"](https://jax-ml.github.io/scaling-book/transformers/) diagram from the JAX Scaling Book. It did a brilliant job of making the tensor shapes of a Transformer intuitive on a single device.

But as we scale up, the complexity shifts. We stop worrying about just matrix dimensions and start worrying about the 'alphabet soup' of N-D parallelism (**DP, TP, SP, CP, EP**).

Here is the irony: The core ideas behind these parallelisms are actually **fundamentally easy**. Conceptually, we are just decomposing a global tensor operation into local tensor compute chunks connected by communication collectives. It's like an assembly line: instead of one worker building the whole car, we have a line of workers (GPUs) passing parts (tensors) back and forth.

But in practice, this simplicity gets buried under layers of dense abstractions and "spaghetti" implementations in production codebases.

I created the diagrams below to cut through that noise. I want to restore that intuition by visually mapping the dance between **Compute** and **Collectives**. The goal is to show exactly *how* the compute is sharded and *when* the synchronization happens: stripping away the implementation complexity to reveal the simple logic underneath.

## **How to Read Diagrams below**

**The Golden Rule:** To interpret the figures below, imagine you are sitting inside one single GPU.

### Model shape symbols
To make the abstract symbols concrete, I've included the dimensions of **[Llama 3 70B](https://ai.meta.com/blog/meta-llama-3/)** as a reference.

| Symbol | Definition | Description | Llama 3 70B |
| :--- | :--- | :--- | :--- |
| **B** | Batch Size | Global batch size (**sequences**). Total ~4M tokens. | 512 |
| **S** | Sequence Length | Context window size. | 8192 |
| **D** | Hidden Dim | Width of the residual stream. | 8192 |
| **V** | Vocab Size | Total size of the tokenizer vocabulary. | 128k |
| **F** | FFN Dim | Expansion dimension in MLP. | 28672 |
| **E** | Num. Experts | Total experts. | - (Llama 3 is dense) |
| **C** | Capacity | Max tokens per expert. | - (MoE specific) |

### Parallel Configuration (The Sharding Strategy)
These symbols represent the **size** of the process group used to shard a specific dimension.

| Symbol | Definition | Shards What? |
| :--- | :--- | :--- |
| **tp** | Tensor Parallel | Weights (shards **Heads** in Attention, **FFN Dim (F)** in MLP) |
| **sp** | Sequence Parallel | Activations (S) in Element-wise Ops |
| **cp** | Context Parallel | Sequence (S) in Attention (QKV) |
| **ep** | Expert Parallel | Experts (E) in MoE Layers |
| **vp** | Vocab Parallel | Vocabulary (V) in Embeddings/Loss |
| **dp** | Data Parallel | Batch (B) |

### The "Local Shape"
You will frequently see this specific shape entering and exiting the major blocks:

`[B/dp, S/(cp*sp), D]`

Read this literally from left to right:
1.  **`B/dp`**: "I only hold a fraction of the Batch." (Data Parallelism)
2.  **`S/(cp*sp)`**: "I only hold a tiny fragment of the Sequence." (Context & Sequence Parallelism)
3.  **`D`**: "I hold the **full** Hidden Model vector."

---

## **The Visual Walkthrough**

### Overview
This diagram provides a high-level overview of layers of a Transformer model. Note that for the Feed-Forward Network (FFN) block, I cover both the **dense** variant (standard MLP) and the **sparse** variant (Mixture of Experts), as modern large-scale models frequently toggle between these designs.

{{< figure src="./overview.svg" width="400px" align="center" >}}

---

### **Embeddings**

{{< figure src="./emb_parallel.svg" width="400px" align="center" >}}

The embedding table for a model like Llama 3 is massive (128k vocab * 8k dim ≈ 1GB bf16). Replicating this on every GPU is wasteful, so we shard the vocabulary itself (**Vocab Parallel**).

Each GPU holds a slice of the vocabulary and performs a local lookup. Most tokens won't be found locally and return zeros. Instead of summing these up with a costly AllReduce, we use a **ReduceScatter**: we sum the partial embeddings and *immediately* scatter them into the **Sequence Parallel** dimension (splitting the sequence across GPUs). This cuts communication overhead significantly right at the start.

---

### **Attention**

Here we see the complex interplay of different sequence strategies colliding in one block.

{{< figure src="./attn_parallel.svg" width="400px" align="center" >}}

To compute Self-Attention, every token needs to see every other token, but we enter this block with our sequence chopped up (**Sequence Parallel**). We trigger an `AllGather(sp)` to temporarily rebuild the local sequence.

Inside the attention core, we use **Context Parallel**. While the Query (Q) stays local, the Keys (K) and Values (V) are scattered across the CP ring and must be passed around so every token can attend to them. Finally, the Output Projection is Row Parallel (sharded by input heads), so we use the **ReduceScatter** trick again to sum the partial results and simultaneously return to the memory-efficient SP-sharded shape.

---

### **MLP (Feed Forward)**

{{< figure src="./mlp_parallel.svg" width="400px" align="center" >}}

The MLP block is the heavy lifter of compute. Just like Attention, we first rebuild the sequence with `AllGather(sp)`.

Then comes the "sandwich" of linear layers using **Tensor Parallel**. The first layer (Up Proj) expands the hidden dimension and is sharded column-wise. The second layer (Down Proj) shrinks it back down and is sharded row-wise. Since the Row Parallel output is a partial sum, we run a **ReduceScatter** to sum them up and instantly return to our **Sequence Parallel** (sharded sequence) state.

---

### **Mixture of Experts (MoE)**

This is the most complex diagram. We aren't just sharding tensors; we are actively routing tokens to different devices.

{{< figure src="./moe_parallel.svg" width="400px" align="center" >}}

Tokens need to go to their assigned experts, which live on different GPUs (**Expert Parallel**). We use an `AllToAll` collective—literally "shuffle everything to everywhere"—to dispatch them.

Once the tokens arrive, the computation looks just like a standard MLP (and can even use inner Tensor Parallelism). After processing, we use another `AllToAll` to route the tokens back to their original sequence position. This heavy shuffling makes the network interconnect the bottleneck for MoE training.

---

### **Loss**

Calculating Cross Entropy when the vocab is sharded (**Vocab Parallel**) is non-trivial. You cannot simply take a softmax because the denominator requires a global sum.

{{< figure src="./loss_parallel.svg" width="400px" align="center" >}}

We compute the logits locally (sharded by vocab). To perform the online softmax, we run `AllReduce(max)` and `AllReduce(sum)` to get the global stability stats. The tricky part is the target masking: the ground-truth label exists on *one* specific GPU, but the logits are scattered everywhere. We mask the non-local logits and use an `AllReduce(sum)` to broadcast the correct target logit to the group.

---

## **Closing Thoughts**

These diagrams represent a snapshot of a common, representative setup found in modern frameworks like **Megatron-Core** or **NVIDIA NeMo**. It is not the only way to do it.

The challenge in infrastructure is making these parallelisms easy to understand, debug, and compose without sacrificing the performant timing of those critical collectives. Hopefully, these visual maps help you navigate the territory.

**A Note on FSDP:**
You might notice **FSDP (Fully Sharded Data Parallel)** is intentionally left out. This is because FSDP is fundamentally a memory optimization (sharding weights and optimizer states at rest) rather than a computation parallelism. FSDP gathers the full weights just-in-time for the forward/backward pass, meaning it does not affect the local tensor shapes that participate in the actual compute operations. I might cover FSDP and its interaction with these parallelisms in a follow-up post.
