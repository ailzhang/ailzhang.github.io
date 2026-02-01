---
date: '2026-01-19T22:12:33-08:00'
draft: true
title: 'Visualizing Parallelism in Transformer'
description: "A visual guide to the 'alphabet soup' of Tensor, Sequence, Context, and Expert Parallelism."
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
| **E** | Num. Experts | Total experts (Llama 3 is dense). | - |
| **C** | Capacity | Max tokens per expert (MoE specific). | - |

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
This diagram provides a high-level overview of layers of a Transformer model. Note that for the Feed-Forward Network (FFN) block, I cover both the **Dense** variant (standard MLP) and the **Sparse** variant (Mixture of Experts), as modern large-scale models frequently toggle between these designs.

{{< figure src="./overview.svg" width="400px" align="center" >}}

---

### **Embeddings**

{{< figure src="./emb_parallel.svg" width="400px" align="center" >}}

**The Strategy:** Vocab Parallel (VP) → Sequence Parallel (SP)

**The Story:**
We start with input tokens that are sharded by the sequence dimension. The first challenge is the lookup. The embedding table for a model like Llama 3 is massive (128k rows * 8k dim ≈ 1GB of bf16 weights). We can't replicate this on every GPU.

Instead, we shard the vocabulary itself (Vocab Parallel). Each GPU holds a slice of the vocabulary. When we do a lookup, most tokens won't be found on our local slice—they return zeros.

**The Optimization (The ReduceScatter Trick):**
A naive approach would be to sum up all the partial lookups (AllReduce) and then split them again for the next layer. But that's wasteful. Instead, we use a **ReduceScatter**. We sum the partial embeddings from the Vocab Parallel lookup and *immediately* scatter them into the Sequence Parallel dimension. This cuts communication overhead significantly right at the start.

> **Why this matters:** Without VP, the embedding layer alone can consume gigabytes of redundant memory. VP spreads that cost, while the ReduceScatter ensures we don't pay a double penalty for communication.

---

### **Attention**

Here we see the complex interplay of different sequence strategies colliding in one block.

{{< figure src="./attn_parallel.svg" width="400px" align="center" >}}

**The Strategy:** Tensor Parallel (TP) + Sequence Parallel (SP) + Context Parallel (CP)

**The Story:**
1.  **Entry (Rebuild the Sequence):** We enter this block with our sequence chopped up into tiny pieces (Sequence Parallel). But to compute standard Self-Attention, we need to project Q, K, and V. We trigger an `AllGather(sp)` to temporarily reconstruct the local sequence segment.
2.  **The Attention Core (CP):** Now things get interesting. We use Context Parallelism here. While the Query (Q) stays local, the Keys (K) and Values (V) are scattered across the CP ring. We have to pass them around the ring (often overlapping this communication with the computation) so every token can attend to every other token.
3.  **Exit (The TP/SP Handoff):** After attention, we project the output. This projection is Row Parallel, meaning it results in partial sums sharded by the hidden dimension. We use the ReduceScatter trick again: we sum the partial results from TP and simultaneously reshard them back into the memory-efficient SP-sharded shape.

---

### **MLP (Feed Forward)**

{{< figure src="./mlp_parallel.svg" width="400px" align="center" >}}

**The Strategy:** Tensor Parallel (TP) + Sequence Parallel (SP)

**The Story:**
The MLP block is the heavy lifter of compute.
1.  **Entry:** Just like Attention, we can't process the sequence while it's sharded. We fire an `AllGather(sp)` to rebuild the sequence.
2.  **The Sandwich:** The MLP is a "sandwich" of linear layers. The first layer (Gate/Up projection) expands the hidden dimension (usually 4x). We slice this "column-wise" (Column Parallel). The second layer (Down projection) shrinks it back down, and we slice this "row-wise" (Row Parallel).
3.  **Exit:** The result of the Row Parallel layer is partial sums. We run a `ReduceScatter` to sum them up and instantly return to our sequence-sharded state.

---

### **Mixture of Experts (MoE)**

This is the most complex diagram. We aren't just sharding tensors; we are actively routing tokens to different devices.

{{< figure src="./moe_parallel.svg" width="400px" align="center" >}}

**The Strategy:** Expert Parallel (EP) + Tensor Parallel (TP)

**The Story:**
1.  **The Dispatch:** Tokens need to go to their assigned experts. Since experts live on different GPUs (Expert Parallel), we use an `AllToAll` collective—literally "shuffle everything to everywhere."
2.  **Inner Parallelism:** Once the tokens arrive at their expert, the computation looks just like a standard MLP. Interestingly, inside this expert, we can *still* apply Tensor Parallelism if the expert itself is too big!
3.  **The Return:** After the experts process the tokens, we have to send them back. We first `ReduceScatter` any inner Tensor Parallelism results, and then perform a final `AllToAll` to route the tokens back to their original sequence position.

> **Why this matters:** MoE allows us to scale parameters to the trillions without exploding the compute cost, but it turns the network interconnect into a bottleneck. The efficiency of that `AllToAll` shuffle often determines the training speed.

---

### **Loss**

Calculating Cross Entropy when the vocab is sharded (V/vp) is non-trivial. You cannot simply take a softmax because the denominator requires a global sum.

{{< figure src="./loss_parallel.svg" width="400px" align="center" >}}

**The Strategy:** Vocab Parallel (VP)

**The Story:**
1.  **Logits:** We compute the final logits using a Local Unembed (Vocab Parallel). This gives us logits that are sliced by the vocabulary dimension.
2.  **Online Softmax:** To compute softmax, we need the max (for stability) and the sum (for the denominator). We run `AllReduce(max)` and `AllReduce(sum)` across the vocab group to get these global values.
3.  **Target Masking:** This is the tricky part. The ground-truth target label for a token exists on *one* specific GPU, but the logits are scattered everywhere. We mask the non-local logits and use an `AllReduce(sum)` to broadcast the correct target logit to everyone in the group, allowing the loss calculation to proceed.

---

## **Closing Thoughts**

These diagrams represent a snapshot of a common, representative setup found in modern frameworks like **Megatron-Core** or **NVIDIA NeMo**. It is not the only way to do it.

The challenge in infrastructure is making these parallelisms easy to understand, debug, and compose without sacrificing the performant timing of those critical collectives. Hopefully, these visual maps help you navigate the territory.

**A Note on FSDP:**
You might notice **FSDP (Fully Sharded Data Parallel)** is intentionally left out. This is because FSDP is fundamentally a memory optimization (sharding weights and optimizer states at rest) rather than a computation parallelism. FSDP gathers the full weights just-in-time for the forward/backward pass, meaning it does not affect the local tensor shapes that participate in the actual compute operations. I might cover FSDP and its interaction with these parallelisms in a follow-up post.
