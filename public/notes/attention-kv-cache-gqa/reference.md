# Transformer Attention, KV Cache, Prefill, Decode & GQA

*An illustrated, worked technical reference for LLM inference.*

# Reading map

This reference builds the concepts in dependency order: **tokens → hidden states → the four learned attention matrices → projection and reshape → head-wise attention → causal masking → KV cache → prefill/decode → GQA → performance implications**. Thirteen figures and several dimension-checked examples make each stage traceable. All numerical model dimensions are *illustrative configurations*, not claims about a specific checkpoint.

**Viewing note:** This Markdown uses standard relative image paths. Open it from the extracted source bundle with the `attention_kv_cache_figures/` folder alongside it. For a single-file version with all 13 images embedded, use the accompanying HTML reference.

**One-sentence summary:** During autoregressive inference, the current Query selects information from Keys and Values for the current and preceding positions; a KV cache preserves the preceding K/V tensors so that future decode steps need not recompute them.

# Tokens, embeddings, and hidden states

## What is a 4,096-dimensional token vector?

A tokenizer maps text into token IDs. An embedding lookup converts each token ID into a dense numerical representation. For a decoder with hidden size $d_{model}=4096$, each token's embedding and each layer's hidden state have **4,096 scalar components**. These are learned coordinates, not one named human concept per coordinate.

Imagine the input text **“The cat sat”** is represented as three tokens for this teaching example. Real tokenizers may split text differently.

A toy four-dimensional version might be:

| Token | Illustrative vector |
|:--|:--|
| The | $[0.2,-0.5,0.8,0.1]$ |
| cat | $[0.9,0.3,-0.2,0.7]$ |
| sat | $[-0.1,0.6,0.4,0.3]$ |

For the actual **4,096-dimensional** example, we write the input hidden states as a matrix with one row per position:

$$X=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\in\mathbb{R}^{3\times4096}. $$

The first layer receives embedding-based representations, while later layers receive the preceding layer's transformed hidden states. The hidden size is **not** the number of tokens, the number of attention heads, or the KV-cache size.

# The role of attention within one transformer layer

A simplified pre-normalized decoder layer consists of: (1) normalization; (2) linear Q/K/V projections; (3) causal self-attention and output projection; (4) residual addition; (5) a feed-forward network (MLP) with normalization and residual addition. Implementations vary in the ordering and specifics of these suboperations.

![Figure 1. Conceptual dataflow of a decoder transformer layer and its per-layer KV cache.](/notes/attention-kv-cache-gqa/figures/01_transformer_layer.png)

*Figure 1. Conceptual dataflow of a decoder transformer layer and its per-layer KV cache.*
**Attention answers:** “For the current token position, which accessible token positions contain useful information, and how should it be combined?” A Query identifies what the position seeks; Keys determine matching scores; Values contain information that is aggregated. These roles are learned numerical operations, not literal questions and answers.

## Q, K, and V are linear projections

For a sequence of hidden states $X$, a layer computes:

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

The matrices $W_Q,W_K,W_V$ are model parameters learned during training. **They are not the KV cache.** The projected K and V activations, for actual tokens and positions, become cache entries.

We will use this internally consistent **GQA configuration** throughout:

| Parameter | Symbol | Value |
|:--|:--|--:|
| Number of prompt tokens | $S$ | 3 initially |
| Hidden-state dimension | $d_{model}$ | 4,096 |
| Query heads | $H_Q$ | 32 |
| KV heads | $H_{KV}$ | 8 |
| Dimension per head | $D$ | 128 |
| Transformer layers | $L$ | 32 |
| KV precision | $P$ | 2 bytes (BF16) |

## Full projection shapes versus a single head

With three tokens, $X$ has shape $3\times4096$. For the configuration above:

| Operation | Weight shape | Entire projected result | Reshape for attention |
|:--|:--|:--|:--|
| $Q=XW_Q$ | $4096\times4096$ | **$3\times4096$** | $3\times32\times128$ |
| $K=XW_K$ | $4096\times1024$ | $3\times1024$ | $3\times8\times128$ |
| $V=XW_V$ | $4096\times1024$ | $3\times1024$ | $3\times8\times128$ |

![Figure 2. The three projections act independently on the same hidden-state matrix.](/notes/attention-kv-cache-gqa/figures/02_qkv_shapes.png)

*Figure 2. The three projections act independently on the same hidden-state matrix.*
**Common point of confusion:** $Q=XW_Q$ is indeed **$3\times4096$**, *not* $3\times128$. The latter is the **slice for a single Query head** after reshaping the complete projected Query tensor. Likewise, one K or V head is $3\times128$; the **complete** K tensor spans eight such KV heads.

In GQA, four Query heads use one shared KV head, since $H_Q/H_{KV}=32/8=4$. Each Query head remains independent, so it produces different attention weights even when its K/V are shared.

## The four learned attention weight matrices (WQ, WK, WV, WO)

There are **four learned projection matrices per standard attention module**, each with a different role. They are matrices, *not metrics*. $W_Q,W_K,W_V$ project the same input hidden states in **parallel**; $W_O$ combines the outputs of the Query heads **after** attention.

| Matrix | Purpose | Shape in 32-Q / 8-KV GQA | Parameters |
|:--|:--|:--|--:|
| $W_Q$ | Produce all Query features | $4096\times4096$ | 16,777,216 |
| $W_K$ | Produce all Key features | $4096\times1024$ | 4,194,304 |
| $W_V$ | Produce all Value features | $4096\times1024$ | 4,194,304 |
| $W_O$ | Mix concatenated per-Query-head outputs | $4096\times4096$ | 16,777,216 |
| **Total** | **Attention projections only** | | **41,943,040** |

![Figure 3. How WQ, WK, WV, and WO work together within one attention module.](/notes/attention-kv-cache-gqa/figures/03_four_weight_matrices.png)

*Figure 3. How WQ, WK, WV, and WO work together within one attention module.*
These numbers omit optional biases and do not include MLP, embeddings, or normalization weights. Every transformer layer has its **own** attention projection matrices. Weights are fixed during ordinary inference, while **K/V cache entries are generated for the actual token sequence**. The model reuses $W_K$ to compute a new $K_t$ for each token; it does not store the matrix $W_K$ *inside* the KV cache.

| Aspect | Model weight matrices | KV cache |
|:--|:--|:--|
| Contents | Learned parameters such as $W_Q,W_K,W_V,W_O$ | Computed $K_i,V_i$ for each sequence and layer |
| Lifetime | Same model weights across ordinary inference requests | Request-/prefix-specific runtime state |
| Updated per token? | No | Yes: append current $K_t,V_t$ |
| Memory scaling | Depends on model architecture | Grows with cached token count and concurrency |


## Why reshape the projected tensor into separate heads?

The complete Query projection has $3\times4096$ values, but attention is calculated **independently for 32 Query heads**. After projection:

$$Q:(3,4096)\xrightarrow{\text{reshape}}(3,32,128).$$

For one token, dimensions 1–128 of its projected Query form head 1; 129–256 form head 2; and so on. Each head's slice contains 128 values. Corresponding K/V projections reshape as:

$$K,V:(3,1024)\xrightarrow{\text{reshape}}(3,8,128).$$

![Figure 4. The complete Query vector is partitioned into head-specific slices; GQA uses fewer K and V heads.](/notes/attention-kv-cache-gqa/figures/04_reshape_attention_heads.png)

*Figure 4. The complete Query vector is partitioned into head-specific slices; GQA uses fewer K and V heads.*
**Why do this?** Each Query head must produce its **own softmax attention-weight distribution**. One 4,096-dimensional attention head would produce just one score matrix/softmax per sequence; 32 independent 128-dimensional heads produce **32 different attention-weight matrices**. A later output projection can mix what these heads learn. The projections' learned columns create different head-specific features; reshape just reorganizes those features. It does **not** change any scalar values or compress the tensor.

The projection could equivalently be implemented as 32 smaller matrices $W_Q^{(h)}\in\mathbb{R}^{4096\times128}$, then stacked. Computing one large $XW_Q$ and reshaping is a convenient equivalent implementation. Attention runtimes commonly transpose to a head-first tensor layout $[B,H,S,D]$ for parallel head-wise kernels; this transpose is a layout operation too.

## End-to-end attention formula and shapes for GQA

For Query head $h$, let $g(h)$ be its assigned KV head. For zero-indexed heads in this example, $g(h)=\lfloor h/4\rfloor$ because $32/8=4$ Query heads share each KV head. Then:

$$\boxed{O_h=\operatorname{softmax}\left(\frac{Q_hK_{g(h)}^\top}{\sqrt{128}}+M\right)V_{g(h)}}$$

All 32 outputs have shape $3\times128$. They are concatenated and projected:

$$\boxed{Y=\operatorname{Concat}(O_1,\ldots,O_{32})W_O}$$

$$\underbrace{(3,32,128)}_{\text{head outputs}}\longrightarrow\underbrace{(3,4096)}_{\text{concatenated}}\;\underbrace{(4096,4096)}_{W_O}\longrightarrow\underbrace{(3,4096)}_{Y}.$$

Thus the full output has the same model width as the input and can be added to the residual stream. **Q/K/V projection, reshape, per-head attention, concatenation, and $W_O$ together constitute the attention module**; the layer also includes normalization, residual connections, and an MLP.

# Scaled dot-product causal attention

For **one Query head**, the general scaled dot-product attention equation is:

$$\boxed{\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right)V}$$

Here $M$ is the causal mask: inaccessible future-token scores receive a sufficiently negative value, mathematically $-\infty$, before softmax. When considering only keys up to the current position, there are no future positions and the mask may be omitted from the displayed equation. The $1/\sqrt D$ factor scales dot products; softmax converts each row of scores into attention weights summing to one.

## Matrix multiplication for three prompt positions

For one attention head, the matrices have shape $Q,K,V\in\mathbb{R}^{3\times128}$. Thus:

$$\underbrace{Q}_{3\times128}\underbrace{K^\top}_{128\times3}=\underbrace{QK^\top}_{3\times3}.$$

Expanding the scores:

$$QK^\top=\begin{bmatrix}
Q_1K_1^\top & Q_1K_2^\top & Q_1K_3^\top\\
Q_2K_1^\top & Q_2K_2^\top & Q_2K_3^\top\\
Q_3K_1^\top & Q_3K_2^\top & Q_3K_3^\top
\end{bmatrix}.$$

A decoder uses **causal** attention: Query 1 may attend only to position 1; Query 2 to positions 1–2; Query 3 to positions 1–3.

![Figure 5. The lower-triangular causal mask for three positions; the highlighted row is token 3's attention.](/notes/attention-kv-cache-gqa/figures/05_causal_mask.png)

*Figure 5. The lower-triangular causal mask for three positions; the highlighted row is token 3's attention.*
The attention output for all three tokens is $O\in\mathbb{R}^{3\times128}$ for one head. Each of the **32 Query heads** produces such an output. These are concatenated into a $3\times4096$ representation and passed through an output projection (typically $4096\times4096$ in this configuration).

## Derive the single-token equation from the general equation

If we only need token 3 (“sat”), use the **third Query row**:

$$\boxed{O_3=\operatorname{softmax}\left(\frac{Q_3[K_1,K_2,K_3]^\top}{\sqrt D}\right)[V_1,V_2,V_3].}$$

The shapes, for **one head**, are:

$$\underbrace{Q_3}_{1\times128}\underbrace{[K_1,K_2,K_3]^\top}_{128\times3}
\ \rightarrow\ \underbrace{\text{scores}}_{1\times3}
\ \rightarrow\ \underbrace{[V_1,V_2,V_3]}_{3\times128}
\ \rightarrow\ \underbrace{O_3}_{1\times128}.$$

The bracket notation $[K_1,K_2,K_3]$ means the **stack of row vectors**, not a single concatenated row vector. The earlier general equation and this single-token equation describe the *same operation*. The latter merely calculates one row of the attention output.

## Fully worked numeric example

Use an artificially small $D=2$ to make the numbers visible:

$$Q_{cat}=[2,0],\quad K_{The}=[1,0],\quad K_{cat}=[0,1].$$

Scaled Query–Key scores are:

$$s=\left[\frac{2}{\sqrt2},\frac{0}{\sqrt2}\right]\approx[1.414,0].$$

Softmax produces attention weights $a\approx[0.804,0.196]$. If:

$$V_{The}=[10,0],\qquad V_{cat}=[0,20],$$

then:

$$O_{cat}=0.804[10,0]+0.196[0,20]\approx\boxed{[8.04,3.92]}.$$

This result is a **weighted combination of Value vectors**. The example values and token attention pattern are invented only to demonstrate arithmetic.

# Why causal attention enables a KV cache

The KV cache exists because in a decoder-only *causal* transformer, appending a new token does not alter hidden states already computed at earlier positions. Previous tokens cannot attend to later ones. Their K/V representations for a given layer therefore stay valid for subsequent steps, provided the model, positional conventions and earlier context remain unchanged.

For token $t$, current-head attention needs:

$$O_t=\operatorname{softmax}\left(\frac{Q_t[K_1,\ldots,K_t]^\top}{\sqrt D}\right)[V_1,\ldots,V_t].$$

**What is reused?** $K_1\ldots K_{t-1}$ and $V_1\ldots V_{t-1}$.

**What is newly calculated?** $Q_t,K_t,V_t$, based on the current hidden state at that layer. New $K_t,V_t$ are appended to the per-layer cache.

**Why is Q not cached?** Past Queries $Q_1\ldots Q_{t-1}$ were needed to calculate past outputs. The new output formula has **$Q_t$ but no past $Q_i$ terms**. Keeping those Queries would consume memory without helping standard next-token decoding.

| Representation | Role in attention | Keep for future decode? |
|:--|:--|:--|
| Past Query $Q_i$ | Produced the output at past position $i$ | **No** |
| Past Key $K_i$ | Is matched by later Query vectors | **Yes** |
| Past Value $V_i$ | Is mixed into later attention outputs | **Yes** |

The cache also remains **layer-local**: layer $\ell$ has its own projected $K^{(\ell)},V^{(\ell)}$ from that layer's hidden states. A single cache cannot be reused indiscriminately across all layers.

# Prefill and decode: a precise timeline

**Prefill:** The model processes the initial prompt positions together (often with specialized tiled attention kernels). It computes Q/K/V for the prompt in every layer, uses *causal* attention, and creates the prompt KV cache. The final prompt position's output logits yield the **first generated token**.

**Decode:** The next forward pass consumes that newly generated token as input, forms its Q/K/V in each layer, attends against the cached prompt keys/values plus its own K/V, appends its K/V, and predicts the next token. Repeat for later generated tokens.

![Figure 6. Prefill processes multiple Query rows; decode processes one new Query row for each sequence.](/notes/attention-kv-cache-gqa/figures/06_prefill_decode_shapes.png)

*Figure 6. Prefill processes multiple Query rows; decode processes one new Query row for each sequence.*
![Figure 7. Cache contents after “The cat sat” and after the next forward pass that processes “on”.](/notes/attention-kv-cache-gqa/figures/07_cache_evolution.png)

*Figure 7. Cache contents after “The cat sat” and after the next forward pass that processes “on”.*
The phrase **“The cat sat on”** is pedagogical: after prefill of “The cat sat”, suppose the model predicts **“on”**. The cache initially holds **The / cat / sat**; the *following* decode step processes **on** and appends its K/V. Thus it is more precise to say “first token sampled after prefill” than “prefill processes the generated token.”

## Decode step: computing attention for “on”

In the running example, the prompt `The cat sat` is processed during prefill and the model predicts `on`. On the **next** forward pass, the model processes `on` as input. At each layer, it computes **new** $Q_4,K_4,V_4$ and reads the **old** $K_1,K_2,K_3$ and $V_1,V_2,V_3$ from that layer's cache. It does not need $Q_1,Q_2,Q_3$ again.

![Figure 8. How Q4 matches all four Keys, softmax determines attention weights, and Values are combined into O4.](/notes/attention-kv-cache-gqa/figures/08_decode_on_attention.png)

*Figure 8. How Q4 matches all four Keys, softmax determines attention weights, and Values are combined into O4.*
For one Query head:

$$s_i=\frac{Q_4K_i^\top}{\sqrt{128}},\qquad (a_1,a_2,a_3,a_4)=\operatorname{softmax}(s_1,s_2,s_3,s_4).$$

$$\boxed{O_4=a_1V_1+a_2V_2+a_3V_3+a_4V_4.}$$

The labels $K_i,V_i$ in this head-wise formula mean the **KV head assigned to the particular Query head**. Other Query heads may use the same stored K/V but calculate their own weights.

## Prefill versus decode: the explicit attention equations

**Prefill, three prompt tokens, one Query head.** The Query, Key, and Value matrices each contain three rows. The attention-score matrix has shape $3\times3$ and requires a causal mask $M$:

$$\boxed{O_{1:3}=\operatorname{softmax}\left(\frac{Q_{1:3}K_{1:3}^{\top}}{\sqrt{128}}+M\right)V_{1:3}.}$$

$$\underbrace{(3\times128)}_Q\;\underbrace{(128\times3)}_{K^\top}\longrightarrow\underbrace{(3\times3)}_{\text{scores}}\;\underbrace{(3\times128)}_V\longrightarrow\underbrace{(3\times128)}_O.$$

**Decode, fourth token, one Query head.** Only the current Query $Q_4$ is computed, while K/V include the three cached tokens and the current one:

$$\boxed{O_4=\operatorname{softmax}\left(\frac{Q_4[K_1,K_2,K_3,K_4]^{\top}}{\sqrt{128}}\right)[V_1,V_2,V_3,V_4].}$$

$$\underbrace{(1\times128)}_{Q_4}\;\underbrace{(128\times4)}_{K^\top}\longrightarrow\underbrace{(1\times4)}_{\text{scores}}\;\underbrace{(4\times128)}_V\longrightarrow\underbrace{(1\times128)}_{O_4}.$$

The square-bracketed K/V notation denotes **row-stacked vectors**, not vectors concatenated along their features. For GQA, these computations occur for each of the 32 Query heads, each accessing one of eight KV heads. During prefill the causal mask excludes future keys; in single-token decode all four available positions are valid, so no future-position mask is shown.

![Figure 9. Side-by-side prefill and decode shapes, cache operations, and outputs.](/notes/attention-kv-cache-gqa/figures/09_prefill_vs_decode_attention.png)

*Figure 9. Side-by-side prefill and decode shapes, cache operations, and outputs.*
## Same equation, different tensor sizes

For one Query head, with a three-token prompt and then the fourth token:

| Item | Prefill: positions 1–3 | Decode: position 4 |
|:--|:--|:--|
| Q | $3\times128$ | $1\times128$ |
| K, V available for attention | Each $3\times128$ | Each $4\times128$ (3 cached + 1 new) |
| Attention scores | $3\times3$ (causally masked) | $1\times4$ |
| Output | $3\times128$ | $1\times128$ |
| Cache activity | Build prompt KV | Read prior KV and append new KV |

For the fourth position:

$$\boxed{O_4=\operatorname{softmax}\left(\frac{Q_4[K_1,K_2,K_3,K_4]^\top}{\sqrt D}\right)[V_1,V_2,V_3,V_4].}$$

The attention equation has **not changed**. Only the number of Query rows and the source/number of KV rows have changed. In GQA, this computation is performed separately per Query head, with each group sharing a K/V head.

## Compute consequences

For one head and sequence length $S$, conventional full self-attention computes $S\times S$ Query–Key scores during prefill: roughly $O(S^2D)$ attention arithmetic. A single new decode Query compares against $S$ available Keys: roughly $O(SD)$ attention arithmetic *for that new token*.

These complexities refer to the **attention suboperation**; projection/MLP work, batching, kernel tiling and the full model have additional costs. A modern FlashAttention-style prefill need not materialize the entire $S\times S$ scores in HBM.

Prefill is often relatively compute-intensive, whereas long-context decode is frequently constrained by repeated reads of model weights and cached KV, especially at relevant batching levels. **TTFT** (time to first token) contains queueing, prefill and other overheads; **TPOT** (time per output token) characterizes generation latency after the first token.

# KV-cache memory: derive it from first principles

One token, one transformer layer, with $H_{KV}$ KV heads and head dimension $D$ stores:

- $H_{KV}\times D$ **Key elements**.
- $H_{KV}\times D$ **Value elements**.
- Each element occupies $P$ bytes.

Thus the logical memory formula is:

$$\boxed{M_{KV}=2\times L\times B\times S\times H_{KV}\times D\times P}$$

where $L$ is layers, $B$ is number of sequences of equal cached length, $S$ is tokens cached per sequence, $H_{KV}$ is KV heads, $D$ is elements per K/V head, and $P$ is bytes per element. For unequal sequence lengths, replace $B\times S$ with $\sum_b S_b$.

## Worked example: 32 layers, GQA, 8,192 tokens, BF16

Use $L=32$, $B=1$, $S=8192$, $H_{KV}=8$, $D=128$, $P=2$.

1. **One K head per token per layer:** $128\times2=256$ bytes.
2. **One V head:** another 256 bytes.
3. **All eight KV heads:** $8\times(256+256)=4096$ bytes = **4 KiB** per token per layer.
4. **All 32 layers:** $32\times4\text{ KiB}=128\text{ KiB}$ per token.
5. **At 8,192 cached tokens:** $8192\times128\text{ KiB}=\boxed{1\text{ GiB}}$ for one sequence.
6. **At batch size 32**, assuming each sequence has 8,192 cached tokens: **32 GiB** logical KV data.

The hidden state is **4,096 numerical dimensions**. Separately, the computed K/V storage here is **4,096 bytes** per token per layer. Those equal-looking numbers refer to different units and their equality is incidental.

Actual allocated memory may differ because of page/block granularity, fragmentation, reservations, sliding-window or other attention variants, concurrency, cache eviction, prefix sharing, replication or distribution across devices. This formula estimates the logical **full-history** KV tensor payload.

# Multi-head, grouped-query, and multi-query attention

Three attention designs vary mainly in **how many K/V heads the model learns and stores** relative to Query heads.

| Design | $H_Q$ | $H_{KV}$ | Query heads per KV head | Logical KV vs MHA |
|:--|--:|--:|--:|--:|
| Multi-Head Attention (MHA) | 32 | 32 | 1 | 100% |
| Grouped-Query Attention (GQA) | 32 | 8 | 4 | **25%** |
| Multi-Query Attention (MQA) | 32 | 1 | 32 | **3.125%** |

![Figure 10. MHA, GQA and MQA illustrated with eight Queries.](/notes/attention-kv-cache-gqa/figures/10_mha_gqa_mqa.png)

*Figure 10. MHA, GQA and MQA illustrated with eight Queries.*
For GQA, **four independent Q heads share one K head and one V head**, but each Query obtains its **own attention weights**. These heads are not averaged into one Query. Efficient implementations avoid physically replicating the shared K/V tensors in the persistent cache.

The reduction compared with an otherwise identical MHA design is:

$$\text{KV memory reduction factor}=\frac{H_Q}{H_{KV}}=\frac{32}{8}=\boxed{4\times}. $$

## Why shared K/V does not make Query heads identical

Consider a toy $D=2$ example with shared Keys $K_1=[1,0]$, $K_2=[0,1]$. Two Query heads have different vectors $Q_A=[2,0]$ and $Q_B=[0,2]$. Their scaled scores are $[\sqrt2,0]$ and $[0,\sqrt2]$, respectively. Softmax therefore produces different attention distributions: approximately **[0.804, 0.196]** and **[0.196, 0.804]**. Both heads can use the **same cached Keys and Values** yet output different weighted Value combinations.

![Figure 11. Two independent Query heads use one shared KV head but generate different attention weights.](/notes/attention-kv-cache-gqa/figures/11_shared_kv_distinct_weights.png)

*Figure 11. Two independent Query heads use one shared KV head but generate different attention weights.*
For the earlier 32-layer, 8K-token example:

![Figure 12. Theoretical logical KV storage for MHA, GQA and MQA under equal layer, head-dimension, context and precision assumptions.](/notes/attention-kv-cache-gqa/figures/12_kv_memory_comparison.png)

*Figure 12. Theoretical logical KV storage for MHA, GQA and MQA under equal layer, head-dimension, context and precision assumptions.*
- **MHA, 32 KV heads:** 4 GiB / sequence.
- **GQA, 8 KV heads:** 1 GiB / sequence.
- **MQA, 1 KV head:** 128 MiB / sequence.

These figures are **KV storage reductions**, not promises of 4× or 32× overall decoding speed. Actual speed is affected by model-weight traffic, attention kernels, bandwidth, batching, compute, quantization and scheduler behavior.

GQA is fundamentally a **model architecture and checkpoint parameterization**. You cannot simply change $H_{KV}$ of an already trained MHA checkpoint at runtime and expect equivalent outputs. Converting checkpoints requires suitable modification of projections and typically additional adaptation/training; the original GQA paper presents an uptraining recipe.

# Why the KV cache influences inference throughput

**Benefits of caching:** It removes repeated computation of past-token hidden states and their Q/K/V projections on each autoregressive step. It is therefore essential to efficient token-by-token generation.

**Its remaining cost:** At each step, the current Query still needs attention over accessible past K/V positions, so their representations are repeatedly consumed. Depending on the workload, this results in substantial HBM traffic. Larger contexts and more concurrent sequences mean more live KV data.

**What GQA changes:** It stores and reads fewer distinct K/V head vectors for a given token history. This improves feasible context/concurrency and may reduce memory-bandwidth pressure during decode. It does not shrink the number of historical positions or eliminate attention arithmetic.

Related methods target different costs:

| Technique | Primary mechanism | What it does **not** mean |
|:--|:--|:--|
| GQA / MQA | Fewer distinct K/V heads | Not runtime-only lossless compression of arbitrary MHA weights |
| KV quantization | Fewer bytes per K/V element | Not fewer token positions by itself |
| PagedAttention | KV cache allocation, paging, and potential block sharing | Does not reduce the theoretical number of K/V values in a full-history attention model |
| FlashAttention | Tiled, IO-aware exact attention computation | Does not make persistent decode KV storage disappear |
| Prefix caching | Reuse K/V for identical cached prefixes | Not automatic reuse for unrelated prefixes |
| KV offload / eviction | Move or release GPU-resident KV | May add transfer cost or require recomputation |

# Prefill/decode disaggregation and KV transfer

In a disaggregated design, one accelerator performs prefill while a different accelerator performs subsequent decode. Instead of transferring the prompt Queries or every hidden state, the decode worker generally needs **compatible per-layer prompt K/V tensors**, together with essential metadata (such as positions, layer/head mapping, data type and layout). The decode worker then computes new Q/K/V tensors for later tokens.

![Figure 13. Conceptual handoff of per-layer KV data between prefill and decode runtimes.](/notes/attention-kv-cache-gqa/figures/13_disaggregated.png)

*Figure 13. Conceptual handoff of per-layer KV data between prefill and decode runtimes.*
For the 8K-token example, GQA reduces the **logical** KV payload from 4 GiB (32 KV heads) to 1 GiB (8 KV heads), provided other parameters are held equal. Real transfer volume and latency may differ due to data distribution across devices, transport framing, paging, dtype conversion, transfer overlap and runtime requirements.

**Cross-vendor caveat (e.g., NVIDIA prefill → another accelerator's decode):** Sharing architecture and the name “KV cache” is not enough to guarantee a working handoff. Both runtimes must agree on model weights, layer/head semantics, RoPE positional conventions, KV layout/order, dtype/quantization/scales, and how cache tokens map to physical buffers. Support for actual transfer/import APIs must be established separately.

# Practical shape-checking example

The following pseudocode is **shape-oriented**, not a full implementation (normalization, RoPE, causal masking, fused kernels, residuals and batch axes are omitted):

```python
# One decoder layer; illustrative GQA dimensions
S, d_model, Hq, Hkv, D = 3, 4096, 32, 8, 128
X = hidden_states                 # [S, 4096]
Q = (X @ Wq).reshape(S, Hq, D)    # [3, 32, 128]
K = (X @ Wk).reshape(S, Hkv, D)   # [3,  8, 128]
V = (X @ Wv).reshape(S, Hkv, D)   # [3,  8, 128]

# On prefill: per-head attention with a causal mask; save K, V.
cache_K, cache_V = K, V

# At a later decode step, X_new represents ONE current token.
q_new = (X_new @ Wq).reshape(1, Hq, D)
k_new = (X_new @ Wk).reshape(1, Hkv, D)
v_new = (X_new @ Wv).reshape(1, Hkv, D)
cache_K = append(cache_K, k_new)  # [4, 8, 128]
cache_V = append(cache_V, v_new)  # [4, 8, 128]

# Each q_new head h attends to KV head (h // 4).
# In optimized GQA, shared KV are NOT stored four times.
```

## Eight correctness checks

1. $Q=XW_Q$ is **$S\times(H_QD)$** in the example, before head reshaping.
2. One Query head is **$S\times D$** after reshaping.
3. One GQA K/V head is also **$S\times D$**, but there are fewer KV heads.
4. The attention score matrix for one Query head has shape **query length × accessible key length**.
5. During causal prefill, future positions are masked.
6. Decode reuses **past K/V**, not past Q, and appends **new K/V**.
7. **Every transformer layer** maintains its own KV content.
8. A fourfold KV-memory reduction is **not** necessarily a fourfold reduction in TPOT.

# Glossary and notation

| Term | Meaning |
|:--|:--|
| Token | Unit of text represented by a tokenizer ID; not necessarily a whole word |
| Embedding / hidden state | Learned dense vector representing a token at a model stage |
| $d_{model}$ | Hidden-state width, e.g. 4,096 |
| $D$ / head dimension | Number of components in one Q, K or V head, e.g. 128 |
| $H_Q$ | Number of independent Query heads |
| $H_{KV}$ | Number of Key heads and corresponding Value heads |
| $S$ | Sequence length or cached token count |
| $L$ | Number of transformer layers |
| $B$ | Number of concurrently cached sequences |
| $P$ | Bytes per cached K/V element |
| KV cache | Per-layer stored Key and Value activations for past positions |
| Prefill | Initial processing of supplied prompt tokens |
| Decode | Subsequent autoregressive processing of generated tokens |
| TTFT | Time to first token |
| TPOT | Time per output token after the first token |
| MHA / GQA / MQA | Multi-Head / Grouped-Query / Multi-Query Attention |

# References and further reading

1. Vaswani et al., **Attention Is All You Need** (2017). Original scaled dot-product and multi-head attention formulation. <https://arxiv.org/abs/1706.03762>
2. Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (EMNLP 2023). GQA architecture and checkpoint adaptation. <https://aclanthology.org/2023.emnlp-main.298/>
3. Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention** (SOSP 2023). KV-cache blocks, memory management, and vLLM serving. <https://doi.org/10.1145/3600006.3613165>
4. Dao et al., **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (NeurIPS 2022). IO-aware attention kernels. <https://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html>

*Scope note:* Numerical attention weights and the model dimensions in this tutorial are worked examples. Actual models may use distinct head sizes, local/sliding-window attention, RoPE variants, MLA, special cache layouts, or KV quantization; adapt the equations to the model architecture and runtime implementation being investigated.
