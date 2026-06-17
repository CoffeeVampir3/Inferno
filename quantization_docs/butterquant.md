# ButterQuant Reference

## Int8 Encoding and Dequantization Identities for Hadamard-Domain Operators

This document specifies an int8 quantization scheme that operates on top of the Hadamard-domain linear algebra of the Hadamard Arithmetic Reference (HAR). HAR establishes which operations preserve which Hadamard bases. This document specifies the int8 encoding of activations and weights, the dequantization identities at operator boundaries, and the recipes that match each operator class.

This is a specification of *encodings and identities*, not a procedure. Where an order of operations is stated, it is because a non-commutation (HAR §6.4) or the basis in which a quantity is defined forces that order; such cases are called out explicitly. Operations that are not so constrained may be applied in any order, and no ordering should be inferred from the order of presentation.

The conventions of HAR are inherited. $\mathcal{H}_d$ is the normalized block-diagonal Hadamard with block size $n$ such that $n \mid d$. $\mathcal{Q}$ is a coordinatewise quantization-reconstruction operator. $\tilde x = \mathcal{H}_d x$.

---

# I. The Symmetric Int8 Quantizer

## 1.1 Definition

For $S > 0$, the symmetric int8 quantizer $Q_S : \mathbb{R} \to \{-128, \ldots, 127\}$ is

$$Q_S(x) = \mathrm{clamp}(\mathrm{round}(127 x / S),\ -128,\ 127).$$

The coordinatewise reconstruction is $\hat x = Q_S(x) \cdot S / 127$.

## 1.2 Reconstruction error

For $|x| \le S$,

$$|\hat x - x| \le S / 254$$

under round-to-nearest-even. For $|x| > S$, the reconstruction saturates at $\mathrm{sign}(x) \cdot S$ and the error is $|x| - S$.

## 1.3 Dynamic absmax

Let $\Omega \subseteq \{1, \ldots, n\}$ be a coordinate support. The dynamic absmax over $\Omega$ for $x \in \mathbb{R}^n$ is

$$S_\Omega(x) = \max_{i \in \Omega} |x_i|.$$

For any $S' < S_\Omega$, at least one coordinate $|x_i|$ with $i \in \Omega$ saturates under $Q_{S'}$. The dynamic absmax is the smallest scale that admits no saturation over $\Omega$.

A positive floor $\epsilon_S$ may be applied: $S = \max(S_\Omega, \epsilon_S)$. The floor is inactive when $S_\Omega > \epsilon_S$ and avoids division by zero when $S_\Omega = 0$.

## 1.4 The u8/i8 affine relation

For $i \in \{-128, \ldots, 127\}$,

$$u = i + 128 = i \oplus \mathtt{0x80} \in \{0, \ldots, 255\}.$$

For $W \in \{-128, \ldots, 127\}^{N \times K}$ and $i \in \{-128, \ldots, 127\}^K$,

$$\sum_k (i_k + 128) \cdot W_{n,k} = \sum_k i_k \cdot W_{n,k} + 128 \cdot \sum_k W_{n,k}.$$

Define the colsum

$$\mathrm{cs}[n] = \sum_k W_{n,k}.$$

Then

$$\sum_k i_k \cdot W_{n,k} = \sum_k (i_k + 128) \cdot W_{n,k} - 128 \cdot \mathrm{cs}[n].$$

The signed-signed dot product is recoverable from the unsigned-signed dot product via subtraction of one per-output scalar. For per-block colsum on block $b$ with support $\Omega_b \subseteq \{1, \ldots, K\}$,

$$\mathrm{cs}[n, b] = \sum_{k \in \Omega_b} W_{n,k}.$$

## 1.5 Hardware operand constraints

This is a hardware-aware encoding: the int8 storage form of each operand follows the operand-sign requirements of the dot-product instruction that consumes it. Different SIMD/tile dot-product instructions impose different requirements:

| Instruction | A operand | B operand |
|---|---|---|
| `vpdpbusd` (AVX-512 VNNI) | u8 | i8 |
| `vpdpbssd` (AVX-VNNI-INT8) | i8 | i8 |
| `tdpbsud` (AMX) | i8 (tile) | u8 (tile) |
| `tdpbssd` (AMX) | i8 (tile) | i8 (tile) |

Where the dot product instruction takes one unsigned operand and that operand encodes signed data via the affine relation of §1.4, the signed-signed inner product is recovered by subtracting $128$ times the sum of the operand that remains signed, taken over the contraction index. The location of this correction depends on which operand is unsigned: where the activation is encoded as $u_8$ and the weight stays $i_8$, the correction is $128 \cdot \sum_k W_{i_8}[n, k]$ (one value per output row, the weight colsum); where one operand of a score-style dot is encoded as $u_8$ and the other stays $i_8$, the correction is $128$ times the sum of the signed operand over the contraction index (one value per output of that operand).

Where the unsigned operand encodes non-negative data directly (without the $+128$ offset, e.g., a u8 value in $[0, 255]$ representing a non-negative quantity), no such correction is applied. Where the instruction admits a signed-signed form, both operands are consumed as $i_8$ and no correction is applied.

The dequantized identity is the same in all forms. The choice of operand sign at each kernel site is a property of the available instruction and the data range of each operand: it selects the storage form and the location of any correction, but does not change the dequantized identity.

---

# II. Encodings

## 2.1 Per-row activation encoding

For $x \in \mathbb{R}^K$, the per-row encoding stores

$$x_{i_8}[k] = Q_{S_a}(x_k), \qquad S_a = \max_k |x_k|.$$

Output: one i8 vector of length $K$ and one f32 scalar.

## 2.2 Per-block activation encoding

Partition $\{1, \ldots, K\}$ into $K/B$ blocks of size $B$ with $B \mid K$. For block $b$ with support $\Omega_b$,

$$x_{i_8}[k] = Q_{S_a[b]}(x_k) \text{ for } k \in \Omega_b, \qquad S_a[b] = \max_{k \in \Omega_b} |x_k|.$$

Output: one i8 vector of length $K$ and $K/B$ f32 scales.

Since $\Omega_b \subseteq \{1, \ldots, K\}$, $S_a[b] \le S_a$ for every $b$. The per-block grid step $S_a[b] / 127$ is therefore at most as coarse as the per-row grid step $S_a / 127$ over the same coordinates. The relative magnitudes of the per-block grid steps within a row depend on the within-row energy distribution and are not bounded above by a fixed function of $B$ and $K$. The cost is $K/B$ scales per row in place of one. The choice between per-row and per-block is governed by measurable factors — scale utilization under the data's magnitude structure (§3.6) and the accumulation structure the consuming hardware permits (§7.6) — not by preference.

## 2.3 Weight encodings

For $W \in \mathbb{R}^{N \times K}$, the per-row encoding stores

$$W_{i_8}[n, k] = Q_{S_w[n]}(W_{n, k}), \qquad S_w[n] = \max_k |W_{n, k}|,$$

and the per-block encoding stores

$$W_{i_8}[n, k] = Q_{S_w[n, b]}(W_{n, k}) \text{ for } k \in \Omega_b, \qquad S_w[n, b] = \max_{k \in \Omega_b} |W_{n, k}|.$$

## 2.4 Storage conventions for scales

The stored value of a weight scale is pre-divided by 127:

$$\bar S_w[n] = S_w[n] / 127.$$

The reconstruction $\hat W_{n, k} = W_{i_8}[n, k] \cdot \bar S_w[n]$ is then one multiplication.

The stored value of an activation scale is the raw $S_a$ without pre-division. The consumer applies the $/127$ factor at the dequantization step.

This asymmetry is a storage convention. The dequantized identity is unchanged.

## 2.5 Colsum companion

For each int8 weight consumed by an unsigned-signed GEMV, the colsum companion is computed once after quantization. The colsum's K-axis structure matches the K-block partitioning of the dequantization step, which is determined by the activation scale granularity at the consumer, not by the weight scale granularity:

$$\mathrm{cs}[n] = \sum_k W_{i_8}[n, k] \quad \text{(consumer uses per-row activation scale)},$$

$$\mathrm{cs}[n, b] = \sum_{k \in \Omega_b} W_{i_8}[n, k] \quad \text{(consumer uses per-K-block activation scale)}.$$

A weight with per-row weight scale that feeds a per-block GEMV (per-block activation scale) still requires per-K-block colsums, since the dequantization tail subtracts $128 \cdot \mathrm{cs}[n, b]$ inside the block sum (§7.2).

Kernel sites using a signed-signed dot product produce the same dequantized output without consuming the colsum. The colsum storage is allocated when the int8 weight is allocated; signed-signed paths read the int8 weight without reading the colsum.

---

# III. Hadamard-Rotated Encodings

## 3.1 K-rotated weight, single-sided

For $W \in \mathbb{R}^{N \times K}$, define

$$W^\sharp = W \mathcal{H}_K^T.$$

By HAR §3.2, $W^\sharp \mathcal{H}_K x = W x$ in exact arithmetic.

The per-row or per-block int8 encoding of $W^\sharp$ is computed by applying $\mathcal{H}_K^T$ to the rows of $W$ before quantization. The dequantized matmul output is in the original M-coordinate basis.

## 3.2 K- and M-rotated weight, two-sided

For $W \in \mathbb{R}^{N \times K}$ with $\mathcal{H}_M$ block-diagonal of block size $d$ such that $d \mid N$, define

$$\tilde W = \mathcal{H}_M W \mathcal{H}_K^T.$$

By HAR §3.1, $\tilde W \mathcal{H}_K x = \mathcal{H}_M W x$ in exact arithmetic. The dequantized matmul output is in the M-rotated basis.

The two-sided encoding applies $\mathcal{H}_K^T$ to the rows of $W$ and $\mathcal{H}_M$ to the columns of $W$ (per output-axis block) before quantization.

## 3.3 Equivalence at the K boundary

The runtime activation FWHT for the K axis is identical in single-sided and two-sided forms. The two encodings differ at the M-axis output: single-sided produces $W x$ directly; two-sided produces $\mathcal{H}_M W x$.

The two-sided form moves one FWHT — the M-axis rotation that a consumer operating in the rotated basis would otherwise perform — into the offline weight encoding. The single-sided form leaves that FWHT at the output.

## 3.4 Block-size constraint

When the M-axis rotation $\mathcal{H}_M$ has block size $d$, a consumer that operates in the rotated basis must have a K-axis encoding with the same block size $d$. If the block sizes differ, a basis-change FWHT is required, which is the same FWHT the single-sided encoding would have placed at the output; the offline rotation is then redundant.

For per-block consumers (where the next operation is structured per block of dimension $d$), the M-axis block size is $d$.

## 3.5 Effect on the per-row absmax

By Parseval, $\|H_n x\|_2 = \|x\|_2$. The per-coordinate distribution differs in general. By HAR §1.3, $|(H_n x)_i| \le \|x\|_2$ and $|(H_n x)_i| \le \|x\|_1 / \sqrt n$.

The relationship between the rotated absmax $\max_k |(H_n x)_k|$ and the original absmax $\max_k |x_k|$ is not monotone. For $x$ aligned with a single coordinate ($x = c \cdot e_i$), the rotated absmax is $|c|/\sqrt n$, smaller than the original $|c|$ by factor $\sqrt n$. For $x$ aligned with one row of $H_n$ ($x = c \cdot h_i^T$), the rotated absmax is $|c|$, larger than the original $|c|/\sqrt n$ by factor $\sqrt n$. For inputs in between, the rotated absmax falls in between.

The rotation does not eliminate per-row magnitude variance across rows. Per-row dynamic scales handle inter-row variance independently of the rotation.

## 3.6 Scale axis and utilization

A scale covers the absmax of every value it is shared across, so a value whose local absmax $a$ is smaller than the shared scale $S$ occupies only the fraction $u = a / S \le 1$ of the int8 range. The effective precision loss relative to a scale matched to $a$ is approximately $\log_2(1/u)$ bits. The axis along which a scale is held fixed is therefore a measurable choice, not a free one.

A block-diagonal Hadamard redistributes energy among the coordinates *within* a vector (§3.5), but acts identically and independently on every vector along an index axis. It flattens within-vector coordinate variance and leaves cross-vector magnitude variance along the index axis unchanged. A scale shared across an index axis that carries magnitude variance — for example one fixed scale per coordinate, held constant across rows or across cached positions — must size to the largest-magnitude vector on that axis, leaving every smaller vector at low utilization. The rotation cannot recover this loss, because it does not act across that axis.

The dynamic scale is therefore placed on the axis that carries the magnitude variance: one scale per vector along that axis (per-row, §II; per cached position, §VIII), computed from that vector's own absmax. The rotation handles the orthogonal within-vector variance so that one per-vector scale suffices. A scale shared across an index axis is appropriate only where magnitude variance along that axis is small enough that the resulting utilization is acceptable; otherwise the shared-scale form is measurably lossier at no benefit the rotation can offset.

---

# IV. Gain Split

## 4.1 Decomposition

For $\gamma \in \mathbb{R}^d$ and $\gamma_k \ne 0$,

$$\gamma_k = \mathrm{sign}(\gamma_k) \sqrt{|\gamma_k|} \cdot \sqrt{|\gamma_k|}.$$

For $\gamma_k = 0$, the right-hand side is $0$ when $\mathrm{sign}(0)$ is taken as $0$.

## 4.2 Application to RMSNorm followed by a linear map

Let $W \in \mathbb{R}^{N \times K}$ follow $\mathrm{RMSNorm}_\gamma$ in the operator chain $W \cdot \mathrm{RMSNorm}_\gamma(x)$. Define

$$W'_{n, k} = W_{n, k} \cdot \sqrt{|\gamma_k|},$$

$$x'_k = \frac{x_k}{\mathrm{rms}(x)} \cdot \mathrm{sign}(\gamma_k) \sqrt{|\gamma_k|}.$$

Then

$$\sum_k W'_{n, k} \cdot x'_k = \sum_k W_{n, k} \cdot \frac{x_k}{\mathrm{rms}(x)} \cdot \gamma_k = W \cdot \mathrm{RMSNorm}_\gamma(x).$$

In exact arithmetic, the split factorization equals absorption (HAR §5.4) with $W'' = W \mathrm{diag}(\gamma)$ and no separate gain. The split is a reparameterization of absorption.

## 4.3 Effect on quantization grids

Under absorption, the offline quantizer $Q_{S_{w''}}$ for $W'' = W \mathrm{diag}(\gamma)$ sees per-row dynamic ranges scaled by $\max_k |\gamma_k|$ relative to $W$. The activation quantizer sees $x / \mathrm{rms}(x)$ with no gain factor.

Under split, the offline quantizer for $W' = W \mathrm{diag}(\sqrt{|\gamma|})$ sees per-row dynamic ranges scaled by $\max_k \sqrt{|\gamma_k|}$. The activation quantizer sees $x / \mathrm{rms}(x) \cdot \mathrm{sign}(\gamma) \sqrt{|\gamma|}$, scaled by $\max_k \sqrt{|\gamma_k|}$.

The split distributes $\sqrt{|\gamma|}$ symmetrically across the two quantization grids. Absorption concentrates the full $|\gamma|$ on the offline grid.

## 4.4 Stability floor

The activation-side factor may be evaluated as

$$\sigma_\gamma[k] = \mathrm{sign}(\gamma_k) \sqrt{\max(|\gamma_k|, \epsilon_\gamma)}.$$

For $|\gamma_k| > \epsilon_\gamma$, $\sigma_\gamma[k] = \mathrm{sign}(\gamma_k) \sqrt{|\gamma_k|}$ as in §4.1, and the offline–activation product equals $\gamma_k$.

For $|\gamma_k| \le \epsilon_\gamma$, the product is $\sqrt{|\gamma_k| \cdot \epsilon_\gamma}$ when the offline factor uses $\sqrt{|\gamma_k|}$ without floor. The mismatch is bounded by $\sqrt{\epsilon_\gamma \cdot \max(|\gamma_k|, \epsilon_\gamma)}$ and vanishes at $\gamma_k = 0$ when $\mathrm{sign}(0) = 0$. The floor is applied on the activation side only; the offline factor uses $\sqrt{|\gamma_k|}$ without floor.

## 4.5 Availability

The split is defined when an RMSNorm with gain $\gamma$ immediately precedes a linear map: that is the only configuration in which a $\gamma$ exists to split and a private weight exists to absorb the offline half. It has no meaning otherwise. When the weight that would carry the offline factor is shared or tied to another operator, the offline half cannot be baked in without altering the other use; the full gain then stays on the activation side and the weight is encoded in its un-absorbed form.

---

# V. Single-Sided and Two-Sided Weight Forms

## 5.1 Two encodings of one linear map

For $y = Wx$ with $W \in \mathbb{R}^{N \times K}$, two int8 encodings are available:

- Single-sided: $W^\sharp = W \mathcal{H}_K^T$, dequantized output $y$ in original basis.
- Two-sided: $\tilde W = \mathcal{H}_M W \mathcal{H}_K^T$, dequantized output $\mathcal{H}_M y$ in M-rotated basis.

Both are exact reformulations of $y = Wx$. The cost of the K-axis FWHT on the activation is identical. The output basis differs.

## 5.2 Operations preserving the M-rotated basis

The following operations on $y$ commute with the M-axis Hadamard $\mathcal{H}_M$ and preserve the M-rotated basis through to their output:

- Scalar multiplication: $\mathcal{H}_M(\alpha y) = \alpha \mathcal{H}_M y$.
- Vector addition with another vector in the same M-rotated basis: $\mathcal{H}_M(y + z) = \mathcal{H}_M y + \mathcal{H}_M z$.
- Inner product with a vector in the same M-rotated basis: $\langle \mathcal{H}_M y, \mathcal{H}_M z \rangle = \langle y, z \rangle$.
- Sum over an index axis disjoint from the M axis: $\mathcal{H}_M \sum_i y_i = \sum_i \mathcal{H}_M y_i$.
- A subsequent linear map $W_2$ with K-axis encoding rotated by the same $\mathcal{H}_M$: $W_2^\sharp = W_2 \mathcal{H}_M^T$ accepts $\mathcal{H}_M y$ as its rotated input.

## 5.3 Operations not preserving the M-rotated basis

The following operations do not in general commute with $\mathcal{H}_M$:

- Coordinatewise nonlinearities $\phi(y)$: HAR §6.1, $\mathcal{H}_M \phi(y) \ne \phi(\mathcal{H}_M y)$ in general.
- Coordinatewise gain $\mathrm{diag}(\beta) y$ when $\beta$ is not constant per Hadamard block: $\mathcal{H}_M \mathrm{diag}(\beta) \mathcal{H}_M^T$ is dense (HAR §6.4).
- Position-dependent rotations applied in the original coordinate basis: $\mathcal{H}_M R_p \mathcal{H}_M^T$ is dense (HAR §6.4).
- Reductions whose statistic is not norm-based.
- Residual addition into a vector in the original basis.
- A coordinate-basis output (e.g. an output consumed directly in its own coordinate system, such as a set of logits read by a sampler).

After such an operation, a downstream linear map's encoding starts from a fresh K-axis rotation if int8 is to be re-entered.

## 5.4 Legality of the two-sided form

The two-sided form for $W$ is legal — meaning the encoding requires no basis-change FWHT beyond what is already in the operator chain — if and only if every operation between $W$ and the next operation requiring original coordinates is in the class of §5.2.

When the two-sided form is legal, the M-axis FWHT is applied once offline at weight quantization, in place of one FWHT at the matmul output. When the two-sided form is not legal, using it requires inserting an explicit inverse FWHT before the next operation that requires original coordinates.

The single-sided form requires no basis-change FWHT in either case; it leaves the M-axis FWHT at the output when one is required by the next consumer, and inserts no FWHT when one is not.

A coordinatewise nonlinearity, a non-block-constant gain, or a position-dependent rotation between $W$ and the next consumer therefore disqualifies the two-sided form for $W$: the operation in §5.3 forces a return to original coordinates that the two-sided rotation would have to be undone for.

---

# VI. Per-Row GEMV Dequantization

## 6.1 Inputs

For one activation row $i \in \mathbb{R}^K$ and a per-row int8 weight encoding of $W \in \mathbb{R}^{N \times K}$:

| Quantity | Type | Shape | Storage |
|---|---|---|---|
| $x_{i_8}$ | int8 | $(K,)$ | activation int8 |
| $S_a$ | f32 | $()$ | activation absmax (raw) |
| $W_{i_8}$ | int8 | $(N, K)$ | weight int8 |
| $\bar S_w$ | f32 | $(N,)$ | $S_w / 127$ |
| $\mathrm{cs}$ | f32 | $(N,)$ | per-row weight colsum |

## 6.2 Unsigned-signed form

Define the unsigned-signed dot product

$$r[n] = \sum_k (x_{i_8}[k] + 128) \cdot W_{i_8}[n, k].$$

The dequantized matmul output is

$$\hat y[n] = (r[n] - 128 \cdot \mathrm{cs}[n]) \cdot \frac{S_a}{127} \cdot \bar S_w[n].$$

In exact arithmetic and absent quantization error, $\hat y[n] = (Wx)[n]$ for the K-rotated reformulation of §3.1.

## 6.3 Signed-signed form

Define

$$r'[n] = \sum_k x_{i_8}[k] \cdot W_{i_8}[n, k].$$

The dequantized output is

$$\hat y[n] = r'[n] \cdot \frac{S_a}{127} \cdot \bar S_w[n].$$

The colsum is not consumed.

## 6.4 Equivalence

In exact arithmetic, the two forms produce identical $\hat y[n]$. The choice is determined by the available dot-product instruction at the kernel site (§1.5).

---

# VII. Per-Block GEMV Dequantization

## 7.1 Inputs

For one activation row with per-block scales over $K/B$ blocks of size $B$, and a per-row or per-block weight encoding:

| Quantity | Type | Shape | Storage |
|---|---|---|---|
| $x_{i_8}$ | int8 | $(K,)$ | activation int8 |
| $S_a$ | f32 | $(K/B,)$ | per-block activation absmax (raw) |
| $W_{i_8}$ | int8 | $(N, K)$ | weight int8 |
| $\bar S_w$ | f32 | $(N,)$ or $(N, K/B)$ | $S_w / 127$ |
| $\mathrm{cs}$ | f32 | $(N, K/B)$ | per-block weight colsum |

## 7.2 Unsigned-signed form, per-row weight scale

For each K-block $b$, define

$$r[n, b] = \sum_{k \in \Omega_b} (x_{i_8}[k] + 128) \cdot W_{i_8}[n, k].$$

The dequantized output is

$$\hat y[n] = \bar S_w[n] \cdot \sum_b (r[n, b] - 128 \cdot \mathrm{cs}[n, b]) \cdot \frac{S_a[b]}{127}.$$

## 7.3 Unsigned-signed form, per-block weight scale

When the weight scale is per-block,

$$\hat y[n] = \sum_b (r[n, b] - 128 \cdot \mathrm{cs}[n, b]) \cdot \frac{S_a[b]}{127} \cdot \bar S_w[n, b].$$

## 7.4 Signed-signed form

The signed-signed instantiation drops the colsum subtraction in the inner sum:

$$r'[n, b] = \sum_{k \in \Omega_b} x_{i_8}[k] \cdot W_{i_8}[n, k],$$

$$\hat y[n] = \bar S_w[n] \cdot \sum_b r'[n, b] \cdot \frac{S_a[b]}{127}$$

(per-row weight scale form; the per-block weight scale form analogous to §7.3 is obtained by multiplying inside the sum).

## 7.5 Output scale folding

The per-block GEMV admits a per-call output scalar $\beta \in \mathbb{R}$:

$$\hat y[n] = \beta \cdot \bar S_w[n] \cdot \sum_b (r[n, b] - 128 \cdot \mathrm{cs}[n, b]) \cdot \frac{S_a[b]}{127}.$$

The scalar $\beta$ multiplies the dequantized f32 result before it is accumulated or written. Because the dequantization tail is linear in its result, $\beta$ may be applied at any point after the block sum is formed and before the output store. This admits per-call scaling factors carried alongside each output row.

## 7.6 Accumulation structure and tile hardware

The granularity of the activation scale fixes where the scale sits relative to the K-axis reduction, and that placement determines whether the integer reduction can run uninterrupted on tile-accumulation hardware.

- *Per-row activation scale* (§6.2): the scale multiplies outside the K-sum. The contraction is one uninterrupted integer reduction over all of $K$; dequantization is a single f32 post-multiply per output. A tile multiply-accumulate unit can accumulate the full $K$ dimension in-tile and apply the scale once at the end.
- *Per-block activation scale* (§7.2): the scale multiplies inside the cross-block sum. The $K$ reduction is segmented into blocks of size $B$; each block's integer partial is scaled by its own f32 scale and summed in f32. The integer accumulator cannot span the full $K$ dimension.

On a per-output (GEMV) path the per-block form costs little. On a tile-accumulation (AMX-style) path it breaks the tile reduction pattern: the accumulator must be drained per block and an f32 scale applied mid-reduction. The per-block weight colsum is segmented the same way (§2.5), with the correction $128 \cdot \mathrm{cs}[n, b]$ applied per block; a per-row weight scale stays a single post-accumulation multiply and is tile-accumulation-friendly, whereas a per-block weight scale is not.

The granularity decision is thus a hardware decision as much as a precision one: per-block buys finer adaptation to within-row energy structure (§2.2) and tighter utilization (§3.6) at the cost of the tile reduction; per-row preserves the tile reduction at the cost of one scale per row. Neither dominates; the choice follows from the consuming hardware and the measured magnitude structure of the operand.

---

# VIII. Cache Encoding

A cache stores, per cached index, the int8 encoding of a per-element vector that a later operator consumes on a contraction axis — the score contraction (§X) and the aggregation (§XI). Per cached index the cache holds int8 data plus one f32 scale per (group, index).

## 8.1 Cache contents

For one cached position $t$ at group $g$:

| Quantity | Type | Storage |
|---|---|---|
| $K_{i_8}[g, t]$ | int8 | $(d,)$ in u8 or i8 form |
| $V_{i_8}[g, t]$ | int8 | $(d,)$ in i8 form |
| $K_{\text{scale}}[g, t]$ | f32 | $()$ |
| $V_{\text{scale}}[g, t]$ | f32 | $()$ |

where $d$ is the per-group block size.

## 8.2 Cached vector

Let $c_{g, t} \in \mathbb{R}^d$ be the per-element vector the downstream contraction consumes in original coordinates. The cache stores the int8 quantization of its block-Hadamard rotation:

$$\hat c_{i_8}[g, t, k] = Q_{S_c[g, t]}\big((\mathcal{H}_d\, c_{g, t})_k\big), \qquad S_c[g, t] = \max_k |(\mathcal{H}_d\, c_{g, t})_k|.$$

Forced orderings:

- Any coordinatewise gain or position-dependent rotation that the producing operator applies to $c$ in the original basis must complete before $\mathcal{H}_d$, since its conjugate under $\mathcal{H}_d$ is dense (HAR §6.4).
- The quantization follows $\mathcal{H}_d$: the scale $S_c$ is the absmax of the rotated vector, not of the original.

Free choices:

- The rotation $\mathcal{H}_d c$ may be formed at cache-write time, or absorbed offline into the producing weight (the two-sided form of §3.2 with M-axis block size $d$), in which case the producing operator already emits $\mathcal{H}_d c$ and the cache write applies no transform. The stored layout, the scale, and the dequantized result are identical either way.
- The int8 may be stored in i8 or u8 form (the u8 form is $\hat c_{i_8} \oplus \mathtt{0x80}$, §1.4). The choice follows the operand convention of the consuming dot product (§1.5). The dequantized result is identical under both storage choices.

## 8.3 Storage footprint

Per cached position per group, the cache holds $d$ int8 values for each cached vector, plus the per-(group, position) f32 scale of §8.1. The byte footprint is $d$ bytes per cached vector per position, independent of arrangement.

The physical layout of these bytes — contiguous per position, grouped across positions, or transposed between two cached tensors — is not fixed by this encoding. It is determined by the dot-product access patterns of the consuming kernels (§X, §XI), and the int8/u8 storage choice follows the consuming dot's operand convention. The dequantized result is identical across layouts.

---

# IX. Score-Operand Preparation

The non-cached operand of the score contraction (the query side) is encoded once per use rather than stored in the cache. The encoding mirrors §8.2; the difference is that auxiliary scalars are stored for the dequantization tail of §X.

## 9.1 Encoding

For a query operand $q \in \mathbb{R}^d$,

$$q_{i_8}[k] = Q_{S_q}\big((\mathcal{H}_d\, q')_k\big), \qquad S_q = \max_k |(\mathcal{H}_d\, q')_k|,$$

where $q'$ is the query after any preprocessing the producing operator applies in the original basis (for example normalization, a coordinatewise gain, or a position-dependent rotation). The two orderings of §8.2 are forced for the same reasons: original-basis coordinatewise or position-dependent operators precede $\mathcal{H}_d$ (HAR §6.4); the quantization follows $\mathcal{H}_d$. Whether $q'$ involves any such preprocessing, and in what order, is a property of the producing operator and is not fixed by this encoding.

## 9.2 Stored auxiliary scalars

Per query operand:

$$b_q = 128 \cdot \sum_k q_{i_8}[k], \qquad f_q = S_q.$$

$b_q$ is the §1.4 correction consumed by the score dot when the cached operand is stored in u8 form: it is $128$ times the sum of the signed query operand over the contraction index. $f_q$ is the raw query absmax, applied at score dequantization (§10.2).

Any additional per-output scalar that a consumer applies to the score (for example a fixed inner-product normalization) is not part of this encoding; it is a separate scalar (HAR §3.4) applied at or after the dequantization tail.

---

# X. Score Dequantization Identity

## 10.1 Raw score

For one query operand at position $p$ and one cached group $g$ at position $t$,

$$r[p, g, t] = \sum_k q_{i_8}[p, k] \cdot K_{u_8}[g, t, k]$$

where $K_{u_8}[g, t, k] = K_{i_8}[g, t, k] + 128$ (or $K_{u_8}$ stored directly as u8).

## 10.2 Dequantized score

The dequantized score is

$$s[p, g, t] = (r[p, g, t] - b_q[p]) \cdot \frac{f_q[p]}{127} \cdot \frac{K_{\text{scale}}[g, t]}{127} = (r[p, g, t] - b_q[p]) \cdot \frac{S_q[p] \cdot K_{\text{scale}}[g, t]}{127^2}.$$

Derivation: $r - b_q$ is the signed-signed inner product of $q_{i_8}$ and $K_{i_8}$ by §1.4. Dequantization applies $S_q/127$ on the query side and $K_{\text{scale}}/127$ on the cached side, contributing $S_q \cdot K_{\text{scale}} / 127^2$. No further normalization is part of this identity; any fixed score-scaling scalar is applied separately (§9.2).

## 10.3 Masking

For positions outside the valid contraction support, the score is set to $-\infty$ before the softmax (or, equivalently, those positions are excluded from the contraction). Whichever realization is used, it acts after dequantization and before the softmax, and does not change the dequantized score at valid positions.

---

# XI. Online Softmax and Aggregation

## 11.1 Per-position folded weight

For each cached position $t$, the unnormalized weight is $a_t = \exp(s[p, g, t] - m)$, where $m$ is the running max maintained by the online softmax. The folded weight is

$$w[t] = a_t \cdot V_{\text{scale}}[g, t].$$

## 11.2 Aggregation identity

Let $\ell[p] = \sum_t a_t$ be the softmax denominator. The dequantized aggregation output is

$$y[p, d] = \frac{1}{127 \cdot \ell[p]} \sum_t a_t \cdot V_{\text{scale}}[g, t] \cdot V_{i_8}[g, t, d] = \frac{1}{127 \cdot \ell[p]} \sum_t w[t] \cdot V_{i_8}[g, t, d].$$

The factor $1/127$ is the V dequantization; $1/\ell$ is the softmax normalization. This identity fixes the result. The order of accumulation and the arithmetic used to form the sum are implementation choices that do not change the dequantized output.

## 11.3 Efficient u8-fold implementation (non-normative)

The sum of §11.2 may be formed in any arithmetic that realizes the identity; a direct f32 accumulation of $w[t] \cdot V_{i_8}$ uses no intermediate quantization. One efficient alternative quantizes the folded weights of a tile of positions to u8 so the sum becomes a u8·i8 dot. Over a tile with positions $t \in \mathrm{grp}$:

$$m_w = \max_{t \in \mathrm{grp}} w[t], \qquad w_{u_8}[t] = \mathrm{clamp}(\mathrm{round}(255 \cdot w[t] / m_w), 0, 255)$$

(u8 because $w[t] \ge 0$), and the tile contributes $\big(\sum_{t} w_{u_8}[t] \cdot V_{i_8}[g, t, d]\big) \cdot (m_w / 255)$. The per-tile $m_w$ cancels to within the $w_{u_8}$ rounding, so no cross-tile bookkeeping of $m_w$ is required. The 8-bit grid is then shared between the softmax weight $a_t$ and the within-tile $V_{\text{scale}}$ variation: for $V_{\text{scale}}$ varying by factor $f$ across the tile the effective grid for $a_t$ is $256/f$, so smaller tiles keep $V_{\text{scale}}$ variation local and admit a tighter grid.

---

# XII. Norm-Output Encoding

## 12.1 Definition

The int8 activation produced from an RMSNorm-with-gain followed by a K-axis rotation is the quantization of the Hadamard-rotated, gain-split normalized vector. Per row $x \in \mathbb{R}^d$:

$$x_{i_8} = Q_{S_a}\!\Big(\mathcal{H}_d\big(\sigma_\gamma \odot \tfrac{x}{\mathrm{rms}(x)}\big)\Big), \qquad S_a = \max_k \big|\big(\mathcal{H}_d(\sigma_\gamma \odot \tfrac{x}{\mathrm{rms}(x)})\big)_k\big|,$$

where $\sigma_\gamma$ is the activation-side split-gain factor of §4.4. The per-block variant replaces $S_a$ with a per-block absmax $S_a[b]$ over each $\Omega_b$ and quantizes per block.

Two orderings are forced, each by a non-commutation or by the basis in which a quantity is defined; everything else is free:

- **The gain $\sigma_\gamma$ precedes $\mathcal{H}_d$.** A coordinatewise gain does not commute with the block-Hadamard — $\mathcal{H}_d \mathrm{diag}(\sigma_\gamma) \mathcal{H}_d^T$ is dense (HAR §6.4) — so it cannot be applied in the rotated basis without materializing a dense operator.
- **The quantization follows $\mathcal{H}_d$.** The int8 grid and its scale $S_a$ are defined on the rotated coordinates: $S_a$ is the absmax of $\mathcal{H}_d(\cdot)$, not of the original vector.

The RMS factor $\alpha = 1/\sqrt{\mathrm{rms}(x)^2}$ is a scalar and commutes with every step (HAR §3.4); where it is applied is immaterial, and it may be folded into $\sigma_\gamma$ or applied separately for arithmetic economy. The squared sum that defines it is norm-based and may be collected in either basis (HAR §5.1), including from per-shard partial sums (HAR §7.3).

The output dtype is i8 for $x_{i_8}$ and f32 for $S_a$ (or $S_a[\cdot]$).

---

# XIII. Router Encodings

## 13.1 Linear router score

A router computes a per-expert linear score

$$\delta_e = \sum_k x_k \cdot W[e, k]$$

for a router weight $W \in \mathbb{R}^{E \times d}$. Any selection policy applied to $\delta$ — an activation $\sigma$, an additive per-expert term, a top-$K$ selection, a mixture-weight renormalization — is a property of the consuming operator. The encodings below concern only the storage and exact reconstruction of $\delta_e$; they are agnostic to the policy, except where a policy property (shift-invariance, §13.5) permits dropping stored data.

## 13.2 Centered bf16 router with gauge

Define the column gauge

$$g[k] = \frac{1}{E} \sum_e W[e, k]$$

and the centered weight

$$C[e, k] = W[e, k] - g[k].$$

The stored encoding:

| Tensor | Type | Shape |
|---|---|---|
| $C$ | bf16 | $(E, d)$ |
| $g$ | bf16 | $(d,)$ |

The reconstruction:

1. $p = \sum_k x_k \cdot g[k]$ (one dot per token).
2. $c_e = \sum_k x_k \cdot C[e, k]$ (one dot per expert per token).
3. $\delta_e = c_e + p$.

In exact arithmetic, $\delta_e = \sum_k x_k \cdot W[e, k]$, since $c_e + p = \sum_k x_k C[e, k] + \sum_k x_k g[k] = \sum_k x_k (C[e, k] + g[k])$.

Centering can reduce the bf16 cast error of the stored matrix when the column-mean component dominates per-element magnitudes: the bf16 round-trip relative error scales with element magnitude, and the gauge $g$ is stored at full bf16 precision separately. The pivot $p$ is the cost of recovering the exact $\delta_e$ from the centered representation.

## 13.3 Pivot requirement

When the selection policy is not shift-invariant in $\delta$ (for example, a nonlinear activation applied to $\delta_e$ before selection), the absolute $\delta_e$ must be reconstructed, and the pivot $p$ of §13.2 is required: omitting it replaces the policy's argument $\delta_e$ with $c_e = \delta_e - p$, shifting every argument by $-p$, which a non-shift-invariant policy does not absorb.

## 13.4 Additive-term precision

If the selection policy adds a per-expert term that is compared directly against the score margin between adjacent selection ranks, the dtype of that term sets the noise floor of the selection. If the margin between ranks $K$ and $K+1$ falls below the bf16 floor on a non-trivial fraction of decisions, storing the term as bf16 introduces selection mismatches at that fraction; storing it at f32 keeps it above the bf16 noise floor. This applies only to a policy that has such an additive term.

## 13.5 Shift-invariant policies

For a policy that is invariant to subtracting one scalar from every expert score — for example selection from `softmax(δ)` with no pre-selection additive term — the gauge pivot is not required. Centering produces

$$c_e = \sum_k x_k C[e, k] = \delta_e - p$$

with the same $p = \sum_k x_k g[k]$ for every expert, and

$$\operatorname{softmax}(\delta - p) = \operatorname{softmax}(\delta), \qquad \operatorname{topk}(\delta - p) = \operatorname{topk}(\delta).$$

The encoding for this class stores only the centered bf16 weight and omits the gauge sidecar, because the policy never needs the absolute score.
