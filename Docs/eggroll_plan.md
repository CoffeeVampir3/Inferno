# EGGROLL Training — Engineering Plan

Scope: continuous-batching strategy, model-dispatch integration, and the kernels.
Target: the butterquant model (`modeling/gemma_4_moe_bq.mojo`). Built on the
memory module already in place (`inspectable_toolkit/eggroll.mojo`). On-disk
checkpointing is out of scope here.

---

## 0. Invariants every kernel assumes

Math being implemented (per weight matrix `W`):

```
M      = M0 + D                         mean weights (M0 frozen, D trained)
fitness  evaluate  f( quant(M) + σ·Eᵢ ) batched over the population
Eᵢ     = (1/√r) · Aᵢ Bᵢᵀ               Aᵢ∈[m,r]  Bᵢ∈[n,r]   from seeds
fold   D ← (1−λ)·D + (α/N) · Σⱼ fⱼ Eⱼ   weight-decay λ + ES ascent
```

Resident buffers and who reads/writes them:

| Buffer | dtype | role | reader | writer |
|---|---|---|---|---|
| live arena (existing) | int8 bq | **working** = `quant(M0+D)` | forward (unchanged) | re-quant kernel |
| `EggrollBase` (add, ShadowWeights-style) | int8 bq | frozen `M0` | re-quant | populated once at arm |
| `EggrollDiff` (built) | **bf16** | running diff `D` | fold, re-quant | fold |
| `EggrollWorkspace` (built) | f32 | transient `A`/`B` factor tiles | fold / correct | fold / correct |
| `EggrollState` (built) | — | `N, r, σ, α, λ, base_seed, step`, slot→worker | all | driver |

Key consequence: **the forward keeps binding the live arena**; we only refresh
that arena from `EggrollBase + D` each fold. Perturbations are never stored —
`Aⱼ,Bⱼ` regenerate from `(base_seed, step, worker, matrix_id)` identically on
every rank, so the only cross-NUMA traffic is the scalar fitness all-reduce.

Identity for the matrix id (drives the RNG, must be stable across forward+fold):

```
eggroll_matrix_id(layer, slot, expert=0) -> Int     # pack: layer<<16 | slot<<8 | expert
```

---

## 1. Continuous batching strategy

**Population ↔ rows.** The population maps onto the continuous-batch row axis.
Each resident slot is assigned one worker (member) for the step; every token-row
of that slot inherits the slot's worker. The base `uMᵀ` is one shared GEMM over
the whole packed batch (working weights, identical for all members); members
differ only in the per-row low-rank correction. So one forward evaluates the
whole population at batched-inference throughput.

**Assignment.** At the top of `execute`, after `pack_slot_starts`, the driver has
already called `EggrollState.assign_workers(workers, num_slots)`. Row→worker is
then `worker = state.worker_for( slot_of_row(row) )`, where `slot_of_row` comes
from `buf_starts` (the same map steering uses). Pass a compact per-row worker
vector (length `total`) into the correction kernels so the inner loop is a direct
index, not a search.

**Fitness.** Fitness is collected only at emit rows (the sampled position), as the
log-prob the model assigns to the supervised target token (reuse the `logz` /
`flash_kl` path already in the head). Accumulate per worker:
`fit[worker] += logprob(target)`. After the eval batch, **scalar all-reduce
`fit[0..N]`** across ranks (`dispatch_allreduce_inplace`, a few KB) and
rank-normalize (Salimans utilities, centered to ~[−0.5, 0.5]) before the fold —
this is also what bounds the per-step update magnitude.

**Stale-base window K (amortization).** Freeze the working weights for K
micro-steps; during the window store only `(seed_step, fit[0..N])` (tiny) and do
one fold + one re-quant at the window end. Folding K micro-steps against a frozen
base is identical to one step at population `K·N`, and it amortizes the ~10×-int8
re-quant/fold bandwidth over `K·N` evaluations. Size `K·N` above the break-even
`P* ≈ a few × (int8_TOPS / mem_BW)` (hundreds–low-thousands of token-evals) or the
f32-grade fold dominates and quantization buys nothing.

**Scheduler loop per ES step:** submit the eval prompts as a wave (as
`measure_refusal.wave_run` does), tag each request's slot with its worker via
`assign_workers`, run `sched.step(model)` to completion, read the emitted
outcomes into `fit[]`, retire. Repeat K times, then fold+requant, then
`state.advance()`.

---

## 2. Model dispatch (forward integration)

Keep it a strap-on, exactly like steering:

1. **Comptime gate.** Add `eggroll_workers: Int = 0`, `eggroll_rank: Int = 0`
   params to `Gemma4`. Zero ⇒ every EGGROLL scratch band sizes to 0 and every tap
   compiles out — same mechanism as `steer_vectors`/`measure_rows`.

2. **State + hooks.** Carry `var eggroll: EggrollState` on the model (alongside
   `steer`). Implement the `Evolvable` trait (`arm_eggroll`, `disarm_eggroll`,
   plus `assign_workers`, `collect_fitness`). No bulk memory on the model — `D`,
   `M0` shadow, workspace are caller-owned (driver), reached via
   `model.layout`/`arena_bases`/`pools`, abliteration-style.

3. **Per-matmul tap.** After each base matmul, gated by `if self.eggroll.armed:`,
   call `dispatch_eggroll_correct(...)` writing into the same output binding. Taps
   in `gemma_4_moe_bq.mojo`:
   - `dispatch_bq_qkv` (q/k/v) and `dispatch_bq_linear` (full q/k) — attn proj
   - `dispatch_bq_block_linear` (o_proj, sliding+full)
   - `dispatch_bq_linear` (dense gate, up); `dispatch_bq_block_linear` (dense down)
   - `dispatch_router_expert` (router)
   - `dispatch_bq_phase1_gate_up`, `dispatch_bq_phase2_down` (experts)
   - `dispatch_bq_flash_sample` (tied head) — fused; correction folds into logits
   - `dispatch_bq_embed_lookup` (embed) — gather; correction adds `(σ/√r)·Aᵢ[row]`
     selected per emitted token (cheap, optional first cut)

4. **Scratch band.** Add a comptime-gated `uB` band to the relevant
   `ScratchIsland`s: `[total_tokens × eggroll_rank]` f32 per matmul context.
   Sized to 0 when `eggroll_rank==0`.

5. **MoE wrinkle (the one non-trivial bit).** Tokens are bucketed by expert
   (`expert_offset`/`routes`). The per-row worker id must ride the permutation:
   when building `routes`, carry `worker` alongside the token index so phase1/2
   can pick `Aᵢ,Bᵢ` for `(expert, worker)`. `matrix_id` encodes the expert.

The forward never sees `D` directly — it reads the live (working) arena. The
correction adds `σEᵢ` (continuous, f32) on top; the committed state advances only
on the quantized lattice at each re-quant.

---

## 3. Kernels

### 3.1 Factor reconstruction (RNG)

Already have `eggroll_factor_counter(worker_seed, matrix_id, side, index)` over
`splitmix64`. Add a SIMD draw — default **Rademacher** (sign bit of the hash):
unit-variance, symmetric, bounded ⇒ sub-Gaussian (satisfies the paper's
Assumption 6), cheapest, and the CLT makes `ABᵀ` Gaussian anyway. Provide a
Box–Muller `eggroll_normal` as an option.

```
A[i,k] = draw( eggroll_factor_counter(ws, mid, FACTOR_A, i*r + k) )   # i∈[0,m) k∈[0,r)
B[j,k] = draw( eggroll_factor_counter(ws, mid, FACTOR_B, j*r + k) )   # j∈[0,n) k∈[0,r)
```

Forward and fold call the identical path ⇒ bit-identical factors.

### 3.2 Forward correction — `dispatch_eggroll_correct`

```
dispatch_eggroll_correct[hidden, ...](
    u_act,            # GEMM input  (bf16, or int8+scale for bq → dequant on read)
    y_out,            # base GEMM output [seq, n_rows] bf16, updated in place
    row_worker,       # [seq] Int   per-row member id
    matrix_id, m_rows, n_cols,
    state,            # σ, r, base_seed, step
    scratch_uB, pools, prof)
```

Per matmul, grouped by active worker `w`:
- reconstruct `B_w [n,r]` and `A_w [m,r]` (shared across all of `w`'s rows);
- for each of `w`'s rows `t`: `s_k = Σ_j u[t,j]·B_w[j,k]`  (r dot products → `scratch_uB`);
- `y[t,i] += (σ/√r) · Σ_k s_k · A_w[i,k]`  (scaled axpy of `A_w` columns).

`r=1` fast path: `s = u·B_w` (one dot), `y[t] += (σ·s)·A_w`. Cost is `m·(#workers)`
reconstruction + `m·seq` axpy — an `n×` smaller than the shared `uMᵀ`, so
arithmetic intensity is preserved. bq path: read `u` as `scale·int8` inline.

### 3.3 Fold — `dispatch_eggroll_fold`

```
dispatch_eggroll_fold[...](
    diff,             # EggrollDiff (bf16 D, in place)
    workspace,        # EggrollWorkspace (f32 A/B tiles)
    fit,              # [N] rank-normalized scalars
    matrix_id, m_rows, n_cols,
    state,            # α, λ, N, r, base_seed, step
    pools, prof)
```

Per trainable matrix:
- apply decay and accumulate in one pass:
  `D[i,j] ← (1−λ)·D[i,j] + (α/N)·Σ_w fit[w]·(1/√r)·Σ_k A_w[i,k]·B_w[j,k]`;
- the `Σ_w` is the paper's `(diag(f)·A)ᵀ·B` GEMM with contraction over `N·r`
  (worker × rank). Tile over workers (`worker_tile`) and output rows to bound the
  f32 workspace; reuse the existing GEMM tiling.
- accumulate the row-tile in f32, then **stochastic-round** to bf16 `D` using the
  RNG (no error-feedback buffer needed). Decay is the `(1−λ)` prescale on the
  existing `D` read.

Replicated matrices (full q/k, router) fold identically on every rank (same
factors, same all-reduced `fit`) and stay bit-identical with no communication;
sharded matrices (experts, attn-out, gate/up/down) fold only their local shard.

### 3.4 Re-quant — `dispatch_eggroll_requant`

```
working_live ← quant( EggrollBase(M0) + EggrollDiff(D) )
```

Reuse the existing butterquant pipeline: dequant `M0` block, add `D`, re-derive
FWHT + per-row/block scale + VNNI pack + colsum (`bake_split_gain_in_place`,
`dispatch_pack_colsum`). Writes the **live arena in place** so the forward binding
is unchanged. Run once per stale-base window. As `‖D‖` grows the per-block scales
inflate (no overflow) but resolution degrades — another reason decay matters.

### 3.5 Fitness — `dispatch_eggroll_fitness`

At each emit row, target-token log-prob from the head's `logz`/logits; scatter-add
into `fit[worker]`. Then scalar `dispatch_allreduce_inplace[F32]` over `fit[0..N]`
and rank-normalize. This is the only place population fitness crosses ranks.

---

## 4. Training step (driver orchestration)

```
arm: populate EggrollBase from live; D=0; state.arm(N, r, σ, α, λ, seed)
loop over ES steps:
    zero fit[0..N]
    repeat K micro-steps (stale base):
        assign workers→slots; run eval wave; accumulate fit[worker] += logprob(target)
    allreduce + rank-normalize fit
    dispatch_eggroll_fold   (D ← (1−λ)D + (α/N)Σ f E)        # NUMA-local
    dispatch_eggroll_requant (live ← quant(M0 + D))          # NUMA-local
    state.advance()
```

Everything between `arm` and the final read is local per rank except the one
`fit` all-reduce.

---

## 5. Weight decay & points of interest

- **Decay = the only bound.** ES ascent has no restoring force; the cumulative
  diff diffuses as √T. `D ← (1−λ)D + …` gives a hard equilibrium
  `‖D‖ ≲ α‖ĝ‖/λ`, doubles as trust-region anchoring to the pretrained `M0`
  (decay→0 ≡ pull toward `M0`), and fills the scale-regulation channel that
  freezing the RMSNorm γ removes. Carry `λ` in `EggrollState` (already there);
  fuse it as the `(1−λ)` prescale in the fold.
- **Per-step vs cumulative.** Per-step update is a population *average* (bounded,
  more so with rank-shaping); the accumulation is pure-additive — hence the decay.
- **σ never quantizes.** The perturbation is applied in f32 inside the matmul
  accumulator, so σ may sit far below the int8 LSB with no loss. Only the
  *committed* state lives on the quant lattice; `D` (bf16) is the sub-LSB
  integrator that decides when a lattice cell flips. Tune `α` so folds cross
  thresholds at a useful rate.
- **Stochastic rounding is mandatory** on the bf16 `D` write (RNG already
  available) — keeps sub-ULP folds unbiased without an error-feedback buffer.
- **bf16 D range is fine** (fp32 exponent); the 7-bit mantissa is the only cost,
  mitigated by stochastic rounding.
- **NUMA.** `EggrollBase`/`D`/workspace mirror the live sharding (per-rank
  arenas); folds are local writes; replicated copies stay in sync via shared seeds
  + the scalar all-reduce. No remote weight reads/writes.
- **Break-even.** `K·N` token-evals per fold must clear `P*`; below it the f32-grade
  fold/requant bandwidth dominates and bf16-forward would be faster. Log the chosen
  `K·N` so a small debug run isn't silently in the wrong regime.
- **Embed/head.** The `VOCAB×HIDDEN` factor `A` is large — reconstruct it in row
  tiles in both correct and fold (never materialize `[VOCAB,r]` whole).
- **matrix_id** must be identical in correct and fold (and across ranks); experts
  fold the expert index into the id and into the bucket-carried worker tag.
- **First cut.** Validate the loop on dense taps only (attn-out + gate/up/down) at
  `r=1`, fitness from `logz`, before wiring the MoE bucket worker-tag and the
  head/embed taps — those are the only non-mechanical pieces.
```
