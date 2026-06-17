# MiniMax-M3 Architecture Reference

This document describes the shipped `MiniMaxAI/MiniMax-M3` release artifacts as they are packaged: the serialized model config, the checkpoint-local Hugging Face config code, the tokenizer and processor metadata, the generation config, and the safetensors weight index. It separates exact observed behavior (config fields and on-disk tensor shapes/dtypes read directly from the release) from metadata that is present in the release but not enforced by every runtime.

All tensor shapes and dtypes below were read from the safetensors headers of the released shards (`model-000{01,03,26,59}-of-00059.safetensors`) and from `model.safetensors.index.json`. The structural counts are cross-checked against the full weight index, which contains exactly `23416` tensors.

## 1. Model Summary

MiniMax-M3 is a native multimodal (image/video/text) Mixture-of-Experts model: a CLIP-style vision tower and a patch-merge projector feed a decoder-only causal MoE language backbone. It is the successor to MiniMax-M2, and its headline change is **MiniMax Sparse Attention (MSA)** — a DeepSeek-style block-sparse attention with a lightweight "lightning indexer" — used on most layers to make million-token context cheap.

- Released as `MiniMaxM3SparseForConditionalGeneration`, `model_type="minimax_m3_vl"`.
- Language backbone: 60 decoder layers, residual width 6144.
- First 3 layers are dense (full attention + dense MLP); the remaining 57 layers are sparse (MSA + MoE).
- 64 query heads, 4 key/value heads, head dimension 128 (grouped-query attention, KV expansion 16).
- Partial RoPE on the first 64 of each 128 head dimensions; per-head Q/K RMSNorm.
- Gemma-style `(1 + weight)` RMSNorm throughout (`use_gemma_norm`).
- MSA sparse attention on layers 3-59 via a 4-head lightning indexer selecting top-16 key blocks of 128 tokens.
- 128 experts per MoE layer, top-4 routing, plus 1 shared expert. SwiGLU-OAI (GPT-OSS-style clamped) experts, intermediate size 3072.
- Dense MLP layers use SwiGLU-OAI with intermediate size 12288.
- Untied token embedding and LM head; vocab 200064.
- Weights are bfloat16 on disk (the router gate and correction bias are float32). No FP8/NVFP4 quantization in the official release — community NVFP4/GGUF repos are separate re-quantizations.
- Release advertises Multi-Token Prediction (`num_mtp_modules=7`), but the released weight map contains no MTP tensors.
- Vision tower: 32-layer CLIP encoder, hidden 1280, patch 14, 2×2 spatial / 2-frame temporal patch merge.

Official figures (README / model card): ~428B total parameters, ~23B activated per token, 1M context, and "9× prefill / 15× decode" speedups vs M2 at 1M context. The on-disk `total_size` is `869,157,697,024` bytes, i.e. ~434.6B parameters in bfloat16. Technical report: arXiv:2606.13392. MSA operator: `github.com/MiniMax-AI/MSA`.

## 2. Serialized `config.json`

`config.json` is a composite VL config with `text_config` and `vision_config` sub-objects.

### Top level

| Field | Value |
| --- | --- |
| `architectures` | `["MiniMaxM3SparseForConditionalGeneration"]` |
| `auto_map` | `AutoConfig -> configuration_minimax_m3_vl.MiniMaxM3VLConfig` |
| `model_type` | `"minimax_m3_vl"` |
| `image_token_index` | `200025` |
| `video_token_index` | `200026` |
| `image_seq_length` | `576` |
| `process_image_mode` | `"dynamic_res"` |
| `projector_hidden_act` / `projector_hidden_size` | `"gelu"` / `6144` |
| `vision_feature_layer` | `-1` |
| `vision_feature_select_strategy` | `"full"` |
| `multimodal_projector_bias` | `true` |
| `torch_dtype` | `"bfloat16"` |
| `transformers_version` | `"4.52.4"` |

### `text_config`

| Field | Value |
| --- | --- |
| `architectures` | `["MiniMaxM3SparseForCausalLM"]` |
| `hidden_size` | `6144` |
| `intermediate_size` (routed expert) | `3072` |
| `dense_intermediate_size` | `12288` |
| `shared_intermediate_size` | `3072` |
| `num_hidden_layers` | `60` |
| `num_attention_heads` | `64` |
| `num_key_value_heads` | `4` |
| `head_dim` | `128` |
| `vocab_size` | `200064` |
| `max_position_embeddings` | `1048576` |
| `rms_norm_eps` | `1e-6` |
| `use_gemma_norm` | `true` |
| `attention_output_gate` | `false` |
| `rope_theta` | `5000000` |
| `rotary_dim` / `partial_rotary_factor` | `64` / `0.5` |
| `hidden_act` | `"swigluoai"` |
| `use_qk_norm` / `qk_norm_type` | `true` / `"per_head"` |
| `tie_word_embeddings` | `false` |
| `num_local_experts` | `128` |
| `num_experts_per_tok` | `4` |
| `n_shared_experts` | `1` |
| `scoring_func` | `"sigmoid"` |
| `use_routing_bias` | `true` |
| `routed_scaling_factor` | `2.0` |
| `swiglu_alpha` / `swiglu_limit` | `1.702` / `7.0` |
| `moe_layer_freq` | 60 entries: `[0,0,0, 1×57]` |
| `num_mtp_modules` | `7` |
| `num_nextn_predict_layers` | `1` |
| `sparse_attention_config` | see below |

### `text_config.sparse_attention_config`

| Field | Value |
| --- | --- |
| `use_sparse_attention` | `true` |
| `sparse_index_dim` | `128` |
| `sparse_num_index_heads` | `4` |
| `sparse_topk_blocks` | `16` |
| `sparse_block_size` | `128` |
| `sparse_score_type` | `"max"` |
| `sparse_init_block` | `0` |
| `sparse_local_block` | `1` |
| `sparse_attention_freq` | 60 entries: `[0,0,0, 1×57]` |
| `sparse_disable_index_value` | 60 entries: `[0,0,0, 1×57]` (the index value path is disabled on every sparse layer) |

### `vision_config`

| Field | Value |
| --- | --- |
| `model_type` | `"clip_vision_model"` |
| `hidden_size` | `1280` |
| `intermediate_size` | `5120` |
| `num_hidden_layers` | `32` |
| `num_attention_heads` | `16` |
| `patch_size` | `14` |
| `image_size` | `2016` |
| `projection_dim` | `6144` |
| `position_embedding_type` / `rope_mode` | `"rope"` / `"3d"` |
| `rope_theta` | `10000.0` |
| `hidden_act` | `"gelu"` |
| `layer_norm_eps` | `1e-5` |
| `img_token_compression_config` | `patch_merge`, `spatial_merge_size=2`, `temporal_patch_size=2` |
| `vision_segment_max_frames` | `4` |

Derived shape facts (confirmed against on-disk tensors):

- Query projection width is `64 * 128 = 8192`.
- Key/value projection width is `4 * 128 = 512`. GQA repeats each KV head 16 times.
- `num_attention_heads * head_dim = 8192 > hidden_size = 6144`, so attention projects into a wider space and `o_proj` maps `8192 -> 6144`.
- `rotary_dim / head_dim = 64 / 128 = 0.5`: RoPE rotates half of each head.
- `moe_layer_freq` and `sparse_attention_freq` share the same `[0,0,0, 1×57]` pattern, so the dense/full-attention layers and the sparse/MoE layers coincide: layers 0-2 are dense, layers 3-59 are sparse MoE.

## 3. Runtime Defaults from the Config Class

The bundled `configuration_minimax_m3_vl.py` (shipped for `trust_remote_code` loading) is intentionally thin: it defines `MiniMaxVLBaseConfig`/`MiniMaxM3VLConfig`, coerces `text_config`/`vision_config` dicts into sub-configs, and supplies the multimodal defaults (`image_token_index=200025`, `video_token_index=200026`, `image_seq_length=576`, `process_image_mode="dynamic_res"`, `projector_hidden_act="gelu"`, `vision_feature_select_strategy="full"`). The text backbone dict uses a `model_type` not in HF `CONFIG_MAPPING`, so its keys are attached to a generic `PretrainedConfig`.

The richer text/vision defaults (initializer ranges, RoPE parameter objects, per-layer `layer_types`/`mlp_layer_types` derivation) live in the upstream `transformers` `minimax_m3_vl` package, not in the bundled checkpoint-local file. In that upstream config:

- `bos_token_id=200034`, `eos_token_id=200020`, `pad_token_id=None`.
- `max_position_embeddings` defaults to `524288` in the class, but the shipped `config.json` overrides it to `1048576`.
- `layer_types[i]` is `"minimax_m3_sparse"` where `sparse_attention_freq[i]==1`, else `"full_attention"`.
- `mlp_layer_types[i]` is `"sparse"` where `moe_layer_freq[i]==1`, else `"dense"`.
- The legacy `sparse_attention_config` keys are mapped to flat `index_*` fields: `index_n_heads=4`, `index_head_dim=128`, `index_block_size=128`, `index_topk_blocks=16`, `index_local_blocks=1`.
- `hidden_act` is normalized to `"silu"` as the pointwise fallback; the actual gate is computed inline from `swiglu_alpha`/`swiglu_limit`.

## 4. Top-Level Module Structure

The packaged conditional-generation entry point (`MiniMaxM3SparseForConditionalGeneration`) wraps:

1. `vision_tower`: CLIP-style 32-layer ViT encoder (`vision_tower.vision_model.*`).
2. `multi_modal_projector`: GELU MLP `1280 -> 6144 -> 6144`.
3. `patch_merge_mlp`: GELU MLP that fuses `2×2` merged patches, `24576 -> 6144 -> 6144`.
4. `language_model`: the causal MoE backbone, itself containing:
   - `model.embed_tokens`: `(200064, 6144)`.
   - `model.layers`: 60 decoder layers (3 dense + 57 sparse MoE).
   - `model.norm`: final Gemma RMSNorm.
   - `lm_head`: `(200064, 6144)`, untied.

Each decoder layer is pre-norm with two residual sub-blocks:

1. `input_layernorm` (Gemma RMSNorm) -> self-attention -> residual add.
2. `post_attention_layernorm` (Gemma RMSNorm) -> MLP (dense or MoE) -> residual add.

Note: upstream `transformers` folds the patch-merge step inside one `MiniMaxM3VLMultiModalProjector` (with `merge_linear_1/2` submodules), whereas the **released checkpoint stores it as a separate top-level `patch_merge_mlp` module**. The on-disk tensor names are `multi_modal_projector.linear_{1,2}.{weight,bias}` and `patch_merge_mlp.linear_{1,2}.{weight,bias}`.

## 5. Attention Block

Every layer uses grouped-query causal self-attention with per-head Q/K normalization and partial RoPE. Layers 3-59 additionally run the MSA lightning indexer to make attention block-sparse.

### Projection Shapes (identical on dense and sparse layers)

- `q_proj`: `(6144 -> 8192)`, biasless.
- `k_proj`: `(6144 -> 512)`, biasless.
- `v_proj`: `(6144 -> 512)`, biasless.
- `o_proj`: `(8192 -> 6144)`, biasless.
- `q_norm`, `k_norm`: shape `(128,)` — per-head RMSNorm over `head_dim`.

### Q/K Normalization

`qk_norm_type="per_head"`: Q and K are reshaped to `(..., heads, 128)` and normalized per head with Gemma-style RMSNorm (`(1 + weight)` scale), before RoPE. This differs from M2.7, which used full-vector Q/K norm.

### Rotary Position Embeddings

- RoPE base `5000000`, configured span `1048576`.
- Only the first 64 of each 128 head dimensions are rotated; the remaining 64 pass through.
- Cos/sin are computed in float32 from the positions and cast back to the activation dtype.

### MSA Sparse Attention (layers 3-59)

Each sparse layer carries a **lightning indexer** (`self_attn.index_*`), a selection-only branch modeled on DeepSeek-V4's DSA indexer:

- `index_q_proj`: `(6144 -> 512)` = `index_n_heads(4) * index_head_dim(128)`.
- `index_k_proj`: `(6144 -> 128)` = a single shared index key head.
- `index_q_norm`, `index_k_norm`: `(128,)` Gemma RMSNorm, applied per index head before RoPE.
- The indexer has **no value projection and produces no residual output** (`sparse_disable_index_value=1` on every sparse layer).

Indexer selection per query:

1. Project hidden states to index Q `(B, 4, S, 128)` and index K `(B, 1, S, 128)`; normalize and apply partial RoPE (first 128 dims, i.e. the full index head dim).
2. Score every (query, key) pair: `scores = idx_q · idx_kᵀ` in float32; mask future keys to `-inf`.
3. Pool keys into blocks of `sparse_block_size=128`: take the max score within each block, then the max across the 4 index heads, giving a per-query, per-block score.
4. Always-keep the `sparse_local_block=1` block(s) immediately preceding the query (their score is boosted to `+inf`).
5. Keep the top `sparse_topk_blocks=16` blocks per query; future/empty blocks sort to the end and are tagged `-1` (left-packed indices, `-1` right-padding) — the format the block-sparse attention kernel consumes.

The main attention then computes softmax attention with scale `1/sqrt(128)` only over the selected key blocks. On the `eager`/`sdpa` path the block indices are expanded into a dense additive `[B, 1, S_q, S_k]` mask (`0` on kept (query,key) pairs, `-inf` elsewhere) by `build_block_mask`. Block boundaries are anchored to absolute key slots, so only right-padding is equivalent to an unpadded run (a documented limitation shared with DeepSeek-V4).

The dense layers 0-2 (`layer_types[i]=="full_attention"`) have no indexer and run ordinary dense causal attention.

### `attention_output_gate`

`config` sets `attention_output_gate=false`; there is no output-gating branch on attention in this release.

## 6. MLP Blocks

### Dense layers (0-2): `MiniMaxM3VLDenseMLP`

On-disk tensors `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`:

- `gate_proj`, `up_proj`: `(6144 -> 12288)`.
- `down_proj`: `(12288 -> 6144)`.

The activation is SwiGLU-OAI (GPT-OSS style), computed from `swiglu_alpha=1.702`, `swiglu_limit=7.0`:

```
gate = clamp(gate, max=7.0)
up   = clamp(up, min=-7.0, max=7.0)
glu  = gate * sigmoid(gate * 1.702)
out  = down_proj((up + 1.0) * glu)
```

(Upstream `transformers` packs `gate_proj`/`up_proj` into one fused `gate_up_proj`; the checkpoint stores them split.)

### Sparse layers (3-59): `block_sparse_moe`

A MoE block with a router, 128 routed experts, and 1 shared expert.

#### Router (`block_sparse_moe.gate`)

- Weight shape `(128, 6144)`, dtype **float32**.
- `block_sparse_moe.e_score_correction_bias`: shape `(128,)`, dtype float32.

Routing (sigmoid scoring, as in M2):

1. `router_logits = gate(x)` in float32.
2. `routing_weights = sigmoid(router_logits)`.
3. `scores_for_choice = routing_weights + e_score_correction_bias`.
4. Select top-4 experts by `scores_for_choice`.
5. Gather the **raw sigmoid weights** (not the bias-adjusted scores) for the chosen 4 and renormalize them to sum to 1.

The correction bias steers selection only; the mixture weights come from renormalized sigmoid scores. The renormalized routed output is multiplied by `routed_scaling_factor=2.0`, then the shared-expert output is added (the shared expert is not scaled).

#### Routed experts (`block_sparse_moe.experts.{0..127}`)

Each expert is a SwiGLU-OAI MLP with `w1`/`w2`/`w3` (gate/down/up):

- `w1` (gate): `(6144 -> 3072)`
- `w3` (up):   `(6144 -> 3072)`
- `w2` (down): `(3072 -> 6144)`

`expert(x) = w2( (up + 1.0) * (gate * sigmoid(gate * 1.702)) )` with the same clamps as the dense MLP (`gate ≤ 7`, `up ∈ [-7, 7]`).

#### Shared expert (`block_sparse_moe.shared_experts`)

A single always-on dense SwiGLU-OAI MLP, stored as `gate_proj`/`up_proj`/`down_proj`:

- `gate_proj`, `up_proj`: `(6144 -> 3072)` (`shared_intermediate_size=3072`).
- `down_proj`: `(3072 -> 6144)`.

This is the `n_shared_experts=1` branch — absent in M2.7, present in M3.

## 7. Vision Tower and Projector

`vision_tower.vision_model` is a CLIP-style ViT:

- `embeddings.patch_embedding`: conv weight `(1280, 3, 2, 14, 14)` — a 3D patch embed over `temporal_patch_size=2` × `14×14` spatial patches, dtype float32.
- `pre_layrnorm`: `(1280,)` weight + bias (the misspelling matches the CLIP checkpoint convention).
- 32 encoder layers, each with:
  - `layer_norm1`, `layer_norm2`: `(1280,)` weight + bias.
  - `self_attn.{q,k,v,out}_proj`: `(1280, 1280)` weight + `(1280,)` bias each.
  - `mlp.fc1`: `(1280 -> 5120)`, `mlp.fc2`: `(5120 -> 1280)`, both with bias; GELU activation.
- Position information is a 3D RoPE (`rope_mode="3d"`, `theta=10000`) over (temporal, height, width); attention uses no causal mask.

Projection into the text stream:

1. `multi_modal_projector`: `linear_1 (1280 -> 6144)` -> GELU -> `linear_2 (6144 -> 6144)`, per patch (with bias).
2. `2×2` neighbouring projected patches are concatenated along the channel dim to width `6144 * 4 = 24576`.
3. `patch_merge_mlp`: `linear_1 (24576 -> 6144)` -> GELU -> `linear_2 (6144 -> 6144)` (with bias), producing one text-width token per merged patch group.

The merged image/video tokens are scattered into the text embedding sequence at the `image_token_index=200025` / `video_token_index=200026` placeholder positions. `image_grid_pinpoints` enumerates the allowed dynamic-resolution tilings up to `2016×2016` in 336-pixel steps.

## 8. On-Disk Tensor Layout and dtypes

The release is bfloat16 with **no weight quantization config** (`config.json` has no `quantization_config`).

dtypes observed in the safetensors headers:

- bfloat16: `embed_tokens`, `lm_head`, all attention projections, all Q/K/index norms, all RMSNorm weights, all dense-MLP / shared-expert / routed-expert weights, the projector and patch-merge weights, and all vision-encoder weights.
- float32: `block_sparse_moe.gate.weight`, `block_sparse_moe.e_score_correction_bias`, and the vision `patch_embedding.weight`.

`model.safetensors.index.json` `metadata.total_size` is `869,157,697,024` bytes across 59 shards. Community FP8/NVFP4/GGUF builds (e.g. `brandonmusic/MiniMax-M3-NVFP4`, `unsloth/MiniMax-M3-GGUF`) are downstream re-quantizations, not part of the official checkpoint.

## 9. Weight Inventory

The weight index contains exactly `23416` named tensors. The structure decomposes cleanly:

Language top-level (3): `language_model.model.embed_tokens.weight`, `language_model.model.norm.weight`, `language_model.lm_head.weight`.

Per dense layer (layers 0-2), 11 tensors each:
- 2 layernorm (`input_layernorm`, `post_attention_layernorm`)
- 4 attention projections (`q/k/v/o_proj`)
- 2 attention norms (`q_norm`, `k_norm`)
- 3 dense MLP (`mlp.gate_proj/up_proj/down_proj`)

Per sparse layer (layers 3-59), 401 tensors each:
- 2 layernorm
- 4 attention projections + 2 attention norms
- 4 indexer tensors (`index_q_proj`, `index_k_proj`, `index_q_norm`, `index_k_norm`)
- 2 router tensors (`gate.weight`, `e_score_correction_bias`)
- 3 shared-expert tensors
- `128 * 3 = 384` routed-expert tensors

Vision (515): `patch_embedding` (1) + `pre_layrnorm` (2) + 32 encoder layers × 16 tensors (512).

Projector (8): `multi_modal_projector.linear_{1,2}.{weight,bias}` (4) + `patch_merge_mlp.linear_{1,2}.{weight,bias}` (4).

Check: `3 + 3*11 + 57*401 + 515 + 8 = 3 + 33 + 22857 + 515 + 8 = 23416`. ✓

The highest language-model layer index is `59`; the highest vision layer index is `31`.

## 10. MTP Metadata Versus Released Weights

The serialized `text_config` advertises `num_mtp_modules=7` and `num_nextn_predict_layers=1`, but:

- The released weight map contains no `mtp.*` tensors.
- The upstream `transformers` model declares `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`, i.e. it tolerates/ignores MTP weights if present.
- vLLM ships a `minimax_m3/nvidia/mtp.py` (and SGLang an MTP path) for speculative decoding when MTP weights exist.

Practical consequence (same as M2.7): the released MiniMax-M3 checkpoint is a 60-layer causal VL-MoE model with no MTP weights. The MTP fields are forward-looking metadata, not a populated part of the shipped parameter set. M3 raises the advertised module count from M2.7's 3 to 7.

## 11. Tokenizer, Special Tokens, and Prompt Formatting

M3 ships a GPT-2-style BPE tokenizer (`vocab.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) with added special tokens in the `200000+` range.

### Special tokens

- `special_tokens_map.json`: `bos_token = "]~b]"`, `eos_token = "[e~["`.
- From `added_tokens.json` (selected):
  - `"]~b]"` -> `200019`, `"[e~["` -> `200020`, `"]!d~["` -> `200021`, `"]~!b["` -> `200034`.
  - `]<]image[>[` -> `200025`, `]<]video[>[` -> `200026` (the multimodal placeholders referenced by `image_token_index`/`video_token_index`).
  - vision span markers `]<]start of image[>[`/`]<]end of image[>[` (200029/200030), video equivalents (200031/200032), `]<]vision pad[>[` (200033), frame markers (200054-200056).
  - reasoning/tool tokens: `<think>` 200050, `</think>` 200051, `<tool_call>` 200052, `</tool_call>` 200053, `<mm:think>` 200059, `</mm:think>` 200060.
  - a large block of code/agent tokens (FIM, repo/commit, `<edit_file>`, `<function_call>`, etc.).

### Generation metadata

`generation_config.json`: `do_sample=true`, `temperature=1.0`, `top_p=0.95`, `bos_token_id=200019`, `eos_token_id=200020`. The model card additionally recommends `top_k=40`.

### Reasoning modes

The chat template implements a `thinking_mode` switch with three values, wrapping reasoning in `<mm:think>` / `</mm:think>` (tokens 200059/200060):

- `enabled` — always think step by step before responding.
- `adaptive` — model decides per turn (the default when unset).
- `disabled` — no thinking output.

### Metadata notes

As with M2.7, the BOS convention is not single-sourced: `special_tokens_map`/`generation_config` use `200019` (`]~b]`), while the upstream text config defaults `bos_token_id=200034` (`]~!b[`). Exact prompt parity requires picking one convention rather than assuming all artifacts agree.

## 12. Comparison to MiniMax-M2.7

| Aspect | M2.7 | M3 |
| --- | --- | --- |
| Modality | text-only | native image/video/text VL |
| Layers | 62 (all causal, all MoE) | 60 (3 dense + 57 sparse MoE) |
| Hidden size | 3072 | 6144 |
| Q / KV heads | 48 / 8 | 64 / 4 |
| Attention | dense GQA | dense GQA (3 layers) + MSA block-sparse (57 layers) |
| Q/K norm | full-vector | per-head |
| RMSNorm | standard | Gemma `(1+w)` |
| Experts | 256, top-8, no shared | 128, top-4, 1 shared |
| Expert act / inter | SwiGLU 1536 | SwiGLU-OAI 3072 (dense MLP 12288) |
| Routed scaling | none | `2.0` |
| Context | 196608 | 1048576 |
| Weights on disk | FP8 E4M3 (128×128 tiles) | bfloat16 (unquantized) |
| MTP modules (metadata) | 3 | 7 |

## 13. Implementation Notes for Engineers

A checkpoint-matching implementation reproduces these facts:

- VL wrapper: CLIP ViT (32 layers, hidden 1280, 3D RoPE) -> GELU projector -> 2×2 patch merge -> scatter into text embeddings at tokens 200025/200026.
- 60-layer causal backbone, hidden 6144, untied embeddings/LM head, vocab 200064.
- Layers 0-2: full causal attention + dense SwiGLU-OAI MLP (inter 12288).
- Layers 3-59: MSA block-sparse attention + MoE.
- Attention: 64 Q heads, 4 KV heads, head dim 128; per-head Gemma Q/K RMSNorm; partial RoPE on the first 64 dims; base 5e6.
- MSA: 4-head lightning indexer (single shared index-key head, index dim 128), max-pool keys into 128-token blocks, max over heads, always keep 1 local block, select top-16 blocks per query; indexer is selection-only (no value path).
- MoE: float32 sigmoid router, additive `e_score_correction_bias` for selection, top-4, renormalize raw sigmoid weights, scale routed sum by 2.0, add 1 shared expert (inter 3072), routed expert inter 3072.
- SwiGLU-OAI activation everywhere in the MLPs: `gate ≤ 7`, `up ∈ [-7,7]`, `glu = gate·σ(1.702·gate)`, `out = down((up+1)·glu)`.
- Gemma-style `(1 + weight)` RMSNorm, eps 1e-6.
- bfloat16 weights (float32 router gate, correction bias, and vision patch embedding); no quantization config.
- No released MTP tensors despite `num_mtp_modules=7`.

Exact packaging parity also requires honoring the tokenizer/generation BOS mismatch and the separate on-disk `patch_merge_mlp` module described above.

## 14. References

Concrete reference files pulled into `references/` (all from the public sources below, fetched 2026-06-16):

- `references/hf_repo/` — `MiniMaxAI/MiniMax-M3` checkpoint files: `config.json`, `generation_config.json`, `configuration_minimax_m3_vl.py`, `processing_minimax.py`, `image_processor.py`, `video_processor.py`, `preprocessor_config.json`, `special_tokens_map.json`, `added_tokens.json`, `tokenizer_config.json`, `chat_template.jinja`, `README.md`, `model.safetensors.index.json`.
- `references/transformers/` — upstream `huggingface/transformers` `models/minimax_m3_vl/` package (`configuration_`, `modeling_`, `modular_`, `processing_`, `image_processing_`, `video_processing_`).
- `references/vllm/` — `vllm-project/vllm` MiniMax-M3 support (merged PR #45381, `vllm/models/minimax_m3/`): `nvidia/model.py`, `nvidia/sparse_attention_msa.py`, `nvidia/mtp.py`, `common/sparse_attention.py`, `common/indexer.py`, `common/vision_tower.py`, `common/mm_preprocess.py`, `common/ops/{sparse_attn,index_topk}.py`, and `reasoning/minimax_m3_reasoning_parser.py`.
- `references/sglang/` — `sgl-project/sglang` MiniMax-M3 support (open PR #27944, head `JustinTong0323/sglang@34f604b`): `models/minimax_m3.py`, `models/minimax_m3_vl.py`, `models/minimax_vl_common.py`, `configs/minimax_vl.py`, `layers/attention/minimax_sparse_ops/msa.py`, `layers/attention/minimax_sparse_backend.py`, `function_call/minimax_m3.py`.

External: model card `huggingface.co/MiniMaxAI/MiniMax-M3`; technical report arXiv:2606.13392; MSA operator `github.com/MiniMax-AI/MSA`.
