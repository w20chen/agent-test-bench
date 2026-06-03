# sparse_attention — 插件层 schema

`src/serving/sparse_attention/` 与 `src/serving/kv_policies/` 在结构上**对
称、运行时互斥**：

| 子系统 | 改什么 | 何时介入 | artifact |
|--------|--------|---------|----------|
| `kv_policies` | K/V cache 内容（物理 drop） | `Cache.update()` | `kv_eviction.npz` |
| `sparse_attention` | per-query attention mask（保留 K/V） | self_attn pre-forward hook | `sparse_attention.npz` |

CLI 层 `validate_attention_method_exclusivity()` 在 config 解析完之后、模型
加载之前抛 `ValueError`；Provider `__init__` 再兜一层 assert。

---

## 协议（`base.py`）

```python
class BaseSparseAttention(Protocol):
    name: str
    observe_only: bool
    def build_additive_mask(*, layer_idx, query_len, key_len, phase,
                            decode_step, device, dtype, context) -> torch.Tensor | None: ...
    def record_metadata(*, layer_idx, phase, decode_step) -> dict: ...
```

- 返回 4D additive mask，broadcast 到 `[B, H, Q, K]`。`0` = 允许，`-inf` =
  屏蔽。
- pre-hook 会把 `attention_mask` 设成 non-None；HF SDPA 的 implicit causal
  不能再依赖。**method 必须自己处理 causal**，Q>1 prefill mask 要带
  upper-triangular `-inf`。
- `record_metadata` 返回的 dict 直接进 recorder 的 `extras_json` 列。

---

## sliding（`sliding.py`）

```
k ∈ [0, sink_size)                     → 0
k ∈ [key_len - recent_window, key_len) → 0
其余                                     → -inf
```

Decode（`Q == 1`）时每个 query row 共用同一份 key mask（mask shape
`[1,1,1,K]`，靠 broadcast）。Prefill（`Q > 1`）时返回 `[1,1,Q,K]`，
在同一份 sink+recent key pattern 上叠加 per-row causal upper-triangular
cut，避免 query row 看到 future tokens。

约束（构造时校验）：

- `sink_size >= 0`、`recent_window >= 0`
- `sink_size + recent_window > 0`（否则全屏蔽，无意义）

当 `sink_size + recent_window >= key_len`，两段已覆盖全长度，等效 dense
attention。

`streaming` 是 `sliding` 的 CLI/YAML alias，语义相同。

---

## dynamic decode methods

第一批动态方法都默认 `phase_scope: decode_only`：prefill 返回 dense causal
mask，decode 才做 sparse selection。这样避免在 prefill 中 materialize 或记录
大规模 per-query keep set，也避免把 full dense attention 事后 top-k 伪装成
可 enforce 的方法。

| method | 选择信号 | keep set |
|--------|----------|----------|
| `heavy_hitter` | 之前 forward 通过 `AttentionBus` 发布的 post-softmax attention 累积分数 | sink + recent + top middle token |
| `block_topk` | 当前 query 与 cached/current K 的 pre-softmax QK logits | sink + recent + top contiguous blocks，按 budget 截断 |
| `quest` | Quest-style KV page min/max envelope 对当前 query 的上界估计 | sink + recent + top pages，按 budget 截断 |

Research integrity 边界：

- `block_topk` / `quest` 只使用 forward 前可得的 Q/K states，不读取当前
  forward 的 post-softmax attention。
- `heavy_hitter` 只使用历史已观察分数；当前 step 的 attention 会在 mask 决策
  之后才发布到 bus。`prefill_observe_mode = "full"` 保证 score buffer 覆盖
  prefill 所有 query row（完整 full-prefill bus path），避免只用 80-row
  sampled subset 导致 decode 选择退化。
- 当前 HF backend 仍是 research/recording backend；SDPA 加 mask 不等于真实
  sparse kernel 加速。

---

## SparseAttentionConfig（frozen dataclass）

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `name` | `"none" \| "sliding" \| "streaming" \| "heavy_hitter" \| "block_topk" \| "quest"` | — | 必填 |
| `record` | bool | `False` | 是否写 `sparse_attention.npz` |
| `observe_only` | bool | `False` | True = 旁路只录，不改 attention_mask；可与 KV eviction 共存 |
| `sink_size` | int | `4` | sink-prefix 长度 |
| `recent_window` | int | `256` | recent-tail 长度 |
| `budget` | int \| null | `null` | dynamic method 必填；总 keep token 目标上限 |
| `block_size` | int | `16` | `block_topk` block / `quest` page size |
| `score_reduction` | `"max" \| "mean"` | `"max"` | block/page score 聚合 |
| `phase_scope` | `"decode_only"` | `"decode_only"` | dynamic method 目前只支持 decode sparse |

---

## 两种 mode

| mode | 写 kwargs["attention_mask"]? | 改 generation? | 互斥 `--kv-policy` |
|------|----------------------------|---------------|-------------------|
| **enforce** (默认) | 是，强制 sparse mask | 是 | 是 |
| **observe-only** | 否，全 attention 跑 | 否（greedy 下文本逐字相同） | 否，可共存 |

**为什么需要 observe-only**：sliding (sink=4, recent=N) enforce 在长上下文上
通常让 model 丢失中间 task definition / 错误信息 / 之前工具调用结果，
generation 退化成 hallucination。Observe-only 让我们**不毁掉 generation**
的前提下，旁录"假如开 sparse 会选哪些 key"，再和真实 `attention.npz` 做
offline cross-reference 计算 selection 质量（见
`scripts/recoding_figures/score_sparse_selection.py`）。

Pre-hook 在 observe-only 下：method 自己在 `build_additive_mask` 内先更新
record state（`_last_kept_count` / `_last_metadata`），然后 `return None`
跳过 `[1,1,Q,K]` tensor 物化。Hook 看到 None 后短路 writeback。Recorder
仍正常 append（`kept_count` / `density` / `extras_json` 二进制不变）。
区别是 prefill 不再每层白构造 ~5GB fp16 tensor —— 实测让 block_topk 在
fix-git 长 context 上从 220s/turn 降到 ~60-80s/turn（接近
`--record-internals` baseline）。

Recording 完整性字段 `sparse_attention_observe_only` 让 loader 端可分流两种
trace。

---

## CLI / YAML

镜像 `kv_policies` 同款 overlay 顺序：

1. `--sparse-attn-config PATH` 提供 base map
2. 显式 `--sparse-attn-*` 覆盖 YAML
3. `--sparse-attn none`（argparse 默认）**不会**清掉 YAML 提供的 `name`

CLI flag：
- `--sparse-attn {none,sliding,streaming,heavy_hitter,block_topk,quest}` / `--sparse-attn-sink-size N` / `--sparse-attn-recent-window N`
- `--sparse-attn-budget N` / `--sparse-attn-block-size N`
- `--sparse-attn-score-reduction {max,mean}` / `--sparse-attn-phase-scope decode_only`
- `--sparse-attn-record` / `--no-sparse-attn-record`
- `--sparse-attn-observe-only`（store_true；observe 模式开关）
- `--sparse-attn-config PATH`

```yaml
# configs/sparse_attention/sliding_b260.yaml
name: sliding
sink_size: 4
recent_window: 256
record: true
observe_only: true     # 与 --kv-policy h2o 共存时必须开
```

```yaml
# configs/sparse_attention/quest_b1024.yaml
name: quest
budget: 1024
sink_size: 4
recent_window: 256
block_size: 16
score_reduction: max
phase_scope: decode_only
record: true
observe_only: true
```

---

## `sparse_attention.npz` schema（per-call 写盘）

每行对应一次 `(call_idx, layer, phase, decode_step)` 的 mask 决策。

| 字段 | 形状 | dtype | 含义 |
|------|------|-------|------|
| `call_idx` | scalar | i32 | 写盘时的 chat 编号 |
| `method_name` | scalar | U16 | sparse method name |
| `record_step` | `[R]` | i32 | append 时的 row index（运行内全局） |
| `record_layer` | `[R]` | i32 | 层号 |
| `record_phase` | `[R]` | U7 | `prefill` / `decode` |
| `record_decode_step` | `[R]` | i32 | decode 阶段步号；prefill = -1 |
| `query_len` | `[R]` | i32 | 该 forward 的 query 行数 |
| `key_len` | `[R]` | i32 | mask 作用的 key 序列长度 |
| `kept_count` | `[R]` | i32 | sparse pattern 下未被屏蔽的 key 位置数。sliding（key-uniform）下是 key mask 精确值；future per-query method（Quest / MInference 等）写 query rows 的 MEAN kept count |
| `density` | `[R]` | fp16 | `kept_count / key_len`（key_len=0 时 = 0）。注意 prefill 时这不是 causal 后的真实 visible-cell 密度 |
| `extras_json` | `[R]` | object/U | method 自定义元数据的 JSON 字符串 |

`extras_json` 例（sliding）—— sink_size / recent_window 已在 attempt 级
`meta.json` 的 `sparse_attention` block，行级不再重复；这里记录 query/key
相关的有效 mask 摘要：

```json
{
  "effective_kept_count_sum": 25,
  "effective_density": 0.390625
}
```

`effective_kept_count_sum` 是 sparse pattern + causal 后所有 query rows 的
可见 `(q, k)` cell 总数；`effective_density = effective_kept_count_sum /
(query_len * key_len)`。Decode Q=1 时它等于顶层 density；prefill Q>1 时通常
更小，即使 sliding window 已覆盖全 key，causal lower triangle 仍会让
`effective_density < 1.0`。

dynamic method 的 `extras_json` 记录 `budget` / `phase_scope` /
`selection_reason` / `selected_middle_count` / `selected_middle_indices`，以及
`block_topk.selected_blocks` 或 `quest.selected_pages`。这些字段支持
`scripts/recoding_figures/score_sparse_selection.py` 在 observe-only trace 上
重建 keep set；仍不写 raw dense mask。

---

## meta.json 增量

attempt 级与 `kv_policy` block 并列新增：

```json
"sparse_attention": {
  "method": "sliding",
  "sink_size": 4,
  "recent_window": 256,
  "record": true,
  "observe_only": false,
  "budget": null,
  "block_size": 16,
  "score_reduction": "max",
  "phase_scope": "decode_only"
}
```

iter 级 `recording_integrity` 内追加：

```json
"sparse_attention_recording_enabled": true,
"sparse_attention_observe_only": false,
"sparse_attention_records": <int>,
"sparse_attention_expected_records": <int>,
"sparse_attention_records_match_expected": true,
"sparse_attention_expected_layers": <int>,
"sparse_attention_observed_layers": <int>,
"sparse_attention_hook_invocations": <int>,
"sparse_attention_hooks_per_layer_min": <int>,
"sparse_attention_hooks_per_layer_max": <int>,
"sparse_attention_hooks_balanced": true
```

下游消费时可用 `sparse_attention_observe_only` 分流：observe 跑出来的
`attention.npz` 是 dense ground truth，可直接做"假设 sparse 方法的离线
recall@k / mass coverage"分析；enforce 跑出来的 `attention.npz` 在被 mask
的 key 位置上是 hard zero，**不能**当 dense 用。

---

## 不做（明确边界）

- 不实现 MInference / DuoAttention / NSA（留下一轮 PR）
- 不和 `kv_policies` 组合（互斥强制）
- 不写完整 mask 到 npz（density / effective_density / selected indices 等
  统计足够；如需要 raw mask 再单开 debug flag）
- 不承诺 HF SDPA 下的 sparse kernel speedup
