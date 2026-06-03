# Recording — 记录了什么

每次 agent 跑一个任务，会在 `{trace_root}/recordings/attempt_NNNN/` 下产出：

```
attempt_NNNN/
├── meta.json                ← 整个 attempt 一份
└── iter_NNNN/               ← 每次 chat() 调用一份
    ├── .done                ← 完整性 sentinel；没有就是半成品
    ├── segments.json
    ├── attention.npz
    ├── routing.npz          ← 仅 MoE 模型
    ├── kv_eviction.npz      ← 仅当 KV policy 在录制（与下条互斥）
    └── sparse_attention.npz ← 仅当 sparse attention method 在录制
```

---

## `meta.json` — attempt 级

- **模型信息**：HF model 名 / commit hash / 层数 / 头数 / 专家数 / 是否 MoE
- **运行环境**：torch / transformers / accelerate 版本 / CUDA 版本 / NVIDIA driver 版本
- **KV policy 配置**：policy 名 / budget / sink_size / recent_window / heavy_ratio / seed / prefill_mode / `prefill_score_bias`(只在 H2O+sampled prefill 才 True)
- **session_history**：每次 chat() 是否复用了上一次的 KV cache，命中前缀长度，是否触发 cache 重建。`used_session_cache=True` 表示当次调用挂了 `past_key_values`；False 出现在三种情况：(1) 环境变量 `OMC_DISABLE_SESSION_CACHE=1` 关闭；(2) 当前 sparse method 声明 `requires_full_prefill=True`（目前只有 `heavy_hitter` —— 它的 score buffer 必须看到每个 prefill token 才能选 top-k）；(3) 第一次 build 之前还没有 cache 可用。**注意**：commit 7d8db25 之后 bare baseline / sliding / block_topk / quest / 所有 eviction policy 都默认开启 session cache，所以这个 flag 不再能用来区分"是否有 eviction policy"——要判别 eviction 看 `kv_policy` 块或 `kv_eviction.npz` artifact，要判别 sparse method 看 `sparse_attention` 块。
- **iters[]**：每次 chat() 调用的汇总
  - 输入 / 输出 / 总 token 数
  - 段数 / attention record 数 / routing record 数
  - decode ring buffer 丢弃了多少条
  - `recording_integrity` 自检块（completeness + done sentinel + drop count）
  - 采样元数据（实际采了多少行，per-call seed）
  - 生成元数据：seed / do_sample / temperature / top_p / top_k / repetition_penalty / 墙时 / CUDA event 耗时 / 各卡 peak memory

---

## `segments.json` — 当次 chat 的角色化 token 分段

- `call_idx`, `input_tokens`, `output_tokens`, `total_tokens`, `complete`
- `segments[]`，每段：
  - `role`：`system` / `user` / `assistant_message` / `assistant_call` / `tool_result`
  - `token_start`, `token_end`（在 full prompt+generation 序列里的位置）
  - `message_index`
  - `tool_call_id` 和 `name`（tool 调用 / 结果的溯源 id）
  - `has_content`, `has_tool_calls`
  - `first_seen_call`（这段最早出现在哪次 chat 中）
- `token_segment_id[]`：每个 token 属于哪个 segment_id

---

## `attention.npz` — 采样的 attention 行

### per-head span 统计字段（sidecar）

仅当 `RecordingConfig.per_head_stats_layers` 非空时填充（默认 `()`，建议实际模型用 `(0, 6, 12, 18, 24, 30, 36, 47)`；47 是 Qwen3-Coder-30B 最后一个 attention 层，retrieval/copy head 集中处）。无论是否启用，这 10 个字段始终写入 npz，空时形状首维为 0。

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `head_stats_layers` | `[L_s]` | i32 | 实际采样的层索引列表 |
| `head_span_mean_prefill` | `[L_s, query_head, S]` | fp16 | prefill 阶段各 head 在各 span 内 token 的 attention 均值，按 query row 平均；无 key 位置的 cell 为 NaN |
| `head_span_var_prefill` | `[L_s, query_head, S]` | fp32 | prefill 阶段 span 内 token 间方差，按 query row 平均（population variance）；无 key 位置的 cell 为 NaN |
| `head_span_query_count` | scalar | i32 | prefill 采样的 query row 数（用于解读均值分母） |
| `head_span_mean_decode` | `[L_s, T_max, query_head, S]` | fp16 | decode 各 step 各 head 在各 span 的 attention 均值；无 key 位置的 cell 为 NaN |
| `head_span_var_decode` | `[L_s, T_max, query_head, S]` | fp32 | decode 各 step span 内 token 间方差；无 key 位置的 cell 为 NaN |
| `head_span_decode_step` | `[L_s, T_max]` | i32 | 各层实际记录的 decode step 索引（-1 为 padding） |
| `head_span_decode_n` | `[L_s]` | i32 | 各层实际记录的 decode step 数（T_max = 跨层最大值） |
| `head_span_kept_token_count_prefill` | `[L_s, S]` | i32 | prefill 各 (层, segment) 实际参与累积的 key 位置总数（所有采样 query row 的 mask.sum() 之和） |
| `head_span_kept_token_count_decode` | `[L_s, T_max, S]` | i32 | decode 各 (层, step, segment) 的 kept-K 数；不足 T_max 的层用 0 填充 |

`query_head` 是 query 头数，GQA 下 ≠ KV 头数（Qwen3-Coder 是 32 query head × 4 KV head）。

**mean/var 公式（Fix 5）**：

```
mean = (1/Q) Σ_q (1/|S_q|) Σ_{k ∈ S_q} A_{q,k}
var  = (1/Q) Σ_q Var_{k ∈ S_q}(A_{q,k})   （population variance）
```

其中 Q = 该层采样的 prefill query row 数（= `head_span_query_count`），|S_q| = 该 query row 在该 segment 内实际参与计算的 key 数（per-row 的 mask.sum()）。当某 segment 对该 query row 完全无 key 位置时（mask.sum()==0），该 cell 写 NaN。

`head_span_kept_token_count_prefill[l, s]` = Σ_q mask_q.sum()，是 NaN 判断的直接分母证据，用于区分"head 不关注此 span"和"此 span 被 KV 驱逐至近零 key presence"。



每条 record 对应一个 (layer, prefill 或 decode step) 的采样快照。

- **采样规则**：prefill 每层最多 80 行（`max_prefill_queries`），用 seeded 分层抖动；decode 走 ring buffer，最多保留 ~64 个 decode step × 全部层
- **每条 record 的元信息**：layer 索引 / phase（prefill/decode）/ decode_step
- **每行（query）的内容**：
  - 该 query 在 prompt 里的绝对位置
  - 该行 attention 在所有 segment 上的归一化质量 `segment_mass[N, S]`（fp16）
  - top-k key 位置 + 权重（CSR 存储；`topk_csr_weights` fp16）
- **每条 record 聚合**：heavy-hitter（该 record 所有采样行平均后的 top-k）
- **每次 chat 聚合**：span × span attention 矩阵（normalized + raw + row_sums + 计数）

**与 sparse_attention 的交互**：当 sparse_attention method 在录制时，sparse
pre-hook 会在 softmax 前向 mask cell 注入 `-inf`，所以这里抓到的 post-softmax
`attn` 在被 sparse mask 屏蔽的位置上是 HARD ZERO，而 dense 跑同样的 query 不会
出现这种零值。下游消费 `segment_mass` / `topk_*` 的代码在 sliding-window 录制
下应预期 middle-token span 上的 mass 为 0。逐层逐步 keep-set 见
`sparse_attention.npz`。

---

## `routing.npz` — MoE 专家路由（仅 MoE 模型）

每条 record 对应一个 (layer, phase, decode_step) 的路由快照。

- 每个 token 选了哪几个专家、各自权重（top-k；`expert_weight` fp16）
- 每个 segment × 每个专家的负载（加权 + 原始 token 计数）
- 该 record 的 capacity / 预期溢出 token 数 / 预期被丢 token 数
- drop signal 模式（当前是 expected_uniform_capacity，不是真实 drop）

---

## `sparse_attention.npz` — sparse attention 决策（仅当 method 在录制）

每行对应一次 `(call_idx, layer, phase, decode_step)` 的 mask 决策。`kv_eviction`
与 `sparse_attention` 互斥；同一 attempt 内至多出现其中之一。

- **元数据**：`method_name`（`sliding` / `streaming` / `heavy_hitter` /
  `block_topk` / `quest`）
- **行键**：`record_step`（运行内全局自增）/ `record_layer` / `record_phase`
  （`prefill` / `decode`）/ `record_decode_step`（prefill = -1）
- **形状**：`query_len`（forward 行数）/ `key_len`（mask 作用的 key 长度）
- **稀疏度**：`kept_count`（未屏蔽 key 位置数）/ `density`（fp16，
  `kept_count / key_len`；key_len=0 时为 0）
- **method 自定义字段**：`extras_json` U-string 列，per-row JSON
  序列化的 method metadata；sliding 记录 effective density，dynamic methods
  记录 `selection_reason` / `selected_middle_indices` / block/page ids。

attempt 级 `meta.json["sparse_attention"]` 与 `kv_policy` 并列；iter 级
`recording_integrity["sparse_attention_records"]` 给出当 iter 该 npz 的行数。

---

## `kv_eviction.npz` — KV 驱逐决策（仅当 policy 在录制）

每行对应一个 (call_idx, layer, decode_step) 的驱逐决策。

- 基本：policy 名 / phase / pre_len / post_len / budget
- **保留**：哪些 token 位置被留下（CSR）
- **驱逐**：哪些 token 位置被踢掉（CSR）+ 驱逐原因（h2o_topk / streaming_window / score_missing 等）
- **分数**：
  - 被保留的 heavy-hitter top-k 的分数
  - **被驱逐 token 的分数全集**（CSR）—— 可以回答"被踢掉的是不是分低"

---

## 重要语义

- **`.done` 哨兵必须先验**：没有 `.done` 或 `segments.json["complete"] != true` 的 iter 必须跳过，不是 partial load
- **session diverged=true**：cache 重建（常因 Qwen3 `<think>` token 模板差异），不影响正确性
- **`prefill_score_bias` False** ≠ "无任何 bias"，只是"未受采样偏置"；采样带来的 spatial bias 始终在
- **decode ring buffer 丢弃是显式的**：丢了多少条在 meta 里有数，不是静默
- **counterfactual replay 不录**：跨 KV policy 的 per-call 对照靠多任务 aggregate 成功率推趋势，不靠同一 prompt 在不同 policy 下重放
