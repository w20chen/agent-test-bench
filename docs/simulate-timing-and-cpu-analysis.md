# Simulate 压力测试：时间语义与 CPU 限流分析

> 本文档总结了对 `trace_collect.cli simulate` 命令在高并发回放场景下的
> 时间语义、trace 数据含义、以及 Docker `--cpus` 限流机制的完整分析。
>
> 背景：在 320 vCPU ARM 机器上，使用 40 条 SWE-rebench 真实 trace，
> 以 N ∈ {40, 80, 160, 320, 640} 个并发 agent 进行 cloud_model 回放，
> 测试不同并发数下系统吞吐和工具执行时间的变化。

---

## 目录

1. [核心概念](#1-核心概念)
2. [Virtual Timeline 回滚分析](#2-virtual-timeline-回滚分析)
3. [Collect vs Simulate：两条路径的时间语义](#3-collect-vs-simulate两条路径的时间语义)
4. [压力测试中应该关注什么](#4-压力测试中应该关注什么)
5. [Docker --cpus 限流机制详解](#5-docker---cpus-限流机制详解)
6. [--cpus=1 对压力测试的必要性分析](#6---cpus1-对压力测试的必要性分析)
7. [最终建议](#7-最终建议)

---

## 1. 核心概念

### 1.1 两种时钟

Python 提供两种时间源，在 trace 系统中各有用处：

| 时钟 | Python API | 特性 | 用途 |
|------|-----------|------|------|
| 墙上时间 (wall clock / epoch) | `time.time()` | 受 NTP 校准、闰秒影响，可漂移 | 跨系统时间对齐，trace 文件中的 `ts_start`/`ts_end` |
| 单调时间 (monotonic) | `time.monotonic()` | 不受系统时间调整影响，单调递增 | 精确测量执行耗时（`duration_ms`） |

> 关键区别：如果 NTP 在工具执行期间校准了系统时间，`time.time()` 差值可能不准确（甚至为负），
> 但 `time.monotonic()` 始终给出正确的时间间隔。

### 1.2 Trace 中的三个时间维度

每个 trace action（LLM 调用或工具执行）记录：

```
ts_start ───────────────────────── ts_end
          │                        │
          │  ← duration_ms →       │
          │  (action 自身耗时)      │
          │                        │
          └── 中间可能包含：         │
              · Event loop 调度延迟 │
              · Pipe 传输开销       │
              · 其他协程的干扰       │
```

- `ts_start` / `ts_end`：epoch 墙上时间，标记 action 的开始和结束时刻
- `duration_ms`：action 自身的执行耗时，可能小于 `ts_end - ts_start`
- 差值 `(ts_end - ts_start) - duration_ms`：**系统开销**（event loop 排队 + 通信开销）

### 1.3 Event Loop 队列失真

Python `asyncio` 是单线程协作式调度。当 640 个协程同时 sleep 到期时：

```
coroutine 1: sleep(1.0) → 到期 → 排队等 event loop 调度
coroutine 2: sleep(1.0) → 到期 → 排队
...
coroutine 640: sleep(1.0) → 到期 → 排队
                                    ↑
                              event loop 逐个处理，队尾的协程可能多等了 50ms+
```

这个排队延迟**正是压力测试想要测量的信号**——它反映了高并发下 event loop 的调度瓶颈。

> **`--workers` 修复：** `--workers N` 将 agent 分配到 N 个独立进程中，每个进程有自己的
> asyncio event loop。当 `--workers` 接近 agent 数（如 `--workers 320 --num-agents 640`，
> 每 loop 仅 2 agent）时，event loop 排队延迟趋近于零。此时观察到的延迟来自 Docker daemon
> 的容器调度、CPU/内存/IO 资源竞争，而非 Python 的 asyncio 记账开销。
> 详见 [Worker Architecture](trace-collect.md#worker-architecture--event-loop-contention)。

---

## 2. Virtual Timeline 回滚分析

### 2.1 时间线

| Commit | 说明 |
|--------|------|
| `5d6a8ce` | 引入 **Virtual Timeline**：用 `wall_start + session_virtual_s` 替代 `time.time()` 作为时间戳 |
| `9236e47` | **回滚 Virtual Timeline**，恢复 `time.time()`，保留 per-agent trace logger |

### 2.2 Virtual Timeline 做了什么

```python
# Virtual Timeline 版本
session_virtual_s = 0.0
for action in actions:
    record_ts_start = wall_start + session_virtual_s       # 不调用 time.time()
    await asyncio.sleep(intended_duration_s)                # 真实 sleep
    record_ts_end = record_ts_start + intended_duration_s   # 纯计算，不调用 time.time()
    session_virtual_s += intended_duration_s

# 当前版本（回滚后）
for action in actions:
    record_ts_start = time.time()                           # 真实墙上时间
    await asyncio.sleep(intended_duration_s)                # 真实 sleep
    record_ts_end = time.time()                             # 真实墙上时间
```

### 2.3 为什么 Virtual Timeline 对压力测试有害

Virtual Timeline 下所有时间戳都是**确定性的**——仅依赖于 source trace 的 duration 和 replay_speed，
完全不反映真实墙上时间。这意味着：

- 640 个并发 agent 的时间戳和 1 个 agent 的时间戳**看起来完全一样**
- Event loop 排队延迟被完全掩盖
- CPU 竞争导致的工具变慢不会体现在时间戳上
- **压力测试变得毫无意义**

### 2.4 回滚结论

**回滚是正确的。** 压力测试需要 `time.time()` 捕获真实的墙上时间，让系统瓶颈（event loop 延迟、
CPU 竞争、pipe 传输开销）在时间戳中暴露出来。Virtual Timeline 恰好把这些信号全部抹掉了。

---

## 3. Collect vs Simulate：两条路径的时间语义

### 3.1 Collect（真实采集）

Collect 是**真实运行 agent** 并记录 trace。agent 调用真实 LLM API，在 Docker 容器内实际执行工具。

#### LLM Call

```
before_iteration():
    _iter_start_wall = time.time()          ← ts_start (epoch)

    [发送 LLM API 请求，等待响应...]

after_iteration():
    llm_ts_end = _resolve_llm_ts_end()
        → 优先: response.extra["llm_wall_ts_end"]     (LLM 提供商的时间戳)
        → 回退: _before_exec_wall                      (LLM 返回后、工具执行前)
        → 回退: time.time()                            (after_iteration 执行时)
```

存入 TraceAction：

| 字段 | 含义 | 时钟源 |
|------|------|--------|
| `ts_start` | LLM 请求发出的 epoch 时间 | `time.time()` |
| `ts_end` | LLM 响应到达的 epoch 时间 | `time.time()`（优先用提供商给的） |
| `llm_latency_ms` | LLM 往返延迟 | **优先 API 内部计时**（OpenRouter generation_time_ms），回退墙上 |
| `llm_wall_latency_ms` | 墙上往返延迟 | `(ts_end - ts_start) * 1000` |
| `llm_timing_source` | 时间来源标记 | `"openrouter_generation_time_ms"` 或 `"wall_clock_ms"` |

#### Tool Execution

```
t0 = time.monotonic()                    ← 单调时间（不受 NTP 影响）
[在容器内实际执行工具...]
wall_ms = (time.monotonic() - t0) * 1000  ← 单调时间差值

# 转换为 epoch 近似：
mono_now = time.monotonic()
epoch_now = time.time()
tool_ts_start = epoch_now - (mono_now - t0)   ← epoch 近似值
tool_ts_end = tool_ts_start + wall_ms / 1000   ← epoch 近似值
```

存入 TraceAction：

| 字段 | 含义 | 时钟源 |
|------|------|--------|
| `ts_start` | 工具开始执行的 epoch 近似时间 | monotonic→epoch 反推 |
| `ts_end` | 工具结束的 epoch 近似时间 | `ts_start + duration_ms / 1000` |
| `duration_ms` | 工具实际执行耗时 | **`time.monotonic()` 差值**（精确） |
| `wall_ms` | 原始 monotonic 时长 | `time.monotonic()` 差值 |
| `start_mono` | 原始单调时间戳 | `time.monotonic()` |

> **设计意图**：`duration_ms` 用 `time.monotonic()` 确保精确，不受 NTP/闰秒影响。
> 并发工具共享同一个 `start_mono`，所以它们的 `ts_start` 相同（反映了它们同时开始的事实）。

#### Summary

| 汇总字段 | 含义 | 时钟源 |
|----------|------|--------|
| `elapsed_s` | 整个 agent 运行耗时 | `time.monotonic()` |
| `total_llm_ms` | LLM 延迟累计 | `llm_latency_ms` 求和 |
| `total_tool_ms` | 工具执行累计 | `duration_ms` 求和（monotonic） |

### 3.2 Simulate（回放）

Simulate 是**从已采集的 trace 回放**。LLM 调用不调真实 API（用 `asyncio.sleep` 模拟），
但工具调用**通过 Docker 容器实际执行**。

#### LLM Call（用 sleep 模拟）

```python
record_ts_start = time.time()                             # epoch
await asyncio.sleep(source_duration_s / replay_speed)      # 模拟 LLM 延迟
record_ts_end = time.time()                                # epoch
```

存入：

| 字段 | 含义 |
|------|------|
| `ts_start` | sleep 开始时的 epoch 时间 |
| `ts_end` | sleep 结束时的 epoch 时间 |
| `llm_latency_ms` | `(ts_end - ts_start) * 1000` = 实际 sleep 时间 + event loop 开销 |
| `source_llm_latency_ms` | 原始 trace 的 LLM 延迟（参考值） |

> **压力测试信号**：高并发下 `llm_latency_ms > source_duration_s / replay_speed * 1000`，
> 差值 = event loop 排队延迟。使用 `--workers` 可消除此延迟，让测量结果反映
> 真实的系统资源竞争（Docker daemon 调度、CPU/内存/IO 压力），而非 Python
> 的单线程记账开销。

#### Tool Execution —— 实际执行（host/container agent）

```python
record_ts_start = time.time()                             # epoch
duration_ms = _exec_tool()                                 # 容器内部计时（agent-side）
record_ts_end = time.time()                                # epoch
```

存入：

| 字段 | 含义 |
|------|------|
| `ts_start` | 工具开始执行的 epoch 时间 |
| `ts_end` | 工具执行完毕 + 分类 + 构造 record 后的 epoch 时间 |
| `duration_ms` | **容器内部计时**（agent-side，不含 pipe 传输和 event loop 开销） |
| `source_duration_ms` | 原始 trace 的 duration（参考值） |

> **关键差异**：`ts_end - ts_start` ≠ `duration_ms`。
> - `ts_end - ts_start` = 墙上总时间（含 pipe 传输 + event loop 调度）
> - `duration_ms` = 容器内部纯执行时间
> - **差值 = 系统调度开销**。`--workers 1` 时含 event loop 延迟；
>   `--workers` 足够大时仅含 Docker pipe 传输和系统调用开销。
>   详见 [Worker Architecture](trace-collect.md#worker-architecture--event-loop-contention)。

#### Tool Execution —— 不回放实际执行（MCP / 未知工具）

```python
record_ts_start = time.time()
await asyncio.sleep(source_duration_ms / 1000 / replay_speed)  # 模拟
duration_ms = (time.time() - record_ts_start) * 1000            # 实际 sleep 时间
record_ts_end = time.time()
```

#### Summary

| 汇总字段 | 含义 | 时钟源 |
|----------|------|--------|
| `agent_exec_s` | 整个 replay 耗时 | `time.time()` epoch |
| `total_llm_ms` | LLM 模拟延迟累计 | `(ts_end - ts_start) * 1000` 求和 |
| `total_tool_ms` | 工具时间累计 | `duration_ms` 求和（实际执行用 agent-side，回放用墙上） |

---

## 4. 压力测试中应该关注什么

跑 `run_simulate_sweep.sh`（N=40/80/160/320/640）时，以下指标随 N 的变化揭示了系统瓶颈：

### 4.1 系统整体吞吐

```
指标: max(agent_exec_s) — 所有 agent 中最长的墙上耗时
含义: 批次总耗时。理想情况下（无资源竞争），不论 N 多大，此值基本不变。
现实: 随 N 增大而增大，说明 CPU/IO/event-loop 出现瓶颈。
使用 `--workers` 可消除 event-loop 因素，使瓶颈分析聚焦于系统资源竞争。
```

### 4.2 Event Loop 排队延迟

```
指标: llm_latency_ms - source_duration_s / replay_speed * 1000
含义: asyncio.sleep 到期后，协程在 event loop 队列中等待的时间。
现实: --workers 1 时随 N 增大而增大，说明 asyncio 单线程事件循环成为瓶颈。
使用 --workers 320（推荐，每 loop 2 agent）后此指标应趋近于零。
若仍显著 > 0，检查 worker 数是否足够（N/workers <= ~4 则延迟可忽略）。
```

### 4.3 工具墙上开销

```
指标: (ts_end - ts_start) * 1000 - duration_ms   （针对实际执行的工具）
含义: pipe 传输 + event loop 调度开销。--workers 足够大时仅含 Docker pipe 传输和
      系统调用开销（event loop 延迟 → 0）。
现实: 随 N 增大而增大。使用 --workers 可分离 event loop 开销与 Docker 系统开销。
```

### 4.4 工具执行时间变化

```
指标: duration_ms（容器内 agent-side 计时） vs source_duration_ms（原始 trace）
含义: 工具在高负载下的实际执行时间是否变长。
现实: 若 N > host_cores，工具执行时间因 CPU 竞争而增加。
```

### 4.5 系统资源

```
文件: system_resources.jsonl（1 Hz 全宿主采样）
字段: CPU utilization, memory usage, disk I/O, network I/O, context switches
用途: 验证系统是否达到瓶颈，哪个资源先饱和。
```

---

## 5. Docker --cpus 限流机制详解

### 5.1 CFS 带宽控制

Docker 的 `--cpus=N` 通过 Linux cgroup 的 CFS 带宽控制器实现：

```
/sys/fs/cgroup/cpu/docker/<container_id>/
├── cpu.cfs_period_us = 100000       ← 计费周期 100ms
└── cpu.cfs_quota_us  = N × 100000  ← 每周期可用 CPU 微秒数
```

语义：**在每个 100ms 周期内，该 cgroup 内所有进程累计最多使用 `N × 100ms` 的 CPU 时间。**

### 5.2 关键理解：不是"占用墙上时间"

```
--cpus=1  →  cpu.cfs_quota_us = 100000
```

如果容器内进程只跑了 10ms CPU 时间就结束了：
- 这 10ms 消耗 10ms 配额
- 剩余 90ms 配额不会"浪费"，容器可以继续使用
- 不影响其他容器，不占用额外的墙上时间

如果容器内进程跑了 150ms CPU 时间（跨两个周期）：
- 周期 1：跑满 100ms → 被 throttle，暂停
- 周期 2：剩余 50ms 跑完 → 墙上时间 ≈ 150ms（包含跨周期的等待间隙）

### 5.3 多线程容器才是限流发挥作用的地方

CFS 以**线程**为调度单位。当容器内有多个线程时：

```
不设 --cpus（无上限）：
  container A: 1 thread  → 1 scheduling entity → 分到 0.5 核（在 640 containers × 320 cores 下）
  container B: 4 threads → 4 scheduling entities → 分到 2.0 核  ← 抢走了！

设 --cpus=1（硬上限）：
  container A: 1 thread  → 1 scheduling entity → 上限 1 核 → 实际 ~0.5 核
  container B: 4 threads → 4 scheduling entities → 上限 1 核 → 4 线程共享 1 核 → 每个 0.25 核
```

**设不设 `--cpus` 的差异只在容器内多线程时体现。**

### 5.4 SWE-rebench 场景分析

SWE-rebench 的典型工具：

| 工具类型 | 线程数 | 受 `--cpus=1` 影响？ |
|----------|--------|---------------------|
| `git diff`, `git checkout` | 单线程 | 否 |
| `bash`, `sed`, `grep` | 单线程 | 否 |
| `python script.py` | 单线程（通常） | 否 |
| `pip install` (编译扩展) | 多线程 (`-j`)  | **是** |
| `pytest` | 单线程（默认，`-n auto` 才多进程） | 多数否 |
| `make -j` | 多进程 | **是** |

绝大多数工具是单线程的。但 SWE-bench 任务可能包含编译扩展、运行测试等操作，少数任务会触发多线程/多进程工具。`--cpus=1` 确保这些少数情况不会破坏公平性。

---

## 6. --cpus=1 对压力测试的必要性分析

### 6.1 定量分析

假设 640 个容器，320 个物理核心，每个容器跑一个单线程工具：

| | 不设 `--cpus` | 设 `--cpus=1` |
|---|---|---|
| 每个容器的 CPU 份额 | ~0.5 核（CFS 进程公平） | ~0.5 核（CFS 进程公平） |
| 与 host 核数的关系 | 640 进程 / 320 核 = 各 0.5 核 | 同上 |
| 是否有容器能超过 1 核 | 可能（如果其他在 sleep） | **不会**（硬上限） |

对于单线程工具，**结果几乎完全相同**。

### 6.2 多线程工具的公平性

假设在 640 个容器中，有 10 个跑了 `pip install`（自动并行编译），其余 630 个跑单线程命令：

| | 不设 `--cpus` | 设 `--cpus=1` |
|---|---|---|
| 10 个多线程容器 | 每个可能抢到 2-4 核 | 每个严格 ≤1 核 |
| 630 个单线程容器 | 份额被挤占，变慢 | 不受影响 |
| 可复现性 | 不可复现（取决于哪 10 个恰好在同一时刻并行编译） | 可复现 |
| 跨 N 可比性 | 部分 N 下可能更多任务恰好并行编译 | 始终可比 |

### 6.3 结论

**单线程工具为主时，`--cpus=1` 几乎不影响结果；但少量多线程工具可能导致不公平。**

`--cpus=1` 是一个**防御性设置**，代价为零（对单线程工具无影响），收益为防止多线程工具破坏压力测试的公平性。

---

## 7. 最终建议

### 7.1 代码状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Virtual Timeline | ✅ 已回滚 | 使用 `time.time()` 捕获真实墙上时间 |
| Per-agent trace logger | ✅ 保留 | 每个 agent 独立写 trace 文件，避免共享文件竞争 |
| CPU_LIMIT 默认值 | ✅ 已改为 1 | `run_simulate_sweep.sh` 默认 `--cpu-limit 1` |

### 7.2 运行时建议

```bash
# 标准压力测试（推荐）：每个 agent 严格 1 核上限
export SOURCE_TRACES_DIR=/path/to/traces
bash scripts/run_simulate_sweep.sh

# 对比实验：无 CPU 限制，测试自然调度行为
CPU_LIMIT="" bash scripts/run_simulate_sweep.sh

# 局部测试
SWEEP_VALUES="40 80" bash scripts/run_simulate_sweep.sh
```

### 7.3 结果分析清单

压力测试完成后，分析以下维度随 N 的变化：

- [ ] `max(agent_exec_s)` — 批次总耗时是否随 N 线性增长
- [ ] `llm_latency_ms` vs 预期 — event loop 排队延迟
- [ ] 实际执行工具的 `(ts_end - ts_start) * 1000 - duration_ms` — pipe + 调度开销
- [ ] 实际执行工具的 `duration_ms` vs `source_duration_ms` — CPU 竞争导致的工具变慢
- [ ] `system_resources.jsonl` 中的 CPU 利用率曲线 — 是否达到饱和
- [ ] `system_resources.jsonl` 中的 context switch 数量 — 调度开销
