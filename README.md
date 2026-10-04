# wfos

> WFOS is an Agent Harness for reproducible execution, independent evaluation,
> benchmarking, experimentation, and evidence-driven skill optimization.
>
> WFOS 是一个面向 Agent 执行的 Harness，用于提供隔离执行、独立评估、Benchmark、
> 实验比较，以及基于证据的 Skill 优化与安全发布 / 回滚能力。

它的核心不是"Agent 本身"，而是**让 Agent 的执行结果变得可验证、可比较、可追踪、可优化**。

它把"改一次代码"当成一个**可测量、可比较、可回溯**的对象：在隔离环境里跑，把发生
的事记成只增不改的证据，用独立于 agent 的判定器给它打分，和冻结的基线比较，从真实
的恢复记录里推导改进候选，再由人决定要不要让它进生产。

```
Task → 隔离执行 → Trace → Evaluation → Benchmark → Baseline/Compare
     → Experiment → RSI Candidate → 显式 Promotion → Rollback
```

只有一条硬规矩贯穿全部：**任何结论都必须能追到一次真实发生过的运行。**
所以 `unknown` 不是 `pass`，没报告的 token 是 `None` 不是 `0`，候选不能自己变成
正式版本。

---

## 安装

```bash
python -m pip install -e .          # 运行时依赖：mcp / pydantic / httpx
python -m pip install -e ".[dev]"   # 加上 pytest / ruff
wfos --help
```

**Clone 后可直接安装并运行内置的 mock / benchmark 流程** —— `WFOS_PROVIDER=mock`
是确定性离线大脑，基准、实验、判定、晋升、回滚全都能在它上面跑通，不需要任何凭据。
**使用真实 LLM Provider 时，需要额外配置对应 Provider 的凭据和模型**
（`WFOS_PROVIDER` + 该 Provider 的 API key 环境变量，见
[`docs/operations.md`](docs/operations.md) §1）。仓库不存放任何凭据。

要求 Python ≥ 3.10。仓库根目录要作为工作目录 —— `benchmark/` 与 `experiments/`
是相对路径（见文末限制）。

## 最小运行

```bash
export WFOS_PROVIDER=mock        # Windows: set WFOS_PROVIDER=mock
wfos                             # 交互式会话：直接说你想做什么
wfos chat --once "帮我看看这个项目"   # 只跑一轮，便于脚本化
wfos run "新增一个模块，改 app.py"    # 或者跑一遍完整状态机
wfos status --json
wfos history --json
wfos trace show <run_id>         # append-only 轨迹，含已被删除的步骤
wfos result <run_id>             # 当前工作状态（和 trace 不是一回事）
```

`wfos`（不带参数）进入**交互模式**：直接说想做什么，agent 会自己读文件、改文件、
跑构建与测试，每轮都在同一个 run 里，`wfos trace show` 能回看全部往来。写/改项目内的
文件直接生效，**删除需要审批**（`wfos pending` / `wfos approve`）。

## 交互式会话

`wfos`（不带参数）或 `wfos chat` 进入交互模式：

```bash
wfos                                    # 直接说你想做什么
wfos chat --once "帮我看看这个项目"      # 只跑一轮，不读 stdin，便于脚本化
```

agent 会自己决定调哪些工具 —— 读文件、grep、看 git 状态、改文件、跑构建与测试，
然后回答你。实测一次真实会话：

```
> 给 app.py 里的 compute 加上类型标注和 docstring
… workspace.list_files → git.status → workspace.read ×2 → workspace.search
  → wiki.search → workspace.patch → workspace.read → build.check → test.run
「已完成 app.py 中 compute 的类型标注与 docstring 补充。既有断言未回归。」
```

**它是普通 run。** 一次会话是一个 run：出现在 `wfos history`，`wfos trace show <id>`
能回看到每一轮（`chat.user` / `tool.called` / `chat.assistant` 三类事件，actor 分别是
`human` / `model`）与每一次文件改动（`wfos result` 显示归因）。

**权限沿用现有策略。** 改/写项目内的文件直接生效（和 `wfos mcp --role implementer` 一致），
**删除需要审批** —— 被拦下的删除会自动生成一条 `<tool>:<path>` 的待审批项，`wfos approve`
之后下一轮即可执行。

`wfos run "…"` 则是另一条路：它按关键词在 feature / bugfix 之间路由，走完整状态机
（11 个状态）—— 认不出关键词时会报错，此时用 `--kind feature` 或 `--kind bugfix` 指定。

要跑一个**声明式任务**、在自己的工作区里：

```bash
wfos task run benchmark/smoke feature-happy-path --workspace .data/tasks
```

它会打印这个运行的库在哪、以及该用哪条命令去判定它 —— 因为运行住在自己的库里，
`evaluate` 不指过去是看不到的。

## Benchmark / Baseline / Compare

```bash
# 跑一层（smoke / regression / challenge），每个任务独立 workspace + 独立 DB
wfos benchmark run regression --workspace .data/bench

# 冻结一份基线。已经存在的基线**拒绝覆盖** —— 候选不能靠重跑题目来产生新基线
wfos baseline create --suite regression --out base.json --workspace .data/bws

# 拿新记录和基线比，得到五种判定之一
wfos compare base.json fresh.json --json
```

`compare` 的输出是 `regression / improvement / unchanged / unknown / noise` 的计数。
`unknown` 既不是变好也不是变坏：有一侧没测、或环境不同，比较就降级为 `unknown`
而不是硬说"没变"。

判定一次具体的运行：

```bash
wfos evaluate <run_id> benchmark/regression build-error-is-classified --json
```

判定读的是运行留下的记录，不问 agent 自己觉得怎么样。退出码 `0` 表示通过、`1` 表示
判定为不通过、`2` 表示参数有问题 —— 判定失败是**结果**，不是命令失败。

## Experiment

实验把一个矩阵展开成互相隔离的 cell，每个 cell 有自己的 workspace、库和 run：

```bash
cat experiments/model-sweep.json
# matrix: {"model": ["mock", "mock-alt"], "config": [null, {...}]}   → 4 个 cell
wfos experiment run experiments/model-sweep.json \
    --workspace .data/sweep --out sweep.json
wfos experiment show sweep.json --json
```

## RSI：候选 → 晋升 → 回滚

候选**从证据推导**，不是写出来的。一个 skill 候选的文本来自一次真实的运行：那次
运行命中了一个失败类，**并且越过了它**。一个没人解决过的失败只产生 finding，不产生
候选。

```bash
# 1. 从实验结果里推导候选（候选住在 candidates 表，不是 skills 表）
wfos rsi propose sweep.json
wfos rsi list --json

# 2. 让候选跑同一套题，和基线比较 → PASSED / REJECTED / NOT_PROMOTABLE
wfos rsi evaluate skill:test_failure --version 1 \
    --baseline base.json --workspace .data/cand
wfos rsi show skill:test_failure --json

# 3. 晋升是人的决定。这是全仓唯一能把候选写成正式版本的命令
wfos rsi promote skill:test_failure --version 1 \
    --baseline base.json --actor rog --reason "通过门禁"

# 4. 回滚只移动指针，不删除新版本
wfos rsi rollback test_failure 1 --actor rog --reason "线上有问题"
wfos rsi history test_failure --json
wfos rsi promotion show <promotion_id> --json
```

`PASSED` 的含义是"值得人看一眼"，不是"已经在用了"。没有任何东西会自动调用
`promote`。

晋升门不重跑任何东西，它**重读**候选记录下来的判定，所以可以在很久以后再问一次同样
的问题。规则：

| 条件 | 结果 |
|---|---|
| 候选状态不是 `PASSED` | 拒绝，且不移动候选（没评估过 ≠ 被拒绝） |
| 溯源不完整 | `NOT_PROMOTABLE` |
| 候选自己的运行有硬门禁失败 | `REJECT` |
| `functional` / `safety` / `operational` 任一 `fail` | `REJECT` |
| 上述任一 `unknown` | `NOT_PROMOTABLE`（unknown 不是 pass） |
| 相对基线有回归 | `REJECT` |
| 声明母版本 ≠ 当前生产版本 | `REJECT` |

## JSON 输出

12 个子命令支持 `--json`：`run`、`status`、`evaluate`、`benchmark`、`compare`、`rsi`、
`experiment`、`history`、`skills`、`capabilities`、`metrics`、`baseline`。
**stdout 就是那一份 JSON 文档**，前后不混任何人类文本（进度与错误信息去 stderr）。
失败路径也是 JSON，脚本不用为错误分支写特例。

```bash
$ wfos status no-such-run --json ; echo "exit=$?"
{
  "error": "未找到运行 no-such-run",
  "runId": "no-such-run"
}
exit=1
```

退出码：`0` 成功 · `1` 没找到 / 判定不通过 · `2` 参数错误 · `3` 运行被别的进程持有 ·
`4` 缺少凭据。

## 测试

```bash
pytest -q                          # 全量
ruff check .                       # 静态检查（CI 跑的就是这两条）
python artifacts/accept_final.py   # 端到端验收：完整闭环，真跑
python artifacts/mutate_p7.py      # 变异检查：破坏保证，确认对应测试会挂
python artifacts/mutate_p6.py      # 晋升/回滚的变异检查
```

`artifacts/accept_p3.py` … `accept_p6.py` 是各阶段的验收脚本。

### CI

`.github/workflows/ci.yml`：push / pull_request 触发，`windows-latest` +
Python 3.12，三步 —— 安装、`ruff check .`、`pytest -q`。没有矩阵，因为还没有可以
变化的维度（见下）。

## 当前明确限制

- **只在 Windows 上开发和验证过。** CI 因此只跑 `windows-latest`；代码里没有故意
  的平台依赖，但 Linux/macOS 没有证据，不宣称支持。
- **`benchmark/` 和 `experiments/` 是仓库相对路径**，`pip install` 只装 `wfos`
  这个包。从别处调用需要把路径写全，或者把工作目录设在仓库根。
- **`challenge` 层需要真实 provider**，mock 跑不了。
- **`context_overflow` 在当前实现里不可产生**：它只会在 provider 的结构化错误码
  字段上产生，而目前没有 provider 会给。这是明确声明，不是遗漏。
- **五类候选只有两类有自动生产者**（`skill`、`retry_policy`），另外三类
  （`workflow_policy` / `tool_hint` / `context_policy`）只能手工构造，且一律
  `NOT_PROMOTABLE`。
- **候选库和正式 skill 库是同一个 SQLite 文件**：隔离由代码路径保证，不是存储隔离。
- **没有真正的 LLM Judge**，judge 目前是可选且非 LLM 的。

交互式会话相关的：

- **会话不是状态机。** `kind=chat` 只有一个状态、没有迁移，`wfos resume` 对它无效
  （会被安全地忽略），`wfos evaluate` 也不适用。
- **会话之间不共享上下文。** 每次 `wfos` 是一个新 run，多轮对话只在这一次会话内。
- **`WFOS_PROVIDER=mock` 下的对话是脚本化的**：它会真的调工具（列文件、按关键词读/写），
  但不是模型在决定。要真实能力得接真实 provider。
- **流式输出、slash 命令、会话切换、文件附件都没有** —— 这是刻意不做，不是未完成。
