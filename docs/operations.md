# 操作手册

> 所有命令与参数取自 `wfos/cli.py` 的真实注册表，所有输出结构取自真实运行。
> 文中标注「未实现」的，是代码里确实没有的东西。

---

## 1. 安装与环境

```bash
python -m pip install -e ".[dev]"     # 需要 Python >= 3.10
wfos --help
```

**环境变量**（`wfos/config.py`）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `WFOS_DATA` | 数据目录（库、wiki、capabilities） | `./.data` |
| `WFOS_PROJECT` | 项目根（工具可写的范围） | 当前目录 |
| `WFOS_PROVIDER` | `mock` / `anthropic` / `openai` / `gemini` | `mock` |
| `WFOS_MODEL`、`WFOS_BASE_URL`、`WFOS_API_KEY` | provider 参数 | — |
| `PYTHONIOENCODING=utf-8` | **Windows 上建议设置**，否则 CLI 的中文输出可能报编码错 | — |

**wfos 不读 `.env` 文件。** 配置只来自两个地方：`config/*.toml` 与上面这些环境变量
（`wfos/config.py` 里全是 `os.environ`，没有 dotenv 加载器）。放一个 `.env` 在仓库根
不会生效 —— `.gitignore` 里有 `.env` 规则纯粹是防止凭据被误提交。

`mock` 是确定性离线大脑，**基准、实验、判定、晋升、回滚全都能在 mock 下跑通**，
不需要任何凭据；真实工作才需要 key。

---

## 2. 退出码契约

```
0   成功；`run`/`chat` 把运行推进到了终态或等待态
1   没找到 / 判定不通过 / 存在未通过的任务 / `wfos run` 的运行以 failed 结束
2   参数或输入文件有问题（argparse 的用法错误也是 2）
3   运行被另一个进程持有（执行租约未释放）
4   缺少凭据（build_harness 抛 MissingCredentialError）
```

`wfos run` 在运行以 **failed** 结束时返回 1 —— 脚本需要一种方式区分"流程得出了结论"
和"它跑完了"。`waiting_approval` / `waiting_child` 仍然返回 **0**：那是等人接手的中断，
不是结果。

- `3` 不是任何 `cmd_*` 返回的，而是 `_run()` 捕获 `RunLockedError` 后 `raise SystemExit(3)`。
  走到这条路的命令：`run`、`resume`、`resume-child`、`approve`、`reject`。
- `4` 只在 `main()` 里产生，在命令执行之前。
- **`1` 常常是"结果"而不是"失败"**：`evaluate` 判定不通过返回 1，`benchmark run` 有任务失败返回 1，
  `task run` **没跑到终态**（例如停在审批门）也返回 1。

---

## 3. `--json` 契约

12 个子命令支持 `--json`：
`run`、`status`、`evaluate`、`benchmark`、`compare`、`rsi`、`experiment`、`history`、`skills`、
`capabilities`、`metrics`、`baseline`。

**契约**（由 `_p` / `_json` / `_seal_json` 三者在进程级保证）：

1. `--json` 时 **stdout 恰好是一份 JSON 文档**，前后不混任何人类文本；
2. 人类可读的进度与提示改走 **stderr**（`_p` 在 JSON 模式下切目标，不丢信息）；
3. **失败路径也是 JSON**。命令的错误分支本来只打一句话，`main()` 的 `finally` 里
   `_seal_json()` 会把最后一句话包成 `{"error": ...}` 补上 —— 所以一个错误分支不会破坏调用者的解析器。

```bash
$ wfos status no-such-run --json ; echo "exit=$?"
{
  "error": "未找到运行 no-such-run"
}
exit=1
```

`--json` 的键全部来自真实输出：

```jsonc
// wfos status --json（无参数 → 列表）
{ "runs": [ { "id": "...", "kind": "feature", "title": "...", "description": "...",
              "state": "...", "status": "...", "parent_run_id": null,
              "root_run_id": "...", "attempt": 0, "payload": {},
              "result": null, "error": null, "identity_hash": null,
              "key_files": {}, "owner": null, "lease_until": null,
              "session_id": "...", "origin": "interactive",
              "task_id": null, "task_version": null,
              "created_at": "...", "updated_at": "..." } ] }

// wfos run "..." --json
{ "run": { ...同上... },
  "pendingApprovals": [ { "id": "...", "run_id": "...", "action": "...",
                          "scope": null, "risk_level": "...", "status": "pending",
                          "created_at": "..." } ] }

// wfos evaluate <run> <path> <task> --json  （Evaluation.as_json）
{ "runId": "...", "taskId": "...", "taskVersion": 1,
  "verdict": "pass", "passed": true,
  "axes": { "functional": "pass", "safety": "pass", "efficiency": "unknown",
            "quality": "unknown", "operational": "pass" },
  "reasons": [], "failures": [], "hardGate": [],
  "metrics": {}, "artifacts": {...}, "judge": null }

// wfos compare <baseline> <record> --json  （ComparisonReport.as_json）
{ "counts": { "regression": 0, "improvement": 0, "unchanged": 83,
              "unknown": 61, "noise": 0 },
  "environment": "", "findings": [ { "taskId": "...", "subject": "verdict",
                                     "verdict": "unchanged",
                                     "before": "pass", "after": "pass",
                                     "detail": "" } ] }

// wfos metrics --json
{ "metrics": { "runs": 3, "steps": 12, "modelCalls": 0,
               "inputTokens": null, "outputTokens": null, "costUsd": null,
               "coverage": { ... }, "byKind": {...}, "byOrigin": {...} },
  "model": "mock", "pricing": {} }
```

> `inputTokens` 为 `null` 而不是 `0` 是**刻意的**：mock 不上报用量，"没测量"不能变成"免费"。
> 同理 `costUsd` 为 `null` 是因为 `[pricing]` 未配置。

---

## 4. CLI 教程

### 4.0 `wfos` / `wfos chat` —— 交互式会话

```bash
wfos                                   # 不带参数 = 进入交互模式
wfos chat                              # 等价写法
wfos chat --once "帮我看看这个项目"     # 只跑一轮，不读 stdin
```

| | |
|---|---|
| 用途 | 多轮对话：直接说你想做什么，agent 自主读文件、改文件、跑构建与测试 |
| 参数 | `--once <文本>`（只在这个子命令上；跑一轮就退出） |
| 输出 | **stdout 只有回答**；工具调用过程逐行打到 **stderr**（`· workspace.read app.py`），可以边跑边看 |
| 退出 | `exit` / `quit` / Ctrl+C / EOF |
| 退出码 | `0` 正常结束；`1` 仅 `--once` 且那一轮失败时（会话本身仍可用） |

**两条流是分开的**，所以 `wfos chat --once "…" > answer.txt` 拿到的是纯回答，
而过程在终端上照常可见。工具行形如：

```
   · workspace.list_files **/Equipment.cs
   · workspace.search Equipment
   · workspace.read RamanApp/Base/Equipment.cs ×3     ← 连续同样的调用合并计数
   ✗ workspace.delete app.py                          ← 被策略拦下（会生成待审批项）
```

**连续相同的调用合并成一行带 `×N`** —— 一个模型在真实项目里会反复用同样的参数调同一个
工具，逐次打印不是信息，是噪音。串行在一起但不同的调用不会被合并。细节截到 48 字符，
让整行留在 80 列终端里不折行。

回答用**朴素文本**（提示词里明确要求不用 Markdown 标记），终端里直接读。

**预算**：一轮最多 20 次工具调用。最后一轮不再提供工具，模型必须直接作答 —— 所以
"看的东西太多"最多让回答不完整，不会变成一条错误。回复里会标明这一点。

**它是普通 run**：出现在 `wfos history`，`wfos trace show <id>` 能回看每一轮
（`chat.user` / `tool.called` / `chat.assistant`）与每一次文件改动。

**它不是状态机**：`kind=chat` 没有迁移，`wfos resume` 对它无效（会被安全地忽略）。

**权限**：写/改项目内文件直接放行（与 `wfos mcp --role implementer` 一致），
**删除仍需审批** —— 被拦下的删除会自动生成一条审批，`wfos approve` 后下一轮即可执行。

### 4.1 `wfos run` —— 跑一个状态机流程

```bash
wfos run "新增一个用户模块，改 app.py"
wfos run "登录报错，结果不对" --kind bugfix
wfos run "新增一个模块" --json
```

| | |
|---|---|
| 用途 | 从自然语言请求创建并推进一个 run，直到终态 / 审批门 / 子流程等待 |
| 参数 | `text`（位置，可多词）；`--kind {feature,bugfix}`；`--title`；`--json` |
| 输出 | 人类可读：run id、状态机状态、待审批项、子流程 id。`--json`：`{"run", "pendingApprovals"}` |
| 退出码 | `0` 到终态或等待态；`1` 运行以 failed 结束；`2` 路由失败（请求里既没有新功能词也没有诊断词）；`3` 运行被持有；`4` 缺凭据 |

路由失败时 `--json` 会给 `{"error", "hint"}`。

**失败时会说明是哪一种失败**，因为两者处置完全不同：

```
运行 957607f96efa479f  [问题修复] 状态=failed
  终止原因: 流程判定终止: 历史检索已穷尽（0 提交、0 差异、0 命中）：现象无法映射到
            任何源码锚点，缺陷不可定位、不可复现
  完整推理: wfos trace show 957607f96efa479f
```

三种终止原因分别来自：

| 显示 | 含义 |
|---|---|
| `流程判定终止:` + 模型的原话 | **工作流自己的结论** —— 它看过了，判定停比猜好。没有东西需要修 |
| `执行出错:` + 异常 | **Harness 放弃** —— 有异常逃出了状态。这是缺陷，要看 |
| `迁移被拒:` + 校验消息 | 某个状态建议了一个它的机器不允许的迁移 |

记录里没有说明时写「（无终止原因 —— …）」，**不编一个理由**。

### 4.2 `wfos status` —— 看状态

```bash
wfos status              # 最近 20 个
wfos status <run_id>
wfos status --json
```

退出码：`0`；给了 `run_id` 但找不到 → `1`（`--json` 时输出 `{"error", "runId"}`）。

### 4.3 `wfos task run` —— 在隔离环境里跑一个声明式任务

```bash
wfos task run benchmark/smoke feature-happy-path --workspace .data/tasks
wfos task run benchmark/regression build-error-is-classified --workspace .data/tasks --name alt
```

| | |
|---|---|
| 用途 | 按 `TaskSpec` 驱动一次运行，环境完全隔离 |
| 参数 | `task_cmd`（只有 `run`）；`path`（任务集文件或目录）；`task_id`；`--workspace`；`--name` |
| 输出 | run id、状态、步骤数、session、失败类、改动文件、**该运行的库目录**，以及一条 `WFOS_DATA=... wfos evaluate ...` 的提示 |
| 退出码 | `0` **跑到终态**；`1` **没跑到终态**（停在审批门等）；`2` 任务集加载失败 / 没有该 task_id |

**没有 `--json`**（代码中未注册）。

### 4.4 `wfos evaluate` —— 判定一次运行

```bash
wfos evaluate <run_id> benchmark/regression build-error-is-classified
wfos evaluate <run_id> benchmark/smoke feature-happy-path --json
```

| | |
|---|---|
| 用途 | 按某个 task 的期望判定一次运行，并把判定写进 `evaluations` 表 + trace |
| 参数 | `run_id`；`path`；`task_id`；`--json` |
| 输出 | 人类可读的判定渲染，或 `Evaluation.as_json()` |
| 退出码 | `0` 判定通过；`1` **判定不通过**（这是结果）；`2` 任务集加载失败 / 没有该 task_id / **run 不存在** |

> ⚠️ 判定必须在**该运行自己的库**里做。隔离运行的 run 不在主库里，
> 需要 `WFOS_DATA=<该运行的库目录> wfos evaluate ...`（`task run` 会把目录打印出来）。

### 4.5 `wfos benchmark` —— 跑一层

```bash
wfos benchmark run smoke --workspace .data/bench
wfos benchmark run regression --json
```

| | |
|---|---|
| 用途 | 跑一层固定任务集，逐任务判定 |
| 参数 | `benchmark_cmd`（只有 `run`）；`suite` ∈ `{smoke, regression, challenge}`；`--workspace`；`--json` |
| 输出 | 每任务一行（verdict、步骤、工具数、未知轴数）+ 合计；`--json` 是 `BenchmarkRun.as_json()` |
| 退出码 | `0` 无失败；`1` 有任务失败；`2` 未知套件 |

`challenge` 层需要真实 provider（任务带 `brain: "live"`），mock 跑不了。

### 4.6 `wfos baseline` —— 冻结与检查

```bash
wfos baseline create --suite regression --out base.json --workspace .data/bws
wfos baseline check base.json
wfos baseline run            # 冻结可靠性基线（artifacts/harness-baseline.json）
wfos baseline live           # 打真实 provider 的 live 用例（不写 artifact）
```

| | |
|---|---|
| `create` | 跑一层并**拒绝覆盖**已存在的输出文件 |
| 退出码 | `0` 成功；`1` 目标已存在 / 有任务失败；`2` 未知套件 |

`create` 的 `--suite` 收的是**层名**（`regression`），不是路径（`benchmark/regression`）——
传路径会被判为未知套件并返回 2。

### 4.7 `wfos compare` —— 比较

```bash
wfos compare base.json fresh.json
wfos compare base.json fresh.json --json
```

| | |
|---|---|
| 用途 | 两份记录逐项比较，得到五种判定 |
| 参数 | `baseline`（文件）；`record`（文件）；`--json` |
| 退出码 | `0` 无 regression；`1` 有 regression；`2` 文件读不出来 |

### 4.8 `wfos experiment` —— 矩阵实验

```bash
wfos experiment run experiments/model-sweep.json --workspace .data/sweep --out sweep.json
wfos experiment show sweep.json --json
```

| | |
|---|---|
| 用途 | 矩阵展开成互相隔离的 cell，逐 cell 跑一次基准 |
| 参数 | `run`/`show`；`path`；`--out`（**拒绝覆盖**）；`--baseline`；`--workspace`；`--json` |
| 退出码 | `0` 无失败且无回归；`1` 有 cell 失败 / 有回归 / `--out` 已存在；`2` 定义文件有问题 |

### 4.9 `wfos rsi` —— 候选 / 晋升 / 回滚

`rsi_cmd` 的 8 个取值：`propose`、`show`、`evaluate`、`list`、`promote`、`rollback`、
`promotion`、`history`。

> **`rsi candidate` 这个子命令不存在**。查看候选是 `rsi list`（列出）和 `rsi show <id>`（单个）。

```bash
# ① 从实验结果推导候选
wfos rsi propose exp.json
wfos rsi propose exp.json --json

# ② 查看
wfos rsi list
wfos rsi list --json
wfos rsi show skill:regression --json
wfos rsi show skill:regression --version 2

# ③ 评测（跑同一套题 + 和基线比较）
wfos rsi evaluate skill:regression --version 2 \
     --baseline base.json --workspace .data/cand --suite benchmark/regression

# ④ 显式晋升
wfos rsi promote skill:regression --version 2 \
     --baseline base.json --actor rog --reason "通过门禁"

# ⑤ 回滚（两种写法都支持）
wfos rsi rollback regression 1
wfos rsi rollback regression --version 1
wfos rsi rollback regression 1 --actor rog --reason "线上有问题" --json

# ⑥ 查看记录
wfos rsi promotion show promo-xxxxxxxxxxxx --json
wfos rsi promotion promo-xxxxxxxxxxxx --json      # 省略 show 也可以
wfos rsi history regression
wfos rsi history regression --json
```

| 子命令 | 退出码 |
|---|---|
| `propose` | `0` 成功（**即使有候选因版本已存在而被跳过，仍是 0**）；`2` 实验结果读不出来 |
| `list` / `show` | `0` 成功；`1` 找不到候选 |
| `evaluate` | `0` 状态是 `PASSED` 或 `NOT_PROMOTABLE`；`1` 其他（如 `REJECTED`/`FAILED`）；`2` suite 为空 / 找不到候选 |
| `promote` | `0` 晋升成功；`1` 被门禁拒绝或找不到候选 |
| `rollback` | `0` 回滚成功；`1` 被拒绝（不存在 / 没被晋升记录命名 / digest 不符 / 已是当前）；`2` 没给版本 |
| `promotion` / `history` | `1` 找不到记录 / 没有版本；`0` 成功 |

### 4.10 其它命令

| 命令 | 用途 | `--json` | 退出码 |
|---|---|---|---|
| `wfos history` | 最近运行（`--limit`） | ✅ | 恒 `0` |
| `wfos trace show <run_id>` | append-only 轨迹，**含已被删除的步骤** | ❌ | `1` 找不到运行 |
| `wfos result <run_id>` | 当前工作状态（步骤/证据/工具调用/越界写入/方案偏离） | ❌ | `1` 找不到运行 |
| `wfos pending` | 待审批 | ❌ | 恒 `0` |
| `wfos approve <id> --note` / `wfos reject <id> --note` | 审批 | ❌ | `1` 审批不存在；`3` 运行被持有 |
| `wfos resume <run_id> [--force-stale]` | 恢复运行 | ❌ | `1` 找不到；`3` 被持有 |
| `wfos resume-child <child_run_id>` | 子流程完成后恢复父流程 | ❌ | `1` 找不到 |
| `wfos cancel <run_id>` | 取消 | ❌ | `1` 找不到 |
| `wfos skills` | 已沉淀的规程（**全量，含候选**，因为是查看器） | ✅ | 恒 `0` |
| `wfos capabilities` | 已观测到的 provider 能力 | ✅ | 恒 `0` |
| `wfos metrics [run_id] --limit` | token / 延迟 / 成本用量与覆盖率 | ✅ | `1` 找不到运行 |
| `wfos wiki search\|list\|promote` | 三层知识库 | ❌ | `1`/`2` 见下 |
| `wfos chat [--once]` | 交互式会话（同裸 `wfos`） | ❌ | `0`；`1` 仅 `--once` 失败时 |
| `wfos mcp --role --run-id` | 以 MCP stdio 服务端运行 | ❌ | `2` role 不在 `EXTERNAL_ROLES` |

> ⚠️ **`wfos wiki promote` 当前不可用**（真实缺陷，见 `docs/reference.md` §关键 Bug）：
> `id` 是第二个位置参数，`query` 会先把它吃掉，`args.id` 恒为 `None`，
> 随后 `d['id']` 抛 `TypeError` 打印 traceback（退出码 1）。

---

## 5. 从零完整运行

以下每条命令都真实可执行。以 mock provider 为例。

```bash
export WFOS_PROVIDER=mock
export WFOS_DATA=.data
export WFOS_PROJECT=$PWD
export PYTHONIOENCODING=utf-8          # Windows 建议
```

### 步骤 0 · 生产里先有一个规程

RSI 的母版本必须存在，否则候选的 `parent_version` 会是 0，晋升时会被"母版本不符"拒绝。
当前没有 CLI 命令可以直接种一条 skill（`repo.add_skill` 是库 API）：

```bash
python -c "
import sys; sys.path.insert(0,'.')
from wfos.config import load_config
from wfos.storage.repo import Repo
cfg = load_config(); repo = Repo(cfg.db_path)
repo.add_skill('regression', 'v1 规程', '触发: regression\n步骤: 先跑 check.py，再改', 'seed-run')
print('v1 就位')
"
```

### 步骤 1 · 跑一个 Task

```bash
wfos task run benchmark/smoke feature-happy-path --workspace .data/tasks
```
**发生什么**：建一个独立 workspace + 独立库，按状态机跑完，产出 `TaskOutcome`。
**产生什么**：`runs`/`steps`/`transitions`/`tool_calls`/`events` 五行数据，全在 `.data/tasks/.../.data/wfos.db`。
**下一步用什么**：它会打印 run id 和库目录。

### 步骤 2 · 看 Trace

```bash
wfos trace show <run_id>          # 需要 WFOS_DATA 指向该运行的库目录
wfos result <run_id>
```
**发生什么**：读 `events`（按 `event_id` 排序）与 `steps`。
**产生什么**：只读，不写。
**差别**：`trace` 含**已被删除的步骤**（`step.invalidated`），`result` 只给当下的状态。

### 步骤 3 · Evaluation

```bash
wfos evaluate <run_id> benchmark/smoke feature-happy-path --json
```
**发生什么**：硬门禁 → checks → metrics →（可选）judge。
**产生什么**：`evaluations` 表一行 + 一条 `EVALUATED` 事件。
**下一步用什么**：`verdict` 与 `axes`。

### 步骤 4 · Benchmark

```bash
wfos benchmark run regression --workspace .data/bench
```
**发生什么**：8 个任务各一个隔离环境，逐任务判定。
**产生什么**：一份 `BenchmarkRun`（内存），每任务的 run/判定在各自的库里。
**下一步用什么**：要用 `baseline create` 才能把它冻结成可比对的文件。

### 步骤 5 · 创建 Baseline

```bash
wfos baseline create --suite regression --out base.json --workspace .data/bws
```
**产生什么**：`base.json`，**不可覆盖**。
**下一步用什么**：`base.json`。

### 步骤 6 · Compare

```bash
wfos benchmark run regression --workspace .data/bench2 --json > fresh.json
wfos compare base.json fresh.json --json
```
**产生什么**：五态计数与逐项 findings。
**退出码**：有 regression → `1`。

### 步骤 7 · Experiment

```bash
wfos experiment run experiments/model-sweep.json --workspace .data/sweep --out sweep.json
```
**发生什么**：`2 模型 × 2 配置` 展开成 4 个 cell，每 cell 一个隔离环境。
**产生什么**：`sweep.json`（**拒绝覆盖**）。
**下一步用什么**：`sweep.json` 作为 RSI 的输入。

### 步骤 8 · RSI Candidate

```bash
wfos rsi propose sweep.json
wfos rsi list
```
> 注意：`model-sweep` 跑的是 smoke 层，不产生恢复记录，可能没有候选。
> 要产生候选，用 `experiments/recovery-regression.json`（它会在 regression 层制造
> `build_error` / `regression` / `test_failure` 三类失败并恢复）。

```bash
wfos experiment run experiments/recovery-regression.json --workspace .data/exp --out exp.json
wfos rsi propose exp.json
wfos rsi list --json
```
**产生什么**：`candidates` 表里一行 `PROPOSED`。
**下一步用什么**：`candidate_id`（形如 `skill:regression`）与它的 `version`。

### 步骤 9 · Candidate Evaluation

```bash
wfos rsi evaluate skill:regression --version 1 \
     --baseline base.json --workspace .data/cand
```
**发生什么**：`PROPOSED → EVALUATING`，跑同一套题（`Arm(skills=(proposed_change,))`），
和基线比较，然后落到 `PASSED` / `REJECTED` / `NOT_PROMOTABLE` / `FAILED`。
**产生什么**：candidate 行的 status + verdict（含 `evaluationIds`，是 `[{id, store}]` 对）。
**下一步用什么**：`PASSED` 才能晋升。

### 步骤 10 · Promotion

```bash
wfos rsi promote skill:regression --version 1 \
     --baseline base.json --actor rog --reason "通过门禁"
```
**发生什么**：门禁重读候选记录（不重跑），七项检查全过才写。
**产生什么**：`skills` 表新增一个 `live` 版本（`parent_id` 指向旧版）+ `promotions` 表一行。
**旧版本**：仍在，`superseded=1`，内容一字未变。

### 步骤 11 · 用新版本再跑一遍 Benchmark

```bash
wfos benchmark run regression --workspace .data/after-promote
wfos skills --json        # 能看到新版本
```
**验证**：加载到的是新版本的内容（`repo.skills_for([trigger], origin='interactive')`）。

### 步骤 12 · Rollback

```bash
wfos rsi rollback regression 1 --actor rog --reason "线上有问题"
wfos rsi history regression --json
wfos benchmark run regression --workspace .data/after-rollback
```
**发生什么**：只移动 `superseded` 指针。
**v2**：仍在，`superseded=1`，内容未变。
**产生什么**：`rollbacks` 表一行（含 from/to 的版本与 digest）。

### 步骤 13 · 确认什么都没丢

```bash
wfos rsi history regression --json    # versions / promotions / rollbacks 三份都在
```
或者直接跑完整验收：

```bash
python artifacts/accept_final.py
```

---

## 6. 测试体系

```
tests/  42 个文件，10,721 行
```

| 类型 | 文件 | 负责什么 |
|---|---|---|
| **单元** | `test_statemachine.py`、`test_router.py`、`test_tasks.py`、`test_metrics.py`、`test_config_contract.py`、`test_context_budget.py`、`test_relevance.py`、`test_redaction.py` | 纯函数与常量契约 |
| **契约** | `test_llm_contract.py`、`test_wire_contract.py`、`test_credentials.py`、`test_mcp_stdio.py` | provider wire 形状、工具名映射、失败分类、凭据缺失路径、MCP stdio |
| **集成** | `test_harness_flows.py`、`test_delegate.py`、`test_resume_contract.py`、`test_policy.py`、`test_scope.py`、`test_lease.py`、`test_parent_depth.py`、`test_deviation.py` | 状态机全流程、父子委托、恢复指纹、策略网关、租约、方案偏离 |
| **隔离** | `test_isolation.py`、`test_origin.py`、`test_task_runner.py`、`test_runner.py` | 每次运行一套栈、来源过滤 |
| **评测闭环** | `test_eval.py`、`test_bench.py`、`test_baseline.py`、`test_experiment.py`、`test_rsi.py`、`test_promotion.py` | 判定、基准、比较、实验、候选、晋升回滚 |
| **CLI 契约** | `test_cli_contract.py`、`test_cli_integration.py` | 以子进程方式驱动 CLI，断言退出码 / stdout 纯净 / Windows 路径 |
| **验收** | `artifacts/accept_p3.py` … `accept_p6.py`、`accept_final.py` | 阶段验收与最终闭环，**真跑**，不 mock 断言的中间层 |
| **变异** | `artifacts/mutate_p6.py`、`artifacts/mutate_p7.py` | 破坏代码，确认对应的测试会挂 |

### 关键不变量与保护它的测试

| 不变量 | 保护它的测试 | 变异验证 |
|---|---|---|
| `unknown ≠ pass` | `test_promotion.py::test_an_unknown_axis_is_not_a_pass`、`test_an_axis_no_task_spoke_to_is_unknown_not_pass` | mutate_p6「safety 轴不入闸」 |
| `failure ≠ success` | `accept_final.py` §10c（同一 run 换任务 → `pass`/`fail`） | — |
| baseline 不可变 | `test_cli_contract.py::test_an_existing_baseline_is_not_overwritten` | mutate_p7「baseline 可以被覆盖」 |
| candidate 不能自动晋升 | `test_promotion.py::test_a_candidate_cannot_run_its_own_promotion`（`inspect.getsource` 断言无 `promote(` 调用） | mutate_p6「候选状态机放行任意迁移」 |
| v1 不能被覆盖 | `test_promotion.py::test_v1_is_not_overwritten_by_the_promotion` | mutate_p6「旧版本不再被标记 superseded」 |
| rollback 不能删历史 | `test_promotion.py::test_rollback_moves_the_pointer_and_keeps_both_versions` | mutate_p6「rollback 没有真的切换指针」 |
| trace append-only | `test_events.py`、`accept_final.py` §10b（读两次比对 + `step.invalidated` 实证） | — |
| evaluator 独立 | `test_eval.py`（判定只读 repo，不读 agent 输出） | — |
| promotion 需要 `PASSED` | `test_a_candidate_that_was_never_passed_cannot_be_promoted` | mutate_p6「promotion gate 被绕过」 |
| provenance 必须完整 | `test_incomplete_provenance_is_not_promotable` | mutate_p6「provenance 不再被检查」 |
| 判定引用可解析 | `test_the_whole_chain_runs_for_real`（每个 `{id, store}` 都真能查到行） | mutate_p7「候选评测不再引用它据以判断的记录」 |
| 错误分类不退化成一种 | `test_llm_contract.py::test_the_local_failures_do_not_all_collapse_into_a_network_error` | mutate_p7 的三条分类变异 |
| JSON stdout 纯净 | `test_cli_contract.py::test_json_stdout_carries_no_human_text_around_it`（探**错误分支**，因为成功分支处处有手工设防） | mutate_p7「JSON 模式下人类输出也写进 stdout」 |
| 失败路径也是 JSON | `test_a_failed_json_call_is_still_json` | mutate_p7「失败路径不再补 JSON 文档」 |
| 候选不进 Arm 视图 | `test_skills.py::test_a_candidate_in_the_run_s_own_store_is_never_reported_as_in_play` | mutate_p7「arm 的 skills 视图改用全量」 |

**变异覆盖重点**（P6/P7 共 20 条）：promotion gate、safety gate、regression gate、候选状态机、
版本不可变、baseline 隔离、rollback 目标、digest 校验、provenance、错误分类×4、JSON 纯净×2、
CLI 退出码、baseline 可覆盖、评测引用、`list_skills` 过滤、Arm 视图。

---

## 7. 安全与正确性设计

| 机制 | 实现 | 强度 |
|---|---|---|
| **Run 隔离** | `RunEnvironment.create()` 每次建独立 workspace + 独立 `wfos.db` + 独立 MCP 栈；`Test each run gets its own repo` | **代码保证** |
| **Candidate 隔离** | 候选住 `candidates` 表；`skills_for(origin='interactive')` 只认 `status='live' AND origin='interactive'`；`list_skills()` 默认 production | **代码保证** |
| **Production Skill 隔离** | `PRODUCTION_SKILL_STATUS`/`PRODUCTION_SKILL_ORIGIN` 一处定义，`list_skills` 与 `skills_for` 共用 | **代码保证**（同一 DB，非存储隔离） |
| **Baseline 保护** | `write_baseline` 拒绝覆盖 + 候选路径无写 baseline 的调用 + 字节快照测试 | **代码保证** |
| **Append-only Trace** | `Repo` 对 `events`/`evaluations`/`promotions`/`rollbacks` 没有任何 UPDATE/DELETE 方法；唯一 DELETE 是 `delete_steps_for_state`（作用于 `steps`） | **代码保证** |
| **Digest 校验** | 回滚读**晋升记录**里的 digest，与当前行算出的 `content_digest` 比对 | **代码保证** |
| **Promotion Gate** | 七项检查，全部在 `promotion_gate()` 内；重读不重跑 | **代码保证** |
| **Rollback Gate** | 四道 fail-closed：版本存在且是正式 skill → 被晋升记录命名过 → digest 一致 → 不是当前版本 | **代码保证** |
| **Provenance** | 五项必填（实验/运行/失败类/证据/提议内容）+ 晋升记录带 `source_experiment`/`source_runs`/`evaluation_ids`/前后 digest | **代码保证** |
| **Unknown 处理** | 三态常量分离；比较与晋升门都把 unknown 当"证据不足"；token 用 `NULL` 不用 `0` | **代码保证** |
| **失败分类** | 两套词表分离；读 errno 与状态码，**从不读报文字本** | **代码保证** |
| **CLI 退出码** | 0/1/2/3/4，见 §2；`3` 由 `_run` 的 `SystemExit` 产生 | **代码保证** |
| **JSON 契约** | `_seal_json()` 在 `main()` 的 `finally` 里兜底 | **代码保证** |
| **工具策略** | `mcp/policy.py` 的角色/路径根/审批在 server 侧强制；`delegate` 后父进程仍用自己的 principal | **代码保证** |
| **脱敏** | `redact.py` 在写入路径上对模型与子进程产出脱敏；**不**对 harness 自己的 sha256 摘要脱敏 | **代码保证** |
| **密钥不入库** | 凭据只从环境变量读；`MissingCredentialError` 在任何请求之前抛出 | **代码保证** |

**只是约定、或存在限制的：**

| 项 | 现状 |
|---|---|
| 候选库与正式 skill 库 | **同一个 SQLite 文件**。隔离由代码路径保证，不是存储隔离 |
| `events` 的 append-only | 表结构上没有任何约束阻止 UPDATE/DELETE —— 保证来自 `Repo` 不提供方法 |
| 工具写盘的边界 | 由 `policy.py` 的路径根 + 审批保证；`workspace.delete` 需人工审批；但 `shell.run` 只在 `safe_commands` 白名单内 |
| baseline 文件 | 靠 `FileExistsError` 保护；文件系统层面仍可被别的进程改写 |
| trace 的 `at` 列 | 可被写坏（它只是字符串），排序不依赖它，所以无害但无保护 |
| `wiki promote` | **当前不可用**，见 reference.md |
| Experiment 的 matrix 键 | **未校验**，拼错会被静默忽略 |
| CI | `.github/workflows/ci.yml` 已写，但**这个目录不是 git 仓库，从未执行过** |
