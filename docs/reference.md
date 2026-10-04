# 参考

数据库结构 / 关键缺陷 / 设计亮点 / 当前限制 / 答辩讲解。

> 全部取自当前代码。标注「未实现」的是确实没有的东西。

---

## 1. 数据库结构

SQLite，13 张表。**所有表名、列名、主键、索引均取自 `wfos/storage/db.py` 的 `SCHEMA`。**

一个全新库建出来就是完整的；旧库通过 `_MIGRATIONS` 幂等地 `ALTER TABLE ADD COLUMN` 补齐
（`_migrate()` 在 `executescript(SCHEMA)` 之后跑，所以新库的 `steps.model` 也是这样补上的）。

### 1.1 `runs` —— 一次运行

| | |
|---|---|
| 主键 | `id` TEXT |
| 核心字段 | `kind`（feature/bugfix）、`title`、`description`、`state`、`status`、`parent_run_id`、`root_run_id`、`attempt`、`payload` JSON、`result` JSON、`error`、`identity_hash`、`key_files` JSON、`owner`、`lease_until`、`session_id`、`origin`、`task_id`、`task_version`、`created_at`、`updated_at` |
| 关联 | `parent_run_id` → `runs.id`（子流程，非外键约束） |
| 用途 | 一次运行的当前状态与身份 |
| append-only | **否** —— 状态迁移会改它 |
| 谁写 | `create_run`、`update_run`、`set_status`、`set_fingerprint`、`claim_run`、`renew_lease`、`release_run` |
| 谁读 | `get_run`、`list_runs`、`child_runs`、`root_run`、`ancestor_depth`、`eval`、`cli` |
| 索引 | 无 |

`origin` 是隔离的边界：`interactive` / `task` / `benchmark`。子 run **继承**父的 origin 和
session（`create_run` 里的注释：*a delegated sub-task of a benchmark run is a benchmark run*）。

### 1.2 `steps` —— 一次状态执行

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 核心字段 | `run_id`、`state`、`agent`、`status`（done/failed）、`input_json`、`output_json`、`error`、`failure_class`、`input_tokens`/`output_tokens`/`cached_tokens`/`reasoning_tokens`/`model_calls`/`latency_ms`/`cost_usd`、`model`、`event_id`、`created_at` |
| 用途 | 当前工作状态 |
| append-only | **否** —— `delete_steps_for_state` 会在重跑前删除它 |
| 谁写 | `add_step`；**删除**：`delete_steps_for_state` |
| 谁读 | `get_step`、`steps_for_run`、`step_done`、`metric_rows`、`skills.recoveries`、`eval.invariants` |
| 索引 | `idx_steps_run ON steps(run_id, state)` |

度量列 **NULL 表示未上报，绝不写 0**（`db.py` 注释：*so "not measured" cannot be averaged into
"free" by any later aggregate*）。`cost_usd` 除未上报外，也会因为 `[pricing]` 未配置而为 NULL。

### 1.3 `transitions` —— 状态迁移

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 核心字段 | `run_id`、`from_state`、`to_state`、`decision`（suggested/validated/blocked/rejected/auto/harness_override）、`reason`、`suggested_by`、`event_id`、`created_at` |
| append-only | 是 |
| 谁写 / 读 | `add_transition` / `transitions` |

### 1.4 `tool_calls` —— 工具审计

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 核心字段 | `run_id`、`agent`、`tool`、`args_json`、`ok`、`error`、`status`（ok/error/rejected）、`error_code`、`security_event`、`source`（agent/mcp）、`affected_paths` JSON、`diff_summary`、`event_id`、`created_at` |
| append-only | 是 |
| 谁写 / 读 | `log_tool_call` / `tool_calls`、`affected_paths_for_run`、`plan_violations_for_run` |

`affected_paths` 来自工作区快照 diff，**刻意不脱敏**（路径不是密钥，且方案偏离检查要读它）。

### 1.5 `approvals` —— 审批

| | |
|---|---|
| 主键 | `id` TEXT |
| 核心字段 | `run_id`、`action`、`scope`、`risk_level`、`reason`（实为 JSON `{"reason", "required_by"}`）、`status`（pending/approved/rejected）、`decided_by`、`decision_note`、`created_at`、`decided_at` |
| append-only | 否 —— `decide_approval` 只在 pending 时改一次 |
| 谁写 / 读 | `create_approval`、`decide_approval` / `get_approval`、`list_pending_approvals`、`pending_approvals_for_run`、`list_approvals`、`list_approved`、`has_approved_approval` |

### 1.6 `evidence` —— 证据

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 核心字段 | `run_id`、`kind`、`source`、`content`、`confidence`、`created_at` |
| append-only | 是 |
| 谁写 / 读 | `add_evidence` / `list_evidence` |

### 1.7 `skills` —— 规程（含正式与候选）

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 外键 | `parent_id INTEGER REFERENCES skills(id)` —— **全库唯一的外键** |
| 核心字段 | `trigger`、`title`、`procedure`、`evidence_run`、`version`、`superseded`、`files` JSON、`digest`（sha256 内容摘要）、`status`（candidate/live）、`origin`、`created_at` |
| 用途 | 提示词可加载的规程，按 trigger 分版本 |
| append-only | **内容**是（无 UPDATE procedure/title/digest，无 DELETE）；`superseded` 是**指针**会被改 |
| 谁写 | `add_skill`（先 `UPDATE ... superseded=1` 再 INSERT）、`set_current_skill`（只改 superseded） |
| 谁读 | `get_skill`、`skill_version`、`skill_versions`、`latest_skill`、`skills_for`、`list_skills` |
| 索引 | `idx_skills_trigger ON skills(trigger, superseded)` |

`digest` **只对 trigger/title/procedure/files/version 计算**（`skills.content_digest`），
不含 `superseded` —— 否则移动指针会被误判成内容篡改。

### 1.8 `promotions` —— 晋升记录

| | |
|---|---|
| 主键 | `promotion_id` TEXT |
| 核心字段 | `candidate_id`、`candidate_version`、`parent_skill_version`、`parent_digest`、`promoted_skill_version`、`promoted_digest`、`source_experiment`、`source_runs` JSON、`evaluation` JSON、`evaluation_ids` JSON、`compare_result` JSON、`gate` JSON、`actor`、`reason`、`created_at` |
| append-only | **是** —— `add_promotion` 的 docstring：*There is no update path and no delete path.* |
| 谁写 / 读 | `add_promotion` / `get_promotion`、`promotions_for`、`promotions_for_skill`、`all_promotions` |
| 索引 | `idx_promotions_candidate ON promotions(candidate_id, candidate_version)` |

**存 digest 而不是只存版本号**是这张表存在的理由（见 §2）。

### 1.9 `rollbacks` —— 回滚记录

| | |
|---|---|
| 主键 | `rollback_id` TEXT |
| 核心字段 | `trigger`、`from_version`、`to_version`、`from_digest`、`to_digest`、`actor`、`reason`、`created_at` |
| append-only | 是（只 INSERT） |
| 谁写 / 读 | `add_rollback` / `rollbacks_for`、`get_rollback` |
| 索引 | `idx_rollbacks_skill ON rollbacks(trigger, created_at)` |

### 1.10 `evaluations` —— 判定

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 核心字段 | `run_id`、`task_id`、`task_version`、`verdict`（pass/fail）、`axes` JSON、`reasons` JSON、`failures` JSON、`hard_gate` JSON、`event_id`、`created_at` |
| append-only | **是** —— 重判是**插第二行**，不替换 |
| 谁写 / 读 | `add_evaluation` / `evaluations_for_run`、`latest_evaluation` |
| 索引 | `idx_evaluations_run ON evaluations(run_id)` |

`task_version` 是主键语义的一部分：**同一份代码在 v1 通过、在 v2 不通过是两个结论**。
`hard_gate` 单独成列，因为"评委根本没看到的是什么"是看到分数时读者的第一个问题。

### 1.11 `candidates` —— RSI 候选

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| **唯一约束** | `UNIQUE(candidate_id, version)` —— **全库唯一的表级 UNIQUE** |
| 核心字段 | `candidate_id`、`version`、`type`、`parent_version`、`status`、`source_experiment`、`source_runs` JSON、`source_failures` JSON、`evidence` JSON、`proposed_change` JSON、`rationale`、`verdict` JSON、`created_at`、`updated_at` |
| append-only | 内容写一次；只有 `status`/`verdict` 会动 |
| 谁写 / 读 | `add_candidate`（IntegrityError → 中文 ValueError）、`set_candidate_status` / `get_candidate`、`candidates_for`、`list_candidates` |
| 索引 | `idx_candidates_status ON candidates(status)` |

### 1.12 `events` —— append-only trace

| | |
|---|---|
| 主键 | `event_id` INTEGER AUTOINCREMENT —— **唯一排序权威** |
| 核心字段 | `seq`（per-run 可读序号）、`at`（UTC 微秒，**从不用于排序**）、`run_id`、`session_id`、`actor`（harness/model/human/runner）、`type`、`payload` JSON、`schema_version` |
| append-only | **是**，且 docstring 写明 *nothing here updates or deletes, and there is no method that does* |
| 谁写 / 读 | `add_event` / `events_for_run`、`events_for_session` |
| 索引 | `idx_events_run ON events(run_id, seq)` |

`seq` 由 `MAX(seq)+1` 在同一条 INSERT 里分配 —— 只在持有执行租约时无竞争。

`events` 的既有 type（真实数据实测）：`run.created`、`task.loaded`、`transition`、`step.completed`、
`step.failed`、`step.invalidated`、`tool.called`、`approval.requested`、`evaluated`、
`rsi.candidate.proposed`、`rsi.candidate.status`。

### 1.13 `wiki` —— 三层知识库

| | |
|---|---|
| 主键 | `id` INTEGER AUTOINCREMENT |
| 核心字段 | `kind`（authoritative/case/candidate）、`title`、`content`、`source`、`evidence` JSON、`scope`、`verified`、`trust`、`version`、`run_id`、`status`（pending/verified/published/rejected）、`tags` JSON、`checksum`、`created_at` |
| append-only | 否 —— `update_wiki` 改 kind/status/verified/trust |
| 谁写 / 读 | `add_wiki`、`update_wiki` / `get_wiki`、`list_wiki`、`search_wiki` |

`search_wiki` 是**元数据过滤 + LIKE 子串检索**，按命中词数再按时间排序 —— **没有 FTS5，没有向量检索**。

### 1.14 汇总

| 表 | 主键 | append-only | 性质 |
|---|---|---|---|
| `runs` | `id` | 否 | mutable state |
| `steps` | `id` | 否（可删） | mutable state |
| `transitions` | `id` | 是 | 证据 |
| `tool_calls` | `id` | 是 | 证据 |
| `approvals` | `id` | 否 | state |
| `evidence` | `id` | 是 | 证据 |
| `skills` | `id` | 内容是，指针否 | 证据 + 指针 |
| `promotions` | `promotion_id` | **是** | immutable evidence |
| `rollbacks` | `rollback_id` | **是** | immutable evidence |
| `evaluations` | `id` | **是** | immutable evidence |
| `candidates` | `id` + `UNIQUE(candidate_id,version)` | 内容是，status 否 | 证据 + state |
| `events` | `event_id` | **是** | **immutable evidence** |
| `wiki` | `id` | 否 | state |

**`Repo` 里唯一的 DELETE 语句是 `delete_steps_for_state`。** 没有任何方法删除
`events` / `evaluations` / `promotions` / `rollbacks` / `candidates` 的行。

---

## 2. 为什么 `evaluation_id` 要用 (id, database) 成对

因为**一次隔离运行的每个任务写自己的数据库**。

`bench.run_benchmark` 对每个 task 调一次 `RunEnvironment.create(workspace, name=tier/prefix/task.id)` ——
每个 task 一个目录、一个 `wfos.db`。判定由 `eval.record(runner.repo, evaluation)` 写进
**那个 task 自己的库**。

于是实测出现的现象是：一套 8 个任务的评测，8 条记录的 `id` **全都是 1**（每个库的第一行）。

```jsonc
// promotions.evaluation_ids，实测
[ { "id": 1, "store": "C:\\...\\cand\\regression\\delete-needs-approval-and-writes-nothing\\.data\\wfos.db" },
  { "id": 1, "store": "C:\\...\\cand\\regression\\build-error-is-classified\\.data\\wfos.db" },
  ... 共 8 条，id 都是 1，store 各不相同 ]
```

三个后果，也是这条设计的理由：

1. **只记 id 会指向八个不相干的第一行**。这不是"不够精确"，是**指向了错误的东西** ——
   比没有引用更糟，因为它看起来像溯源。
2. **`store` 单独放在记录层面也不够**。曾经写成 `evaluationDb`（单个路径），
   而它只是第一个任务的库，对另外 7 条是错的。所以 store 必须跟着每一条 id 走。
3. **可解析性是这条引用的全部价值**。`tests/test_promotion.py::test_the_whole_chain_runs_for_real`
   会拿每个 `{id, store}` 真的去 `Repo(store)` 里 `SELECT 1 FROM evaluations WHERE id=?`，
   查不到就失败 —— 引用要么能打开，要么不算引用。

同一个道理也解释了 `TaskRecord` 为什么同时带 `evaluation_id` 和 `evaluation_store` 两个字段。

---

## 3. 关键缺陷与工程经验

按发现顺序。每条给出：问题 → 根因 → 修复 → 现在由什么防住。

### ① `delegate` 泄漏 principal

**问题**：父流程调用 `delegate` 推进子 run 后，父后续的工具调用被按**子 run 最后一个状态**的角色做检查、
按子 run 的 `allowed_write_paths` 做范围限定、审计记到**子 run 的 run_id** 上。
`affected_paths_for_run(parent)` 因此静默丢掉 delegate 之后的所有写入 —— 而方案偏离检查
和 `writesOutsidePlan` 不变式读的正是它。

**根因**：principal 存在 `gateway.set_principal()` 这个**进程级槽位**里，子 run 每个状态都设它，
没有任何地方恢复父的。

**修复**：principal 改为**随请求走**（`ToolGateway.call(tool, args, *, principal=None)`，
经 MCP `_meta` 的 `wfos/principal` 传递）；`set_principal` 变成栈式 push/pop。

**防住它的是**：一个**失败过的测试** + 一个 gateway 层的集成测试。第一条测试只断言审计行，
而审计行用的是 gateway 的**本地** principal —— 即使 server 忽略 `meta` 它也会通过。
这是"测试通过但没测到东西"的实例，所以补了第二条。

### ② Evaluator 把 `0` 和 `False` 读成门禁失败

**问题**：真实 live 运行里，一个完全合规的运行被判失败。`writesOutsidePlan=0` 和
`escalatedToHuman=False` 被当成"门禁不通过"。

**根因**：`_failed` 用了 `if not facts.get(name)` —— 真值判断。`0` 和 `False` 都是假值。

**修复**：改成拿实际值与任务声明的期望值比较。

**防住它的是**：成对的回归测试（`0` vs 非 `0`、`False` vs `True` 各一条）。
这个 bug 只有真实运行能暴露，"代码看起来对"骗不过它。

### ③ 五轴聚合"最后一个赢" —— 判定依赖任务顺序

**问题**：同一份候选，任务顺序变了，晋升判定就变了。

**根因**：聚合写成 `for task: axes[axis] = value`（后者覆盖前者）。回归套件里四个 resume
任务对 safety 不作声（`unknown`），另外四个测了 safety（`pass`）—— 谁排在后面谁说了算。

**修复**：改成最坏优先、与顺序无关：任一任务 `fail` 就成立；有任务**测量过**这个轴就由它定；
**没有任何任务提到过**才是 `unknown`。

**防住它的是**：`test_the_axis_verdict_does_not_depend_on_task_order` 显式跑两种顺序并断言结果相同。

### ④ Evaluate 引用退化成八个 `1`

见 §2。**根因**是每任务一个库 + 只记 id。**修复**是记 (id, store) 对。

**防住它的是**：`test_the_whole_chain_runs_for_real` 逐个引用真的去查行。

### ⑤ 任何 `OSError` → `network_error`

**问题**：磁盘满、只读文件系统、权限不足和对端连不上，被记成**同一个** `failure_class`。
而它们要的处置完全相反：磁盘要空间、权限要人、网络要重试。

**根因**：分类器问的是一个二元问题 —— "这是不是 `OSError`" —— 而上面四者都答是。

**修复**：新增 `permission_error` / `disk_error` / `generic_os_error` 三类，
`classify_os_error()` **读 errno，不读报文字本**（与 wire 词表同一条规则）。

**防住它的是**：一条专门的测试，断言九种输入产出**至少四类**不同结果 ——
只测"网络错误还是网络错误"的测试在旧代码上也会通过。

### ⑥ `--json` 的错误路径不是 JSON

**问题**：`wfos baseline create --suite nope --json` 在 stdout 打一句人类文本，
调用者的 `jq` 解析失败。

**根因**：20 处错误分支都是"打印一句话然后 return"，没有一处是为 JSON 写的。

**修复**：不在 20 处各教一遍 JSON，而是在 `main()` 的 `finally` 里统一兜底 ——
`_seal_json()` 发现"要了 JSON 却什么都没写"就把最后一句话包成 `{"error": ...}`。
人类输出本身在 JSON 模式下改走 stderr。

**防住它的是**：`test_json_stdout_carries_no_human_text_around_it` —— 它**故意探错误分支**，
因为成功分支处处有手工设防，变异打不中。

### ⑦ `_MIGRATIONS` 里的重复字典键

**问题**：`steps` 和 `tool_calls` 在 `_MIGRATIONS` 里各出现两次，后面的静默覆盖前面的，
迁移列丢失。

**根因**：Python 字典字面量允许重复键，后写的赢，**不报错**。

**修复**：合并进已有条目。`tool_calls` 那条是 **ruff F601** 抓出来的。

### ⑧ `ctx = dict(ctx)` 浅拷贝

**问题**：agent 把运行记录写进副本，`_execute_state` 读到空的 —— 25 个测试失败。

**根因**：为了给上下文加 seed 而做了浅拷贝，写回时写进了副本。

**修复**：在复制**之前**把 seed 写进调用方的 dict。

### ⑨ Cell / 候选目录名在 Windows 非法

**问题**：`experiments/model-sweep.json` 的 8 个 cell 里有 4 个以 `WinError 123` 失败。
同一个缺陷在候选目录 `rsi/skill:regression` 处**再次**出现。

**根因**：cell id 用 `|` 和 `:` 拼接，Windows 路径不允许。

**修复**：抽出共享的 `bench.slug()`，两处都用它。**第一次修的时候没有抽公共函数**，
所以它在另一个模块里复发了一次 —— 这是"修表象而不是修根因"的代价。

### ⑩ Compare 的键错配

**问题**：候选评测报出 8 个假 regression。

**根因**：候选的 arm 让记录带上 `cellId`（`task@arm`），而基线没有这个字段，
比较时键对不上，每个任务看起来"既被移除又被新增"。

**修复**：比较前把 `cellId` 清空，回退到 `taskId` 作键。

### ⑪ `skills.recoveries` 看不到恢复

**问题**：候选推导依赖的"恢复"记录读不到 —— 因为失败的 step 会在状态重跑时被**删除**。

**根因**：`recoveries` 读的是 `steps` 表，而 `steps` 是会被删的。

**修复**：改读 trace；但这破坏了手工构造的 run（没有 trace），所以最终取 **trace 与 steps 的并集**。
—— 这个 bug 正是 4.2「为什么 Trace 和 Workflow State 分开」的实证。

### ⑫ 我自己的验收脚本里有 4 处恒真断言

**问题**：`accept_final.py` 里写着 `... or True`、`if False else True`、
以及 `all(any(f) for t in ...)` 而 `f` 根本没定义 —— 它们会打印 `[OK]`，什么都没验证。

**根因**：写检查时的疏忽；其中最关键的一条由 **ruff F821** 抓出。

**为什么这是最值得记的一条**：一个恒真的检查比没有检查更糟 ——
它会让一次什么都没验证的运行看起来通过了。发现它的方式不是跑测试，是**逐条回读自己的断言**。

### ⑬ `list_skills()` 不按 status/origin 过滤

**问题**：默认返回"所有未 superseded 的行"。今天候选住在 `candidates` 表所以不泄漏，
但一旦 `skills` 表出现候选行，`bench._record` 构造 arm 视图时会把它当正式规程报出去。

**根因**：过滤条件少了两半，且这两个调用者问的是不同问题（一个"加载了什么"，一个"库里有什么"）。

**修复**：`PRODUCTION_SKILL_STATUS` / `PRODUCTION_SKILL_ORIGIN` 一处定义；
`list_skills()` 默认只返回 production；`derived` 那处显式要全量
（否则任务运行学到的候选规程本来就不该被过滤掉，候选推导会失去输入）。

**防住它的是**：一条反例测试（候选行在隔离库里 → arm 视图不含它、derived 含它）+ 两个变异。

### ⑭ `main()` 的 `_JSON_WRITTEN` 不重置（**未修，见 §4**）

---

## 4. 技术亮点

八条，每条都在代码里可验证。

### ① Independent Evaluator 与 Agent 解耦

`eval.evaluate(repo, run_id, task)` 的输入是 **repo 和任务**，没有 agent 产物。
它读 `invariants(repo, run_id, ...)` —— 步骤、工具调用、affected paths、快照观测。
Verifier agent 自己写 `verdict: pass`，对判定没有任何影响。
`judge` 是可选外部回调，且只能往 `reasons` 追加一句话。

### ② append-only Trace 与 mutable Workflow State 分离

`events` 只增不改（`Repo` 不提供任何 UPDATE/DELETE 方法）；`steps` 会被删（重跑语义）。
删除本身写成 `step.invalidated` 事件，带 `state` / `reason` / `erased`。
实测：一次 `accept_final.py` 能数到 21 处这样的删除，每一处都能证明"它跑过"。

排序权威只有 `event_id`（SQLite AUTOINCREMENT，插入时原子分配），
不是秒级的 `at`，也不是只在租约内无竞争的 `seq`。

### ③ Benchmark + immutable Baseline + five-state Compare

`write_baseline` 拒绝覆盖；`compare` 输出 `regression`/`improvement`/`unchanged`/`unknown`/`noise`，
全部由记录逐项比对产生，**从不问模型**。两侧环境 digest 不同时，所有度量结论降级为 `unknown`。
`latencyMs` 属于 `_NOISY`，永远只给 noise。

### ④ Matrix-based isolated Experiment

一个 cell = 一个 workspace + 一个 SQLite + 一整套 MCP 栈。
`2 模型 × 2 配置` 实测展开成 4 个 cell、8 个互不相同的 run id、8 个隔离库。

### ⑤ Evidence-driven RSI Candidate

候选文本来自"某次运行命中失败类**并且越过了它**"的记录；
没有恢复的失败只产生 finding（`Analysis.unactionable`），不产生候选。
`CandidateSpec.__post_init__` 校验 type，所以拼错的类型当场被拒，而不是变成一个
看起来像已知类型的 `NOT_PROMOTABLE`。

### ⑥ Explicit Promotion Gate

`promote()` 是全仓唯一写正式 skill 的函数；`test_a_candidate_cannot_run_its_own_promotion`
用 `inspect.getsource` 断言四个上游函数体内不出现 `promote(`。
门禁**重读**候选记录（不重跑），做七项检查，包括五轴聚合（最坏优先、与顺序无关）和母版本比对。

### ⑦ Immutable Skill Version + digest

`content_digest` 只算 trigger/title/procedure/files/version，**不含 `superseded`** ——
所以移动指针不会被误判成篡改。版本的文字永不改写：`add_skill` 只 INSERT，
`set_current_skill` 只 UPDATE `superseded`。

### ⑧ Fail-closed Rollback + provenance chain

四道拒绝：版本存在且是正式 skill → 被某条晋升记录命名过 → **digest 与该记录一致** → 不是当前版本。
第三道的 digest 来自**晋升记录**而不是被检查那一行 —— 否则改那一行就等于改它自己的不在场证明。

晋升记录里存的是 `parent_digest` 与 `promoted_digest` 一对，
`source_experiment` / `source_runs` / `evaluation_ids` / `gate` 俱全，
实测能从 v2 一路走回 v1 与实验文件。

---

## 5. 当前限制

只列真实存在的。已修的不在此列。

### 5.1 环境与交付

| 限制 | 说明 |
|---|---|
| **只在 Windows 上验证过** | CI 因此只跑 `windows-latest`；代码没有刻意的平台依赖，但 Linux/macOS 没有证据 |
| **CI 从未执行过** | `.github/workflows/ci.yml` 已写（Windows + 3.12，install/ruff/pytest），但这个目录**不是 git 仓库**，工作流一次都没跑过。真正的门禁是本地 `pytest -q` + `ruff check .` |
| **`benchmark/` 与 `experiments/` 不进包** | `pip install` 只装 `wfos` 包；这些是仓库相对路径，需从仓库根运行 |

### 5.2 模型与成本

| 限制 | 说明 |
|---|---|
| **只有 DeepSeek 实测过** | Anthropic / Gemini / OpenAI 官方端点**从未打过真实请求**；`challenge` 层因此本地跑不了 |
| **`[pricing]` 未配置** | `cost_usd` 恒为 `None`；预算的成本半边与基线的 `costUsd` 只经过单元测试 |
| **`context_overflow` 不可产生** | 它只会在 provider 的结构化错误码字段上产生，当前没有 provider 会给。这是**明确声明**，不是遗漏 |
| **真实模型下的记忆注入未验证** | 早失败不记忆，一直没拿到有效样本 |

### 5.3 能力边界

| 限制 | 说明 |
|---|---|
| **五类候选只有两类有自动生产者** | `skill`（恢复证据）与 `retry_policy`（重复失败）；`workflow_policy`/`tool_hint`/`context_policy` 只能手工构造，且一律 `NOT_PROMOTABLE` |
| **没有 LLM Judge** | `judge` 是可选的非 LLM 回调 |
| **没有并行 Experiment** | cell 串行执行 |
| **没有历史趋势序列** | 只有"当前 vs 一份冻结基线"；`metrics aggregate` 是一次性快照 |
| **没有向量检索** | `search_wiki` 是元数据过滤 + LIKE 子串。**刻意不做**（语料在个位数量级，实测 FTS5 对中文更差） |

### 5.4 隔离强度

| 限制 | 说明 |
|---|---|
| **候选库与正式 skill 库是同一个 SQLite 文件** | 隔离由**代码路径**保证，不是存储隔离 |
| **`events` 的 append-only 无结构约束** | 表层面没有任何东西阻止 UPDATE/DELETE；保证来自 `Repo` 不提供方法 |
| **baseline 文件可被别的进程改写** | 保护来自 `FileExistsError`，文件系统层面无防护 |

### 5.5 已知缺陷（未修）

| 缺陷 | 复现 | 根因 | 影响 |
|---|---|---|---|
| **`wfos wiki promote` 不可用** | `wfos wiki promote 5 --by rog` → `TypeError: 'NoneType' object is not subscriptable` + traceback，退出码 1 | `id` 是第二个位置参数，`query` 会先吃掉它（`args.id` 恒为 `None`）；`cmd_wiki` 又没检查 `promote_to_authoritative` 返回 `None` | 知识提升完全用不了；且是 traceback 而非干净的错误信息。`wiki` 不在主链路上（P1–P7 的闭环不经过它） |
| **`main()` 的 `_JSON_WRITTEN` 不重置** | 同进程内第二次调用 `cli.main(['baseline','create','--suite','nope','--json'])` → stdout **为空**（第一次正常输出 JSON） | `_JSON_WRITTEN` 只在模块导入时初始化为 `False`，`main()` 不重置 | CLI 一进程一次，所以**用户看不到**；但把它当库调用两次会漏掉那次兜底文档 |
| **Experiment matrix 的未知键被静默忽略** | matrix 写成 `{"modle": ["a","b"]}` → 仍然展开成 2 个 cell，但两臂完全相同 | `experiment._arm_from` 只认 `model`/`workflow`/`skill`/`config` 四个键，匹配不上就返回默认 `Arm` | 一次扫描会安静地重复测同一个配置，结果看起来像"两个配置一样好" |
| **`wfos skills` 显示全量（含候选）** | 若 `skills` 表出现 candidate 行，它会被列出来 | `cmd_skills` 显式传 `status=None, origin=None`，因为它是**查看器不是加载器** | **这是有意的**，列在这里是为了说明它不是遗漏；但列表不显示每行的 status，可能被误读 |

---

## 6. 答辩 / 面试讲解

### ① 为什么做

普通 agent 框架回答"模型能不能完成任务"。但当你开始改 prompt、换模型、调策略，
你需要的不是"这次成功了吗"，而是**"这次比上次好吗，凭什么这么说"**。
WFOS 做的是第二个问题。

### ② 解决什么问题

三件事：
1. **可信的判定** —— 判定不能来自模型自己的说法；
2. **可比的度量** —— 两次运行的比较不能靠感觉，也不能靠不同的环境；
3. **可控的改进** —— 从证据里推导出来的改动，必须经过复核和人的批准才能进生产。

### ③ 整体架构

```
接口层    cli.py（23 命令）+ mcp/server.py
评测层    eval → bench → bench_compare → experiment → rsi
运行层    runner（隔离）→ orchestrator（唯一迁移决策）→ agents / mcp 策略网关
持久化    repo.py（唯一 SQL 出口）→ db.py
叶子层    models / failures / metrics / events / config / redact / ...
模型层    llm/factory → 5 个适配器
```

关键是**两个方向都不绕行**：状态迁移只有 `orchestrator` 决定；持久化只有 `repo` 出口。

### ④ 最核心的三个设计

1. **Evaluator 独立于 Agent** —— `evaluate(repo, run_id, task)` 读的是运行记录，
   不是 agent 输出。硬门禁失败会短路掉 metrics 和 judge。
2. **append-only Trace 与 mutable Workflow State 分离** —— `steps` 会被删（重跑语义），
   所以删除本身写成事件。排序权威只有 `event_id`。
3. **晋升是显式动作** —— `promote()` 是全仓唯一写正式 skill 的函数，
   有测试用 `inspect.getsource` 断言没有别的函数调用它。

### ⑤ 最难的问题

**"unknown 怎么不被当成 pass"** 和 **"引用怎么才不是自说自话"**。

第一个贯穿全栈：`Evaluation.axes` 是三态；`compare` 遇到任一侧 unknown 就不判方向；
晋升门里 `functional/safety/operational` 任一不是 pass 就不通过；token 未上报存 `NULL` 不存 `0`；
`rollup` 把 unknownAxes 单独计数。

第二个体现在：回滚校验用的 digest 来自**晋升记录**而不是被检查那一行；
评测引用必须是 (id, 库) 对，因为每个隔离任务写自己的库、id 都是 1；
`promotions` 存 digest 而不只是版本号。

### ⑥ 一个真实 Bug

**五轴聚合"最后一个赢"**。回归套件里四个 resume 任务对 safety 不作声（`unknown`），
另外四个测了 safety（`pass`）—— 而聚合是 `for task: axes[axis] = value`，
**谁排在后面谁说了算**。同一份候选，任务顺序变了晋升判定就变。

修法是最坏优先、与顺序无关：任一 `fail` 就成立，有任务测量过就由它定，没有任务提到过才是 `unknown`。
防住它的是一条**显式跑两种顺序并断言结果相同**的测试。

这个 bug 的价值在于：它不是崩溃，是**静默地依赖了一个没人想过的前提**（任务顺序）。

### ⑦ 如何保证结果可信

| 手段 | 具体做法 |
|---|---|
| 判定不看模型自述 | Evaluator 读 repo |
| 环境可比 | 两侧 `environment.digest` 不同时，度量结论全部降级为 unknown |
| 参照系不可动 | `write_baseline` 拒绝覆盖 + 晋升前后字节快照断言 |
| 证据不可改 | `events`/`evaluations`/`promotions`/`rollbacks` 无 UPDATE/DELETE 方法 |
| 删除留痕 | `step.invalidated` 带 state/reason/erased |
| 引用可解析 | (id, 库) 对，测试逐个真查 |
| 未测不当通过 | 三态轴 + `NULL` 而非 `0` |

### ⑧ RSI 如何工作

`analyse` 从实验记录的 `derivedSkills` 里挑出"命中失败类**且越过了它**"的条目 →
每个生成一个 `CandidateSpec`（`parent_version` 取自当前生产版本）→
候选跑同一套题（`Arm(skills=(proposed_change,))`）+ 与同一份基线比较 →
`PASSED` / `REJECTED` / `NOT_PROMOTABLE` / `FAILED`。
**没有恢复的失败只产生 finding，不产生候选** —— 凭空写一条规程正是这套东西要防的事。

### ⑨ Promotion / Rollback 如何保证安全

**晋升**：门禁重读不重跑，七项检查（状态必须 `PASSED`、溯源五项齐全、
候选自己的 `hardGateFailures` 为空、三条关键轴、无回归、母版本仍是当前）。
不通过时**不移动**候选（没评估过 ≠ 被拒绝）。通过后产生新版本 + 一条 append-only 记录，
记录里存前后两个 digest。

**回滚**：四道 fail-closed。关键在第三道 —— digest 来自晋升记录而不是被检查那一行，
所以改那一行就等于改它自己的不在场证明，改了就会被拒。回滚只移动 `superseded` 指针，
新版本仍在、内容未变，多出一条回滚记录。

### ⑩ 最终验证结果

```
pytest -q                    729 passed, 1 skipped
ruff check .                 All checks passed
artifacts/accept_final.py    86 项检查全部通过（端到端真跑）
artifacts/mutate_p6.py       9/9 变异被抓住
artifacts/mutate_p7.py       11/11 变异被抓住
pip install -e .             干净 venv 里可用；从仓库外调用可用
```

验收覆盖：v1 → 实验 → 候选 → 基线 → 评测 → PASSED → 显式晋升 → v2 加载并跑通 →
回滚 → v1 加载并跑通 → 记录仍在、baseline 逐字节未变；以及
"被删掉的步骤在 steps 里消失、在 trace 里留证"、
"同一份运行换一个任务判定就从 pass 变 fail"、
"unknown 轴没有被算进 passed"。
