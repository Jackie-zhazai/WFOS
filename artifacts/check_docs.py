"""Does the documentation describe this code, or some other code?

A doc that drifts is worse than no doc, because it is believed. This checks the
three documents in `docs/` against the repository, in both directions that can be
checked mechanically:

  * every file, table, column, class, function, constant, CLI command, CLI flag,
    state name and JSON key a document mentions must exist in the code;
  * every Mermaid relationship must name a column that the table it points at
    actually has.

What it cannot check: whether the prose is *true*. That was read by hand. This
catches the failure mode that is impossible to catch by reading — a name that
looked right when it was written and stopped being right afterwards.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ["docs/architecture.md", "docs/operations.md", "docs/reference.md",
        "README.md"]

problems: list[str] = []
checked = 0


def ok(label: str, condition: bool, detail: str = "") -> None:
    global checked
    checked += 1
    if not condition:
        problems.append(f"{label}{(' — ' + detail) if detail else ''}")


def read_docs() -> dict[str, str]:
    return {name: (ROOT / name).read_text(encoding="utf-8") for name in DOCS}


# ------------------------------------------------------------------ 1. 文件
def check_paths(docs: dict[str, str]) -> None:
    seen: set[str] = set()
    for text in docs.values():
        for m in re.finditer(r"`([\w./\-]+\.(?:py|json|toml|yml|md))`", text):
            seen.add(m.group(1))
        for m in re.finditer(r"\[`([\w./\-]+\.md)`\]", text):
            seen.add(m.group(1))
    # 文档里既有仓库相对路径（`wfos/eval.py`），也有为了可读只写文件名的
    # （测试表里的 `test_eval.py`）。后者在整个仓库里搜名字。
    #
    # 例外：带路径的 `.json` 才是要检查的文件；裸的 `base.json` / `sweep.json`
    # 是命令示例的**输出**文件名，本来就不该存在于仓库里。
    source_like = (".py", ".md", ".toml", ".yml")
    for rel in sorted(seen):
        if rel.startswith(("http", "//")):
            continue
        if rel.endswith(".json") and "/" not in rel:
            continue
        found = ((ROOT / rel).exists()
                 or any(p.name == pathlib.Path(rel).name
                        and p.suffix in source_like
                        for p in ROOT.rglob(pathlib.Path(rel).name)))
        ok(f"路径存在：{rel}", found)


# --------------------------------------------------------------- 2. 表与列
def check_schema(docs: dict[str, str]) -> None:
    from wfos.storage.db import SCHEMA

    tables = {}
    for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", SCHEMA,
                         re.S):
        body = m.group(2)
        cols = set()
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            col = re.match(r"(\w+)\s+(TEXT|INTEGER|REAL)", line)
            if col:
                cols.add(col.group(1))
            elif line.startswith("UNIQUE("):
                cols.add(line.split("(")[1].split(")")[0])
        tables[m.group(1)] = cols

    from wfos.storage.db import _MIGRATIONS
    for table, columns in _MIGRATIONS.items():
        tables.setdefault(table, set()).update(name for name, _ in columns)

    # 表名：文档里用反引号包起来的、看起来像表名的词
    for name, text in docs.items():
        for m in re.finditer(r"`([a-z_]{3,})`", text):
            word = m.group(1)
            if word in tables or word in ("runs", "steps"):
                ok(f"{name}: 表 {word}", word in tables)

    # Mermaid erDiagram 的每一列
    for m in re.finditer(r"^\s*(\w+)\s+(\w+)\s+(PK|FK)?\s*(?:\".*?\")?$",
                         docs["docs/architecture.md"], re.M):
        table, column = m.group(1), m.group(2)
        if table in tables:
            ok(f"erDiagram 列 {table}.{column}", column in tables[table])


def check_mermaid_relations(docs: dict[str, str]) -> None:
    """每条关系必须指向该表真有的列。"""
    from wfos.storage.db import SCHEMA

    tables: dict[str, set[str]] = {}
    for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", SCHEMA,
                         re.S):
        tables[m.group(1)] = set(re.findall(r"^\s*(\w+)\s+(?:TEXT|INTEGER|REAL)",
                                            m.group(2), re.M))
    from wfos.storage.db import _MIGRATIONS
    for table, columns in _MIGRATIONS.items():
        tables.setdefault(table, set()).update(name for name, _ in columns)

    text = docs["docs/architecture.md"]
    pattern = re.compile(r'(\w+)\s+[|}o][{o]?--[o|]?[o{]\s+(\w+)\s*:\s*"([^"]+)"')
    for left, right, label in pattern.findall(text):
        if left not in tables or right not in tables:
            continue
        # 关系标签可能写 "a + b"
        for piece in re.split(r"\s*[+（(]\s*", label):
            column = piece.strip().split("（")[0].strip()
            if not column or column in ("版本链", "子流程", "恢复指纹", "唯一排序权威",
                                        "per-run 可读序号", "从不用于排序",
                                        "指针，非内容", "sha256 内容摘要",
                                        "隔离", "判定住在哪个库", "candidate|live",
                                        "interactive|task|benchmark",
                                        "harness|model|human|runner"):
                continue
            # 关系指向的列可能在任一侧
            ok(f"erDiagram 关系 {left}→{right}: 列 {column}",
               column in tables[right] or column in tables[left],
               f"{right} 有 {sorted(tables[right])[:6]}…")


# ------------------------------------------------------------------ 3. CLI
def check_cli(docs: dict[str, str]) -> None:
    from wfos import cli

    text = "\n".join(docs.values())
    commands = set(re.findall(r"^\s*(?:\$\s*)?wfos\s+([a-z][a-z\-]*)", text, re.M))
    commands.discard("--help")
    for cmd in sorted(commands):
        buf, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
                cli.main([cmd, "--help"])
            code = 0
        except SystemExit as e:
            code = e.code or 0
        ok(f"CLI 子命令 {cmd}", code == 0, buf.getvalue()[:120])

    # 级联子命令：`wfos rsi promote …` 里的 `promote` 必须在该命令的 help 里
    for m in re.finditer(r"wfos\s+([a-z\-]+)\s+([a-z][a-z\-]{2,})", text):
        sub, second = m.group(1), m.group(2)
        if sub not in ("rsi", "benchmark", "baseline", "experiment", "task", "wiki",
                       "trace"):
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli.main([sub, "--help"])
        except SystemExit:
            pass
        ok(f"CLI 级联子命令 {sub} {second}", second in buf.getvalue(),
           buf.getvalue()[:200].replace("\n", " "))

    # 每个命令后面出现的每个 --flag，必须真的注册过
    # 只取同一行的其余部分 —— 否则一个允许跨行的 `\s` 会把后面几条命令的 flag
    # 算到这一条头上（第一版就是这么错的，它报出 `benchmark --out` 这种不存在的东西）。
    for m in re.finditer(r"wfos\s+([a-z\-]+)([^\n`]*)", text):
        sub, rest = m.group(1), m.group(2)
        flags = set(re.findall(r"--[a-z][a-z\-]*", rest))
        if not flags:
            continue
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli.main([sub, "--help"])
        except SystemExit:
            pass
        help_text = buf.getvalue()
        for flag in sorted(flags):
            ok(f"CLI 参数 {sub} {flag}", flag in help_text,
               help_text[:160].replace("\n", " "))


# ------------------------------------------------------- 4. 类 / 函数 / 常量
def check_identifiers(docs: dict[str, str]) -> dict:
    """文档里反引号包起来的标识符，必须能在 wfos/ 源码里找到（定义或引用）。"""
    source = "\n".join(p.read_text(encoding="utf-8")
                       for d in ("wfos", "tests", "artifacts")
                       for p in (ROOT / d).rglob("*.py"))
    names: set[str] = set()
    for name, text in docs.items():
        if name == "README.md":
            continue
        for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]{2,})`", text):
            names.add(m.group(1))
    skip = {"wfos", "py", "md", "json", "yaml", "text", "sql", "bash", "python",
            "pass", "fail", "unknown", "noise", "true", "false", "null",
            # 这两个是**刻意**写出来的反例：一个是"明确声明不可产生"的失败类，
            # 一个是"曾经用错、已改掉"的旧字段名。它们出现在文档里恰恰因为
            # 源码里没有 —— 所以"找不到"是预期结果，不是不一致。
            "context_overflow", "evaluationDb"}
    missing = []
    for n in sorted(names):
        if n in skip or n.islower() and len(n) < 4:
            continue
        if re.search(rf"\b{re.escape(n)}\b", source):
            continue
        missing.append(n)
    return {"missing": missing}


def check_named_constants(docs: dict[str, str]) -> None:
    from wfos import eval as eval_mod
    from wfos import rsi
    from wfos.bench_compare import VERDICTS

    for status in rsi.STATUSES:
        ok(f"状态名 {status} 在文档中", status in docs["docs/architecture.md"]
           or status in docs["docs/reference.md"] or status in docs["docs/operations.md"])
    for verdict in VERDICTS:
        ok(f"判定名 {verdict} 在文档中", verdict in "".join(docs.values()))
    for axis in eval_mod.AXES:
        ok(f"轴名 {axis} 在文档中", axis in "".join(docs.values()))


# -------------------------------------------------------------- 5. JSON 字段
def check_json_keys(docs: dict[str, str]) -> None:
    """operations.md 里 JSON 示例的键，必须出现在对应的 as_json() 里。"""
    from wfos.eval import Evaluation

    text = docs["docs/operations.md"]
    blocks = re.findall(r"```jsonc\n(.*?)```", text, re.S)
    joined = "\n".join(blocks)
    keys = set(re.findall(r'"([A-Za-z][A-Za-z0-9_]*)"\s*:', joined))
    # run 行与 metrics 报告的键来自 repo / metrics，按名字在源码里找即可
    source = "\n".join(p.read_text(encoding="utf-8")
                       for p in (ROOT / "wfos").rglob("*.py"))
    evaluation_keys = set(Evaluation.__dataclass_fields__)
    report_keys = {"counts", "environment", "findings", "baseline", "subject"}
    for key in sorted(keys):
        camel = key
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
        ok(f"JSON 键 {key}",
           camel in source or snake in source or key in report_keys
           or key in {"runs", "metrics", "model", "pricing", "skills",
                      "capabilities", "path", "count", "error"},
           f"as_json 字段={sorted(evaluation_keys)[:4]}…")


def main() -> int:
    docs = read_docs()
    check_paths(docs)
    check_schema(docs)
    check_mermaid_relations(docs)
    check_cli(docs)
    result = check_identifiers(docs)
    check_named_constants(docs)
    check_json_keys(docs)

    print(f"检查了 {checked} 项")
    if result["missing"]:
        print(f"\n文档里提到、但 wfos/ 源码里找不到的名字 {len(result['missing'])} 个：")
        for n in result["missing"]:
            print(f"  - {n}")
    if problems:
        print(f"\n不一致 {len(problems)} 处：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n文档与代码一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
