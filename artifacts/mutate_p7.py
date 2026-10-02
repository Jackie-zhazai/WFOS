"""P7 mutation check: break each new guarantee, confirm the test that names it fails.

`mutate_p6.py` covers the promotion/rollback invariants; this one covers what P7
added — error classification, the CLI contract, and the append-only and
immutability properties that the final acceptance asserts. Same discipline: one
guarantee removed at a time, the tests that must notice are named, and every
mutation is restored in a `finally` with the bytes compared.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

MUTATIONS: list[tuple[str, str, str, str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "错误分类退化：OSError 一律当网络错误",
        "wfos/failures.py",
        '    if not isinstance(exc, OSError):\n        return "unexpected_error"',
        '    if not isinstance(exc, OSError):\n        return "unexpected_error"\n'
        '    return "network_error"',
        ("test_the_local_failures_do_not_all_collapse_into_a_network_error",),
        ("tests/test_llm_contract.py::test_the_local_failures_do_not_all_collapse_into_a_network_error",),
    ),
    (
        "权限错误不再被单独识别",
        "wfos/failures.py",
        '    if isinstance(exc, PermissionError):        # EACCES / EPERM, and Windows\' own\n'
        '        return FAILURE_PERMISSION\n',
        '',
        ("test_the_local_failures_do_not_all_collapse_into_a_network_error",),
        ("tests/test_llm_contract.py::test_the_local_failures_do_not_all_collapse_into_a_network_error",),
    ),
    (
        "磁盘错误被并回通用 OS 错误",
        "wfos/failures.py",
        '    if code in _DISK_ERRNOS:\n        return FAILURE_DISK\n',
        '',
        ("test_the_local_failures_do_not_all_collapse_into_a_network_error",),
        ("tests/test_llm_contract.py::test_the_local_failures_do_not_all_collapse_into_a_network_error",),
    ),
    (
        "网络判定放宽回“任何 OSError 都算”",
        "wfos/llm/base.py",
        '    return classify_os_error(exc) == "network_error"',
        '    return isinstance(exc, OSError)',
        ("test_a_network_error_is_classified_as_such_not_as_a_bug",),
        ("tests/test_llm_contract.py::test_a_network_error_is_classified_as_such_not_as_a_bug",),
    ),
    (
        "JSON 模式下人类输出也写进 stdout",
        "wfos/cli.py",
        '    print(*args, file=sys.stderr if _JSON_MODE else sys.stdout, **kw)',
        '    print(*args, file=sys.stdout, **kw)',
        ("test_json_stdout_carries_no_human_text_around_it",),
        ("tests/test_cli_contract.py::test_json_stdout_carries_no_human_text_around_it",),
    ),
    (
        "失败路径不再补 JSON 文档（stdout 留空）",
        "wfos/cli.py",
        '    if _JSON_MODE and not _JSON_WRITTEN:\n'
        '        _json({"error": _LAST_LINE or "命令没有产生输出"})',
        '    return',
        ("test_a_failed_json_call_is_still_json",),
        ("tests/test_cli_contract.py::test_a_failed_json_call_is_still_json",),
    ),
    (
        "未知命令不再返回用法错误码",
        "wfos/cli.py",
        "    args = parser.parse_args(argv)",
        "    try:\n        args = parser.parse_args(argv)\n"
        "    except SystemExit:\n        return 1",
        ("test_an_unknown_command_is_a_usage_error",),
        ("tests/test_cli_contract.py::test_an_unknown_command_is_a_usage_error",),
    ),
    (
        "baseline 可以被覆盖",
        "wfos/bench.py",
        "    if target.exists():\n        raise FileExistsError(",
        "    if False:\n        raise FileExistsError(",
        ("test_an_existing_baseline_is_not_overwritten",),
        ("tests/test_cli_contract.py::test_an_existing_baseline_is_not_overwritten",),
    ),
    (
        "list_skills 不再按 status/origin 过滤（候选会被当成正式规程）",
        "wfos/storage/repo.py",
        '        if status is not None:\n            clauses.append("status=?")\n'
        '            args.append(status)\n'
        '        if origin is not None:\n            clauses.append("origin=?")\n'
        '            args.append(origin)\n',
        "",
        ("test_list_skills_returns_production_and_not_the_other_tiers",),
        ("tests/test_skills.py::test_list_skills_returns_production_and_not_the_other_tiers",),
    ),
    (
        "arm 的 skills 视图改用全量（把候选报成正在生效）",
        "wfos/bench.py",
        'skills={str(s["trigger"]): int(s["version"]) for s in repo.list_skills()},',
        'skills={str(s["trigger"]): int(s["version"])\n'
        '                for s in repo.list_skills(status=None, origin=None)},',
        ("test_a_candidate_in_the_run_s_own_store_is_never_reported_as_in_play",),
        ("tests/test_skills.py::test_a_candidate_in_the_run_s_own_store_is_never_reported_as_in_play",),
    ),
    (
        "候选评测不再引用它据以判断的记录",
        "wfos/rsi.py",
        '        "evaluationIds": [{"id": t.evaluation_id, "store": t.evaluation_store}\n'
        '                          for t in run.tasks if t.evaluation_id],',
        '        "evaluationIds": [],',
        ("test_the_whole_chain_runs_for_real",),
        ("tests/test_promotion.py::test_the_whole_chain_runs_for_real",),
    ),
]


def run_tests(targets: tuple[str, ...]) -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:randomly", *targets],
        cwd=str(ROOT), capture_output=True, encoding="utf-8", errors="replace")
    failed = set()
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED "):
            failed.add(line.split("::")[-1].split(" ")[0])
    if proc.returncode not in (0, 1):
        failed.add(f"<pytest 异常退出 {proc.returncode}>")
    return failed


def main() -> int:
    holes = []
    print("基线：")
    for targets in {m[5] for m in MUTATIONS}:
        failures = run_tests(targets)
        print(f"  {'/'.join(targets)} → {'通过' if not failures else sorted(failures)}")
        if failures:
            print("  未变异的代码就有失败，先修它")
            return 1
    print()

    for name, rel, old, new, expected, targets in MUTATIONS:
        path = ROOT / rel
        original = path.read_bytes()
        text = path.read_text(encoding="utf-8")     # universal newlines; see mutate_p6
        if text.count(old) != 1:
            print(f"  [跳过] {name}：锚点命中 {text.count(old)} 次")
            holes.append(name)
            continue
        try:
            path.write_text(text.replace(old, new), encoding="utf-8")
            failed = run_tests(targets)
        finally:
            path.write_bytes(original)
            assert path.read_bytes() == original, f"{rel} 未被还原"

        missed = [t for t in expected if t not in failed]
        if missed:
            holes.append(name)
        print(f"  [{'OK  ' if not missed else 'HOLE'}] {name}")
        print(f"         预期失败 {len(expected)}，实际 {len(failed)}，未按预期：{missed or '无'}")

    print()
    if holes:
        print(f"有 {len(holes)} 个变异没被抓住：{holes}")
        return 1
    print(f"{len(MUTATIONS)} 个变异全部被对应测试抓住")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
