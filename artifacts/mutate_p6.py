"""P6 mutation check: break each guarantee, confirm the tests that name it fail.

The point is not that the suite is green — it is that the suite is green *for the
right reason*. A test that passes against broken code is not evidence, so each
mutation below removes exactly one guarantee and names the tests that must notice.
A mutation whose expected tests all still pass is reported as a hole.

Every mutation is restored in a `finally`, and the file's bytes are compared
before and after, so a crashed run cannot leave the tree broken.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (name, file, old, new, tests that must fail)
MUTATIONS: list[tuple[str, str, str, str, tuple[str, ...]]] = [
    (
        "promotion gate 被绕过（不再复核候选状态）",
        "wfos/rsi.py",
        '    gate = promotion_gate(repo, record, baseline=baseline)\n'
        '    if gate["verdict"] != STATUS_PASSED:',
        '    gate = promotion_gate(repo, record, baseline=baseline)\n'
        '    if False:',
        ("test_a_candidate_that_was_never_passed_cannot_be_promoted",
         "test_a_safety_failure_rejects_the_promotion"),
    ),
    (
        "safety 轴不入闸",
        "wfos/rsi.py",
        '    for axis in ("functional", "safety", "operational"):',
        '    for axis in ("functional", "operational"):',
        ("test_a_safety_failure_rejects_the_promotion",),
    ),
    (
        "regression 不入闸",
        "wfos/rsi.py",
        '    if regressions:\n        critical = [s for s in regressions',
        '    if False and regressions:\n        critical = [s for s in regressions',
        ("test_a_regression_against_the_baseline_rejects_the_promotion",),
    ),
    (
        "候选状态机放行任意迁移",
        "wfos/rsi.py",
        '    ok, why = can_transition(record.status, status)\n    if not ok:',
        '    ok, why = can_transition(record.status, status)\n    if False:',
        ("test_a_promoted_candidate_cannot_be_changed_afterwards",),
    ),
    (
        "旧版本不再被标记 superseded（两个 current）",
        "wfos/storage/repo.py",
        '            "UPDATE skills SET superseded=1 WHERE trigger=? AND origin=? AND status=? "\n'
        '            "AND superseded=0", (trigger, origin, status))',
        '            "UPDATE skills SET superseded=1 WHERE trigger=? AND origin=? AND status=? "\n'
        '            "AND superseded=0 AND 1=0", (trigger, origin, status))',
        ("test_v1_is_not_overwritten_by_the_promotion",),
    ),
    (
        "晋升顺手改写了 baseline",
        "wfos/rsi.py",
        '    spec = record.spec\n    trigger = spec.source_failures[0]',
        '    spec = record.spec\n'
        '    if baseline and Path(baseline).exists():\n'
        '        Path(baseline).write_text("{}", encoding="utf-8")\n'
        '    trigger = spec.source_failures[0]',
        ("test_a_promotion_touches_no_baseline_no_task_and_no_evaluator_rule",),
    ),
    (
        "rollback 没有真的切换指针",
        "wfos/rsi.py",
        '    repo.set_current_skill(trigger, version)',
        '    repo.set_current_skill(trigger, int(repo.latest_skill(trigger)["version"]))',
        ("test_rollback_moves_the_pointer_and_keeps_both_versions",),
    ),
    (
        "rollback 不校验 digest",
        "wfos/rsi.py",
        '    if actual != recorded:',
        '    if False:',
        ("test_rollback_refuses_a_tampered_version",),
    ),
    (
        "provenance 不再被检查",
        "wfos/rsi.py",
        '    gaps = provenance_gaps(spec)\n    if gaps:',
        '    gaps = provenance_gaps(spec)\n    if False:',
        ("test_incomplete_provenance_is_not_promotable",),
    ),
]

TARGETS = ["tests/test_promotion.py"]


def run_tests() -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:randomly", *TARGETS],
        cwd=str(ROOT), capture_output=True, encoding="utf-8", errors="replace")
    failed = set()
    for line in proc.stdout.splitlines():
        if line.startswith("FAILED "):
            failed.add(line.split("::")[-1].split(" ")[0])
    if proc.returncode not in (0, 1):
        failed.add(f"<pytest 异常退出 {proc.returncode}>")
    return failed


def main() -> int:
    baseline_failures = run_tests()
    if baseline_failures:
        print(f"未变异的代码就有失败：{sorted(baseline_failures)}")
        return 1
    print("基线：全部通过\n")

    holes = []
    for name, rel, old, new, expected in MUTATIONS:
        path = ROOT / rel
        original = path.read_bytes()
        # Universal newlines: the anchors below are written with LF, and these
        # files are CRLF on Windows. Matching raw bytes found the single-line
        # anchors and missed every multi-line one — which is exactly the kind of
        # silent skip a mutation harness must not have.
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            print(f"  [跳过] {name}：锚点命中 {text.count(old)} 次")
            holes.append(name)
            continue
        try:
            path.write_text(text.replace(old, new), encoding="utf-8")
            failed = run_tests()
        finally:
            path.write_bytes(original)
            assert path.read_bytes() == original, f"{rel} 未被还原"

        missed = [t for t in expected if t not in failed]
        status = "OK  " if not missed else "HOLE"
        if missed:
            holes.append(name)
        print(f"  [{status}] {name}")
        print(f"         预期失败 {len(expected)} 个，实际失败 {len(failed)} 个"
              f"，其中未按预期失败的：{missed or '无'}")
        unexpected = sorted(failed - set(expected))
        if unexpected:
            print(f"         另外还失败了（不算错，但要看见）：{unexpected[:6]}")

    print()
    if holes:
        print(f"有 {len(holes)} 个变异没被抓住：{holes}")
        return 1
    print(f"{len(MUTATIONS)} 个变异全部被对应测试抓住")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
