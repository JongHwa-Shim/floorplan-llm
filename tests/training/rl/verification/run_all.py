"""모든 verifier를 순서대로 실행하고 결과를 요약한다.

실행 순서: Group 1 (전처리) → Group 2 (보상별) → Group 3 (어드밴티지/손실)
각 verifier를 subprocess로 실행. 종료 코드로 PASS/FAIL 판정.

사용법:
    uv run python tests/training/rl/verification/run_all.py
    uv run python tests/training/rl/verification/run_all.py --skip-microstep
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _VERIF_ROOT.parents[3]


@dataclass
class VerifierSpec:
    group: str
    name: str
    path: Path
    requires_gpu: bool = False


GROUP1 = [
    VerifierSpec("Group 1", "metadata_after_transform",
                 _VERIF_ROOT / "group1_preprocessing/verify_metadata_after_transform.py"),
    VerifierSpec("Group 1", "metadata_after_drops",
                 _VERIF_ROOT / "group1_preprocessing/verify_metadata_after_drops.py"),
]
GROUP2 = [
    VerifierSpec("Group 2", "format",
                 _VERIF_ROOT / "group2_rewards/verify_format_reward.py"),
    VerifierSpec("Group 2", "count_total",
                 _VERIF_ROOT / "group2_rewards/verify_count_total_reward.py"),
    VerifierSpec("Group 2", "count_type",
                 _VERIF_ROOT / "group2_rewards/verify_count_type_reward.py"),
    VerifierSpec("Group 2", "orthogonality",
                 _VERIF_ROOT / "group2_rewards/verify_orthogonality_reward.py"),
    VerifierSpec("Group 2", "no_overlap",
                 _VERIF_ROOT / "group2_rewards/verify_no_overlap_reward.py"),
    VerifierSpec("Group 2", "room_in_outline",
                 _VERIF_ROOT / "group2_rewards/verify_room_in_outline_reward.py"),
    VerifierSpec("Group 2", "outline_in_room",
                 _VERIF_ROOT / "group2_rewards/verify_outline_in_room_reward.py"),
    VerifierSpec("Group 2", "coverage",
                 _VERIF_ROOT / "group2_rewards/verify_coverage_reward.py"),
    VerifierSpec("Group 2", "connectivity",
                 _VERIF_ROOT / "group2_rewards/verify_connectivity_reward.py"),
    VerifierSpec("Group 2", "spatial",
                 _VERIF_ROOT / "group2_rewards/verify_spatial_reward.py"),
    VerifierSpec("Group 2", "input_consistency",
                 _VERIF_ROOT / "group2_rewards/verify_input_consistency_reward.py"),
]
GROUP3 = [
    VerifierSpec("Group 3", "gdpo_group_normalize",
                 _VERIF_ROOT / "group3_advantage/verify_gdpo_group_normalize.py"),
    VerifierSpec("Group 3", "compute_token_advantages",
                 _VERIF_ROOT / "group3_advantage/verify_compute_token_advantages.py"),
    VerifierSpec("Group 3", "batch_normalize",
                 _VERIF_ROOT / "group3_advantage/verify_batch_normalize.py"),
    VerifierSpec("Group 3", "micro_step_loss_flow",
                 _VERIF_ROOT / "group3_advantage/verify_micro_step_loss_flow.py",
                 requires_gpu=True),
    VerifierSpec("Group 3", "rl_adapter_active_state",
                 _VERIF_ROOT / "group3_advantage/verify_rl_adapter_active_state.py"),
]


@dataclass
class RunResult:
    spec: VerifierSpec
    returncode: int
    skipped: bool = False
    error_message: str = ""


def run_one(spec: VerifierSpec) -> RunResult:
    """단일 verifier 실행."""
    print(f"\n{'='*70}")
    print(f"[{spec.group}] {spec.name}")
    print(f"  path: {spec.path.relative_to(_REPO_ROOT)}")
    print(f"{'='*70}")
    try:
        proc = subprocess.run(
            ["uv", "run", "python", str(spec.path)],
            cwd=str(_REPO_ROOT),
            check=False,
        )
        return RunResult(spec=spec, returncode=proc.returncode)
    except Exception as e:
        return RunResult(spec=spec, returncode=-1, error_message=str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-microstep", action="store_true",
                        help="Group 3 micro_step_loss_flow 건너뜀 (모델 로드 우회)")
    parser.add_argument("--only", choices=["group1", "group2", "group3"], default=None,
                        help="특정 그룹만 실행")
    args = parser.parse_args()

    specs: list[VerifierSpec] = []
    if args.only is None or args.only == "group1":
        specs.extend(GROUP1)
    if args.only is None or args.only == "group2":
        specs.extend(GROUP2)
    if args.only is None or args.only == "group3":
        for spec in GROUP3:
            if args.skip_microstep and spec.name == "micro_step_loss_flow":
                continue
            specs.append(spec)

    results: list[RunResult] = []
    for spec in specs:
        results.append(run_one(spec))

    # Summary
    print(f"\n\n{'='*70}")
    print("RUN_ALL 요약")
    print(f"{'='*70}")
    pass_count = sum(1 for r in results if r.returncode == 0)
    fail_count = sum(1 for r in results if r.returncode != 0)
    for r in results:
        status = "PASS" if r.returncode == 0 else f"FAIL (rc={r.returncode})"
        print(f"  [{r.spec.group}] {r.spec.name:30s}  {status}")
    print(f"\n총 {len(results)}개 중 PASS={pass_count}, FAIL={fail_count}")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
