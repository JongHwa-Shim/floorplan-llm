"""DDP 메모리 비대칭 진단용 콜백.

GPU 0과 GPU 1의 메모리 사용량 비대칭(예: 15GB vs 8.6GB)의 원인을 추적하기 위한 디버그 도구.

PyTorch caching allocator의 두 지표를 분리해서 보여준다:
    - allocated: 현재 살아있는 텐서들이 점유한 실제 메모리.
    - reserved: PyTorch가 OS로부터 받아둔 풀 전체 크기 (nvitop/nvidia-smi가 보는 값).
    - peak_allocated: 마지막 reset 이후 도달한 최대 allocated (step 내 spike 추적).

해석:
    - alloc 비슷, reserved 차이: caching allocator 풀 크기/파편화 차이 (실제 객체는 같음).
    - alloc도 차이: 진짜로 객체가 다름 (rank 비대칭 — 잠재적 버그 가능성).
"""

import logging

import torch
import torch.distributed as dist
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MemoryDiagnosticCallback(TrainerCallback):
    """매 step 종료 시점의 GPU 메모리 통계를 rank별로 출력한다.

    Args:
        log_every_n_steps: 출력 주기 (1이면 매 step).
        reset_peak: True면 출력 후 peak 통계를 리셋하여 step별 spike만 추적.

    Notes:
        rank 간 출력 순서가 섞일 수 있으므로, 패턴 분석 시 grep으로
        rank별로 분리해서 비교하는 것을 권장한다.
        예: grep "rank=0" log.txt
    """

    def __init__(self, log_every_n_steps: int = 1, reset_peak: bool = True) -> None:
        self.log_every_n_steps = log_every_n_steps
        self.reset_peak = reset_peak

    def _print(self, tag: str, step: int) -> None:
        """현재 시점 메모리 통계를 한 줄로 출력한다.

        모든 가시 GPU에 대해 메모리를 측정하여 worker가 자기 device가 아닌 device에도
        텐서를 올렸는지(예: cuda:0 하드코드, default device 사용) 검증한다.

        Args:
            tag: 출력 prefix (예: "train_begin", "step_end").
            step: 현재 global_step.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        n_devices = torch.cuda.device_count()

        # 자기 device + 다른 device 모두 측정
        # rank 0 worker가 cuda:1에도 무언가 올렸다면 dev=1의 alloc/reserved가 0이 아님
        per_device = []
        for dev in range(n_devices):
            alloc = torch.cuda.memory_allocated(dev) / 1e9
            reserved = torch.cuda.memory_reserved(dev) / 1e9
            peak = torch.cuda.max_memory_allocated(dev) / 1e9
            marker = "*" if dev == rank else " "    # 자기 device 표시
            per_device.append(
                f"dev{dev}{marker} alloc={alloc:.2f} reserved={reserved:.2f} peak={peak:.2f}"
            )

        print(
            f"[MEM {tag} step={step} rank={rank}] " + " | ".join(per_device),
            flush=True,
        )

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """훈련 시작 시점의 baseline 메모리 출력."""
        self._print("train_begin", state.global_step)
        if self.reset_peak:
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """매 step 종료 후 메모리 출력 (옵션: peak 리셋)."""
        if state.global_step % self.log_every_n_steps != 0:
            return
        self._print("step_end", state.global_step)
        if self.reset_peak:
            # step별 spike 추적: 다음 step의 peak를 새로 측정하기 위해 모든 device 리셋
            for dev in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev)
