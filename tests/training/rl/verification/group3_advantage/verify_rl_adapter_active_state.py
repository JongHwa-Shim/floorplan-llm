"""Group 3: RL 어댑터 활성 상태 / gradient 전파 회귀 검증 (mock, GPU 불필요).

배경 (버그 이력):
    멀티어댑터 구조(sft frozen + rl trainable)에서 활성 상태를 튜너 레벨
    `model.base_model.set_adapter(["sft","rl"])`로만 지정하면, PeftModel 래퍼의 인스턴스
    속성 `active_adapter`는 stale 단일 문자열("sft")로 남는다. TRL GRPOTrainer는 beta != 0일 때
    매 step reference logp를 `use_adapter(model, "ref")`로 계산하는데, 그 컨텍스트가 종료될 때
    `model.set_adapter(previous_adapter)`(= "sft" 단독)로 복원하면서 rl 어댑터를 비활성화 +
    requires_grad=False로 만든다. 그 결과 이후 compute_loss forward에서 rl로 gradient가 흐르지
    않아 lora_B가 0 초기값에 영구 고정됐다.

    수정: `RLTrainer._reassert_active_adapters()`가 매 generation 배치 직후 [sft, rl] 활성 상태를
    재확립한다. 본 verifier는 실제 peft + trl 라이브러리 동작으로 버그를 재현하고, **실제
    RLTrainer._reassert_active_adapters 메서드**(사본 아님)를 호출해 수정이 유효함을 가드한다.

검증 케이스:
    - precondition_stale_active_adapter : 로드 직후 lora_B==0 + active_adapter 불일치 실증
    - bug_use_adapter_deactivates_rl    : use_adapter("ref") 통과 후 rl 비활성 + frozen
    - bug_no_gradient_to_rl             : 버그 상태 backward → rl.lora_B grad 없음
    - fix_reasserts_active_and_trainable: 실제 _reassert_active_adapters 호출 → rl 복구, sft/ref frozen
    - fix_gradient_flows_to_lora_B      : 수정 후 backward → rl.lora_B grad non-zero
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_VERIF_ROOT = Path(__file__).resolve().parents[1]
if str(_VERIF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VERIF_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from trl.trainer.utils import use_adapter  # noqa: E402  (TRL이 실제로 쓰는 컨텍스트 매니저)

from _common import run_cases, summary_and_exit  # noqa: E402

from src.training.rl.trainer import RLTrainer  # noqa: E402  (실제 수정 대상 메서드)


# ---------------------------------------------------------------------------
# 최소 멀티어댑터 모델 빌더 (7B 불필요)
# ---------------------------------------------------------------------------

class _TinyBlock(nn.Module):
    """LoRA target(q_proj) 하나만 가진 최소 모듈."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def _build_loaded_model():
    """production 로드 직후 상태를 재현한다.

    model_loader(sft frozen + rl trainable, base_model.set_adapter) +
    TRL GRPOTrainer.__init__(ref 어댑터 추가) +
    RLTrainer.__init__(활성 [sft, rl] 재설정 + sft/ref freeze) 까지 반영한 PeftModel.
    """
    torch.manual_seed(0)
    base = _TinyBlock()

    # SFT: is_trainable=False (inference_mode=True) 재현
    sft_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], inference_mode=True)
    model = get_peft_model(base, sft_cfg, adapter_name="sft")

    # RL: trainable
    rl_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], inference_mode=False)
    model.add_adapter("rl", rl_cfg)
    model.base_model.set_adapter(["sft", "rl"])
    for n, p in model.named_parameters():
        if ".sft." in n:
            p.requires_grad_(False)

    # TRL: beta != 0 → ref 어댑터 추가 (버그 트리거 전제 조건)
    model.add_adapter("ref", model.peft_config["sft"])
    for n, p in model.named_parameters():
        if ".ref." in n:
            p.requires_grad_(False)

    # RLTrainer.__init__: 활성 [sft, rl] 재설정 + sft freeze
    model.base_model.set_adapter(["sft", "rl"])
    for n, p in model.named_parameters():
        if ".sft." in n:
            p.requires_grad_(False)
    return model


def _rl_lora_B(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_B.rl." in n]


def _rl_active(model) -> bool:
    active = model.base_model.active_adapters
    active = [active] if isinstance(active, str) else active
    return "rl" in active


def _call_real_reassert(model) -> None:
    """실제 RLTrainer._reassert_active_adapters를 stub self로 호출한다 (사본 아님).

    해당 메서드는 self.accelerator.unwrap_model(self.model)만 사용하므로 최소 stub으로 충분하다.
    """
    stub = types.SimpleNamespace(
        model=model,
        accelerator=types.SimpleNamespace(unwrap_model=lambda m: m),
    )
    RLTrainer._reassert_active_adapters(stub)


_X = torch.randn(2, 16)


# ---------------------------------------------------------------------------
# 케이스
# ---------------------------------------------------------------------------

def case_precondition_stale():
    """로드 직후: lora_B==0 이고, PeftModel.active_adapter("sft")와 base 활성([sft,rl])이 불일치."""
    model = _build_loaded_model()
    assert _rl_lora_B(model), "rl lora_B 파라미터를 찾지 못함"
    assert all(torch.count_nonzero(p) == 0 for _, p in _rl_lora_B(model)), \
        "lora_B가 zero-init이 아님 (전제 붕괴)"
    assert model.active_adapter == "sft", \
        f"PeftModel.active_adapter가 'sft'가 아님: {model.active_adapter}"
    assert _rl_active(model), "base 활성 어댑터에 rl이 없음 (로드 상태 오류)"


def case_bug_use_adapter_deactivates_rl():
    """TRL의 use_adapter('ref') 통과 후 rl이 비활성 + requires_grad=False가 되는지 (버그 재현)."""
    model = _build_loaded_model()
    with use_adapter(model, adapter_name="ref"):
        _ = model(_X)
    assert not _rl_active(model), "버그 미재현: use_adapter 이후에도 rl이 활성"
    assert all(not p.requires_grad for _, p in _rl_lora_B(model)), \
        "버그 미재현: use_adapter 이후에도 rl.requires_grad=True"


def case_bug_no_gradient_to_rl():
    """버그 상태에서 backward 시 rl.lora_B로 gradient가 흐르지 않는지."""
    model = _build_loaded_model()
    with use_adapter(model, adapter_name="ref"):
        _ = model(_X)
    model.zero_grad(set_to_none=True)
    model(_X).sum().backward()
    grads = [p.grad for _, p in _rl_lora_B(model)]
    assert all(g is None or torch.count_nonzero(g) == 0 for g in grads), \
        "버그 상태인데 rl.lora_B에 gradient가 흐름 (예상과 다름)"


def case_fix_reasserts_active_and_trainable():
    """실제 _reassert_active_adapters 호출 → rl 활성+trainable 복구, sft/ref는 frozen 유지."""
    model = _build_loaded_model()
    with use_adapter(model, adapter_name="ref"):
        _ = model(_X)
    _call_real_reassert(model)  # ← 실제 수정 대상 메서드
    assert _rl_active(model), "수정 후에도 rl이 비활성"
    assert all(p.requires_grad for _, p in _rl_lora_B(model)), \
        "수정 후에도 rl.requires_grad=False"
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".sft." in n), \
        "sft가 실수로 trainable이 됨"
    assert all(not p.requires_grad for n, p in model.named_parameters() if ".ref." in n), \
        "ref가 실수로 trainable이 됨"


def case_fix_gradient_flows_to_lora_B():
    """수정 후 backward → rl.lora_B에 non-zero gradient가 흐르는지 (핵심 회귀 가드)."""
    model = _build_loaded_model()
    with use_adapter(model, adapter_name="ref"):
        _ = model(_X)
    _call_real_reassert(model)
    model.zero_grad(set_to_none=True)
    model(_X).sum().backward()
    grads = [(n, p.grad) for n, p in _rl_lora_B(model)]
    assert grads, "rl.lora_B 파라미터가 없음"
    assert all(g is not None and torch.count_nonzero(g) > 0 for _, g in grads), \
        f"수정했는데도 rl.lora_B에 gradient가 흐르지 않음: " \
        f"{[(n, None if g is None else float(g.norm())) for n, g in grads]}"


class _Case:
    def __init__(self, name, intent, fn):
        self.name = name
        self.intent = intent
        self.fn = fn


def main():
    cases = [
        _Case("precondition_stale",     "로드 직후 lora_B==0 + active_adapter 불일치 실증",
              case_precondition_stale),
        _Case("bug_deactivates_rl",     "use_adapter('ref') 후 rl 비활성 + frozen (버그 재현)",
              case_bug_use_adapter_deactivates_rl),
        _Case("bug_no_grad",            "버그 상태 backward → rl.lora_B grad 없음",
              case_bug_no_gradient_to_rl),
        _Case("fix_reasserts",          "실제 _reassert_active_adapters → rl 복구, sft/ref frozen",
              case_fix_reasserts_active_and_trainable),
        _Case("fix_grad_flows",         "수정 후 backward → rl.lora_B grad non-zero",
              case_fix_gradient_flows_to_lora_B),
    ]
    results = run_cases(cases, lambda c: c.fn(), label="Group 3: rl_adapter_active_state")
    summary_and_exit(results, label="rl_adapter_active_state")


if __name__ == "__main__":
    main()
