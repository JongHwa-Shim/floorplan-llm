"""최종 모델과 단계별 모델의 추론 어댑터를 명시적으로 선택한다."""


def select_adapters(inference_cfg) -> list:
    """선택한 모델 변형에 필요한 어댑터만 순서대로 반환한다.

    Args:
        inference_cfg: model_variant 및 adapters 설정.
    Returns:
        embed_align은 빈 목록, sft는 SFT, full은 SFT와 RL 목록.
        model_variant를 지정하지 않은 기존 설정은 custom으로 취급한다.
    Raises:
        ValueError: 변형 이름이 잘못되거나 필요한 어댑터가 없거나 중복될 때.
    """
    variant = inference_cfg.get("model_variant", "custom")
    entries = list(inference_cfg.get("adapters", None) or [])
    if variant == "custom":
        names = [entry.get("name", f"adapter_{i}") for i, entry in enumerate(entries)]
        if len(names) != len(set(names)):
            raise ValueError("어댑터 이름이 중복됩니다.")
        return entries
    if variant not in ("full", "sft", "embed_align"):
        raise ValueError("model_variant는 full, sft, embed_align 또는 custom이어야 합니다.")
    required = {"full": ("sft", "rl"), "sft": ("sft",), "embed_align": ()}[variant]
    selected = []
    for name in required:
        matches = [entry for entry in entries if entry.get("name") == name]
        if len(matches) != 1 or not str(matches[0].get("path", "")).strip():
            raise ValueError(f"{variant} 추론에는 '{name}' 어댑터 경로가 정확히 하나 필요합니다.")
        selected.append(matches[0])
    return selected
