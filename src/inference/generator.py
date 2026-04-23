"""모델 추론 엔진 모듈.

condition 토큰 시퀀스에 Chat Template을 적용하고,
model.generate()를 호출하여 output 토큰 시퀀스를 생성한다.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.pre_stage.dataset import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def generate_floorplan(
    condition_tokens: list[int],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    gen_cfg: DictConfig,
) -> list[int]:
    """단일 조건에 대해 평면도 토큰 시퀀스를 생성한다.

    1. condition_tokens를 tokenizer.decode()로 문자열 변환
    2. Chat Template 적용 (system + user + add_generation_prompt=True)
    3. tokenizer.encode()로 input_ids 생성
    4. model.generate()로 토큰 시퀀스 생성
    5. 생성된 부분(prompt 이후)만 추출하여 반환

    Args:
        condition_tokens: 입력 조건 토큰 ID 리스트.
        model: 추론 모드의 AutoModelForCausalLM.
        tokenizer: 커스텀 토큰이 포함된 AutoTokenizer.
        gen_cfg: 생성 파라미터 DictConfig (max_new_tokens, do_sample 등).

    Returns:
        생성된 OUTPUT 토큰 ID 리스트 (프롬프트 제외, assistant 응답만).
    """
    # condition_tokens → 문자열 디코딩 (special token 포함)
    decoded_condition = tokenizer.decode(condition_tokens, skip_special_tokens=False)

    # Chat Template 적용 (훈련 시와 동일한 방식)
    # add_generation_prompt=True: assistant 턴 시작 토큰까지 포함
    prompt_str = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": decoded_condition},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompt_length = len(input_ids)

    # model.generate() 호출
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor)  # 패딩 없는 단일 시퀀스

    # Mod Record: Qwen2.5 Chat Template에서 assistant 턴은 <|im_end|>(151645)로 종료되지만
    # tokenizer.eos_token_id는 <|endoftext|>(151643)만 반환한다.
    # <END_OUTPUT>(커스텀 토큰)도 종료 신호로 등록하여 평면도 생성 완료 후 즉시 중단한다.
    # 세 토큰 모두 종료 신호로 인식해야 생성이 정상 중단된다.
    eos_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)
    # Mod Record: <END_OUTPUT> 이후에도 토큰이 생성되는 문제 발생.
    # <END_OUTPUT>을 EOS로 등록하면 generate()가 해당 토큰 생성 시점에 즉시 중단된다.
    end_output_id = tokenizer.convert_tokens_to_ids("<END_OUTPUT>")
    if end_output_id is not None and end_output_id not in eos_ids:
        eos_ids.append(end_output_id)

    # greedy(do_sample=false)일 때 sampling 파라미터(top_p, top_k, temperature)는
    # 무의미하므로 전달하지 않아 불필요한 warning을 방지
    gen_kwargs = dict(
        max_new_tokens=gen_cfg.max_new_tokens,
        num_beams=gen_cfg.num_beams,
        repetition_penalty=gen_cfg.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=eos_ids,
    )
    if gen_cfg.do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
        )

    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    # 프롬프트 부분 제거, 생성된 토큰만 추출
    generated_ids = outputs[0][prompt_length:].tolist()

    # EOS / <|im_end|> 토큰 이후를 잘라냄 (잔여 생성 방지)
    eos_set = set(eos_ids)
    for i, tid in enumerate(generated_ids):
        if tid in eos_set:
            generated_ids = generated_ids[:i]  # EOS 자체도 제거
            break

    logger.debug(
        "추론 완료 — prompt: %d tokens, generated: %d tokens",
        prompt_length,
        len(generated_ids),
    )

    return generated_ids
