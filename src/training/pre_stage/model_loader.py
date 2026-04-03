"""Pre-Stage 모델 로드 및 파라미터 동결 모듈.

4bit 양자화(QDoRA)로 Base LLM을 로드하고,
새 커스텀 토큰의 embedding 행과 lm_head 행만 훈련 가능하도록 설정한다.

Mod Record: 기존에는 embed_tokens.weight / lm_head.weight 전체에 requires_grad=True를 설정한 뒤
gradient hook으로 기존 토큰 행을 0으로 마스킹하는 방식을 사용했다.
이 방식의 문제:
  - backward 시 152232행 전체에 대한 gradient 계산 (낭비)
  - AdamW optimizer가 152232행 전체의 m, v state를 유지 → ~8.8GB VRAM 낭비
수정: PartialEmbedding / PartialLMHead 모듈로 교체하여 새 토큰 567행만 nn.Parameter로 분리.
optimizer state는 ~16MB로 감소, backward도 필요한 부분만 계산.
"""

import json
import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Partial-trainable 모듈
# ---------------------------------------------------------------------------

class PartialEmbedding(nn.Module):
    """임베딩 테이블에서 새 토큰 행만 학습 가능하게 분리한 모듈.

    frozen base embedding으로 전체 vocab을 조회한 뒤,
    새 토큰이 등장한 위치만 index_put (out-of-place)으로 교체한다.
    index_put은 src(new_embed)가 requires_grad=True이면 결과 텐서도
    grad_fn을 가지므로 gradient가 new_embed까지 올바르게 흐른다.

    Attributes:
        base_embed: frozen 원본 nn.Embedding (requires_grad=False).
        new_embed: 새 토큰 행만 담은 학습 가능 nn.Parameter. Shape: $(N_{new}, H)$.
        new_ids: 새 토큰 global ID 텐서 (buffer, GPU에 위치).
        global_to_local: global ID → local index 매핑 테이블 (buffer).
    """

    def __init__(
        self,
        original_embed: nn.Embedding,
        new_token_ids: List[int],
    ) -> None:
        """PartialEmbedding 초기화.

        Args:
            original_embed: 원본 nn.Embedding 모듈 (resize_token_embeddings 후).
                weight가 이미 GPU에 있을 수 있으므로 device를 명시적으로 맞춤.
            new_token_ids: 새로 추가된 커스텀 토큰 ID 리스트.
        """
        super().__init__()

        # 원본 임베딩을 frozen 서브모듈로 보존
        self.base_embed = original_embed
        self.base_embed.weight.requires_grad_(False)

        # device 명시: original_embed.weight가 GPU에 있을 경우 동일 device로 생성해야 함
        device = original_embed.weight.device

        new_ids_tensor = torch.tensor(new_token_ids, dtype=torch.long, device=device)
        self.register_buffer("new_ids", new_ids_tensor)

        # global_id → local index 매핑 (빠른 조회를 위한 lookup table)
        vocab_size = original_embed.num_embeddings
        global_to_local = torch.full((vocab_size,), -1, dtype=torch.long, device=device)
        global_to_local[new_ids_tensor] = torch.arange(
            len(new_token_ids), dtype=torch.long, device=device
        )
        self.register_buffer("global_to_local", global_to_local)

        # 새 토큰 행만 학습 가능 파라미터로 분리
        # optimizer state: num_new × H × 2(m,v) × 4bytes ≈ 16MB
        self.new_embed = nn.Parameter(
            original_embed.weight[new_ids_tensor].detach().clone()
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """frozen base embedding + 새 토큰 위치 override를 수행한다.

        Args:
            input_ids: 토큰 ID 텐서. Shape: $(B, L)$.

        Returns:
            임베딩 텐서. Shape: $(B, L, H)$.
        """
        # 1. frozen base로 전체 임베딩 조회 (gradient 없음)
        out = self.base_embed(input_ids)  # (B, L, H)

        # 2. 새 토큰이 있는 위치 탐색
        is_new = torch.isin(input_ids, self.new_ids)  # (B, L)
        if not is_new.any():
            return out  # 새 토큰 없으면 그대로 반환

        # 3. 새 토큰 위치에서 local ID 계산
        is_new_flat = is_new.reshape(-1)                          # (B*L,)
        new_global_ids = input_ids.reshape(-1)[is_new_flat]       # (num_new,)
        local_ids = self.global_to_local[new_global_ids]          # (num_new,)

        # 4. 새 토큰 임베딩 조회 (gradient 흐름 유지)
        new_vals = self.new_embed[local_ids]                      # (num_new, H)

        # 5. index_put (out-of-place)으로 해당 위치 교체
        # new_vals.requires_grad=True이므로 결과 텐서에 grad_fn이 생성됨
        # → gradient가 new_vals → new_embed로 정상 전파
        new_idx = is_new_flat.nonzero(as_tuple=True)[0]           # (num_new,)
        H = out.shape[-1]
        out_flat = out.reshape(-1, H)                             # (B*L, H)
        out_flat = out_flat.index_put((new_idx,), new_vals)       # (B*L, H), grad 있음
        return out_flat.reshape(*input_ids.shape, H)              # (B, L, H)

    def merge(self) -> None:
        """훈련된 new_embed를 base_embed.weight에 병합한다.

        저장 전에 호출하여 표준 HuggingFace 형식으로 복원할 수 있도록 한다.

        Args:
            없음.

        Returns:
            없음.
        """
        with torch.no_grad():
            self.base_embed.weight[self.new_ids] = self.new_embed.data


class PartialLMHead(nn.Module):
    """LM Head에서 새 토큰 행만 학습 가능하게 분리한 모듈.

    frozen base lm_head로 전체 vocab logits를 계산한 뒤,
    새 토큰 위치만 학습 가능한 new_lm_head로 scatter-override한다.
    scatter는 src(new_logits)가 requires_grad=True이면 결과도 grad_fn을 가진다.

    Attributes:
        base_lm_head: frozen 원본 nn.Linear (requires_grad=False).
        new_lm_head: 새 토큰 행만 담은 학습 가능 nn.Parameter. Shape: $(N_{new}, H)$.
        new_ids: 새 토큰 global ID 텐서 (buffer).
    """

    def __init__(
        self,
        original_lm_head: nn.Linear,
        new_token_ids: List[int],
    ) -> None:
        """PartialLMHead 초기화.

        Args:
            original_lm_head: 원본 nn.Linear 모듈 (bias 없음).
            new_token_ids: 새로 추가된 커스텀 토큰 ID 리스트.
        """
        super().__init__()

        self.base_lm_head = original_lm_head
        self.base_lm_head.weight.requires_grad_(False)

        device = original_lm_head.weight.device
        new_ids_tensor = torch.tensor(new_token_ids, dtype=torch.long, device=device)
        self.register_buffer("new_ids", new_ids_tensor)

        # 새 토큰 행만 학습 가능 파라미터로 분리
        self.new_lm_head = nn.Parameter(
            original_lm_head.weight[new_ids_tensor].detach().clone()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """frozen base logits + 새 토큰 logits scatter-override를 수행한다.

        Args:
            hidden_states: 마지막 hidden state. Shape: $(B, L, H)$.

        Returns:
            logits. Shape: $(B, L, V)$ — $V$: vocab size.
        """
        # 1. frozen base로 전체 vocab logits 계산
        # hidden_states가 new_embed로부터 grad를 가지므로 logits도 grad_fn을 가짐
        logits = self.base_lm_head(hidden_states)                 # (B, L, vocab)

        # 2. 새 토큰 logits를 학습 가능한 weight로 재계산 (gradient 흐름)
        new_logits = F.linear(hidden_states, self.new_lm_head)    # (B, L, num_new)

        # 3. scatter (out-of-place)으로 새 토큰 위치의 logits 교체
        # src(new_logits).requires_grad=True → 결과 텐서도 grad_fn 보유
        B, L, _ = logits.shape
        num_new = self.new_ids.shape[0]
        new_ids_exp = self.new_ids.view(1, 1, num_new).expand(B, L, num_new)
        logits = logits.scatter(-1, new_ids_exp, new_logits)      # (B, L, vocab), grad 있음

        return logits

    def merge(self) -> None:
        """훈련된 new_lm_head를 base_lm_head.weight에 병합한다.

        Args:
            없음.

        Returns:
            없음.
        """
        with torch.no_grad():
            self.base_lm_head.weight[self.new_ids] = self.new_lm_head.data


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, list[int]]:
    """4bit 양자화 모델과 확장 토크나이저를 로드하고 Pre-Stage용으로 설정한다.

    Args:
        cfg: Hydra DictConfig. cfg.model, cfg.quantization 섹션을 참조한다.

    Returns:
        tuple:
            - model: PartialEmbedding / PartialLMHead가 적용된 AutoModelForCausalLM
            - tokenizer: 커스텀 토큰이 추가된 AutoTokenizer
            - new_token_ids: 새로 추가된 커스텀 토큰 ID 리스트

    Raises:
        FileNotFoundError: vocab_extension.json 또는 tokenizer_dir가 없을 경우.
    """
    tokenizer_dir = Path(cfg.model.tokenizer_dir)
    vocab_extension_path = Path(cfg.model.vocab_extension)

    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"토크나이저 디렉토리를 찾을 수 없음: {tokenizer_dir}")
    if not vocab_extension_path.exists():
        raise FileNotFoundError(f"vocab_extension.json을 찾을 수 없음: {vocab_extension_path}")

    # 새 커스텀 토큰 ID 목록 추출
    new_token_ids = _load_new_token_ids(vocab_extension_path)
    logger.info(f"새 커스텀 토큰 수: {len(new_token_ids)}")

    # 확장된 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    logger.info(f"토크나이저 로드 완료. vocab size: {len(tokenizer)}")

    # 4bit 양자화 설정
    # embed_tokens, lm_head는 양자화 대상에서 자동 제외 → bfloat16 유지
    bnb_config = _build_bnb_config(cfg.quantization)

    # 모델 로드
    logger.info(f"모델 로드 중: {cfg.model.hub_id}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hub_id,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # kbit 훈련 준비:
    # - gradient checkpointing 활성화
    # - use_reentrant=False: PyTorch 2.5+ 기본값 변경에 맞춘 명시적 설정
    # - layernorm을 fp32로 업캐스팅 (안정적인 backward 보장)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 확장 토크나이저에 맞게 embedding table 크기 조정
    # embed_tokens, lm_head는 4bit 대상 제외(bf16)이므로 resize_token_embeddings 정상 동작
    # resize 후 새 토큰 행은 기존 토큰 평균으로 초기화됨 (HF 기본값)
    tokenizer_vocab_size = len(tokenizer)
    model.resize_token_embeddings(tokenizer_vocab_size)
    logger.info(
        f"embedding table 크기 조정: base_vocab → {tokenizer_vocab_size} "
        f"(+{len(new_token_ids)} 커스텀 토큰)"
    )

    # Pre-Stage 파라미터 설정: 새 토큰 행만 학습 가능하도록 모듈 교체
    _setup_partial_training(model, new_token_ids)

    return model, tokenizer, new_token_ids


def merge_and_restore(model: AutoModelForCausalLM) -> None:
    """PartialEmbedding/PartialLMHead를 원본 모듈로 복원하고 새 가중치를 병합한다.

    model.save_pretrained() 호출 전에 실행하여
    표준 HuggingFace 형식으로 저장될 수 있도록 한다.

    Args:
        model: PartialEmbedding / PartialLMHead가 적용된 AutoModelForCausalLM.

    Returns:
        없음. model이 in-place로 수정된다.
    """
    embed = model.model.embed_tokens
    lm_head = model.lm_head

    if isinstance(embed, PartialEmbedding):
        embed.merge()
        model.model.embed_tokens = embed.base_embed
        logger.info("embed_tokens: 새 토큰 가중치 병합 완료, 원본 모듈 복원")

    if isinstance(lm_head, PartialLMHead):
        lm_head.merge()
        model.lm_head = lm_head.base_lm_head
        logger.info("lm_head: 새 토큰 가중치 병합 완료, 원본 모듈 복원")


# ---------------------------------------------------------------------------
# 내부 헬퍼 함수
# ---------------------------------------------------------------------------

def _load_new_token_ids(vocab_extension_path: Path) -> list[int]:
    """vocab_extension.json에서 새로 추가된 토큰 ID 목록을 추출한다.

    Args:
        vocab_extension_path: vocab_extension.json 파일 경로.

    Returns:
        새 토큰 ID의 정렬된 리스트.
    """
    with open(vocab_extension_path, encoding="utf-8") as f:
        vocab_ext = json.load(f)

    base_vocab_size: int = vocab_ext["base_vocab_size"]
    token_to_id: dict[str, int] = vocab_ext["token_to_id"]

    new_ids = [tid for tid in token_to_id.values() if tid >= base_vocab_size]
    return sorted(new_ids)


def _build_bnb_config(quant_cfg: DictConfig) -> BitsAndBytesConfig:
    """BitsAndBytesConfig를 생성한다.

    Args:
        quant_cfg: quantization 설정 DictConfig.

    Returns:
        BitsAndBytesConfig 인스턴스.
    """
    compute_dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map[quant_cfg.bnb_4bit_compute_dtype]

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
    )


def _setup_partial_training(
    model: AutoModelForCausalLM,
    new_token_ids: list[int],
) -> None:
    """모든 파라미터를 동결하고 embed_tokens / lm_head를 Partial 모듈로 교체한다.

    Args:
        model: resize_token_embeddings 이후의 AutoModelForCausalLM.
        new_token_ids: 훈련할 새 토큰의 ID 리스트.
    """
    # 1단계: 전체 동결
    for param in model.parameters():
        param.requires_grad = False

    # 2단계: embed_tokens / lm_head를 Partial 모듈로 교체
    # PartialEmbedding/PartialLMHead 내부에서 각 모듈의 weight를 frozen으로 설정함
    model.model.embed_tokens = PartialEmbedding(
        original_embed=model.model.embed_tokens,
        new_token_ids=new_token_ids,
    )
    model.lm_head = PartialLMHead(
        original_lm_head=model.lm_head,
        new_token_ids=new_token_ids,
    )

    # 훈련 가능 파라미터 수 로그
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    num_new = len(new_token_ids)
    hidden = model.model.embed_tokens.new_embed.shape[1]

    logger.info(
        f"훈련 파라미터: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.4f}%)"
    )
    logger.info(f"  - embed_tokens.new_embed: ({num_new}, {hidden})")
    logger.info(f"  - lm_head.new_lm_head:    ({num_new}, {hidden})")
    optimizer_mb = trainable_params * 2 * 4 / 1024 / 1024  # m, v 각 fp32
    logger.info(f"  optimizer state: ~{optimizer_mb:.1f}MB (기존 gradient hook 방식 ~8800MB 대비)")
