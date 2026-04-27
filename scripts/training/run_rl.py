"""RL(강화학습) 훈련 실행 스크립트.

SFT final 모델에 GDPO + 토큰 수준 신용할당 강화학습을 적용한다.
Rule-based RLVR 보상함수로 7가지 보상을 사용한다.
롤아웃 생성은 기본적으로 vLLM colocate 모드를 사용한다.

사용법:
    # 기본 실행 (config/training/rl/pipeline.yaml 사용, vLLM colocate)
    uv run python scripts/training/run_rl.py

    # DDP 멀티 GPU (2개)
    uv run torchrun --nproc_per_node=2 scripts/training/run_rl.py

    # 하이퍼파라미터 오버라이드
    uv run python scripts/training/run_rl.py training.learning_rate=5e-6

    # 디버그 모드 (10 step, vLLM 비활성화 → HF generate 사용)
    uv run python scripts/training/run_rl.py training.max_steps=10 training.report_to=none rl.use_vllm=false

    # W&B 비활성화
    uv run python scripts/training/run_rl.py training.report_to=none

    # 신용할당 비활성화 (균등 broadcast 모드)
    uv run python scripts/training/run_rl.py advantage.use_token_credit_assignment=false

    # vLLM server 모드 전환 (3 GPU 이상 환경)
    uv run torchrun --nproc_per_node=2 scripts/training/run_rl.py rl.vllm_mode=server

    # Resume
    uv run python scripts/training/run_rl.py resume.enabled=true
"""

import atexit
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 프로젝트 루트를 sys.path에 추가 (패키지 임포트 보장)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.training.rl import (
    RLTrainer,
    RLPromptDataset,
    load_model_and_tokenizer,
)
from src.training.rl.model_loader import prepare_vllm_base_model
from src.training.augmentation.tokenizer import load_vocab

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """재현성을 위해 모든 랜덤 시드를 고정한다.

    Args:
        seed: 고정할 시드 값.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_checkpoint(cfg: DictConfig) -> str | None:
    """Resume 설정으로부터 사용할 체크포인트 경로를 결정한다.

    Args:
        cfg: Hydra DictConfig. cfg.resume 섹션을 참조한다.

    Returns:
        체크포인트 경로 문자열. Resume 불필요 또는 체크포인트 없으면 None.

    Raises:
        FileNotFoundError: checkpoint_path가 지정됐으나 해당 경로가 없을 경우.
    """
    resume_cfg = cfg.get("resume", {})

    if not resume_cfg.get("enabled", False):
        return None

    checkpoint_path = resume_cfg.get("checkpoint_path")
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"지정된 체크포인트를 찾을 수 없음: {path}")
        logger.info(f"지정된 체크포인트로 Resume: {path}")
        return str(path)

    if resume_cfg.get("auto_find_latest", True):
        output_dir = Path(cfg.training.output_dir)
        if not output_dir.exists():
            logger.warning(f"output_dir가 없어 Resume 불가: {output_dir}. 처음부터 시작합니다.")
            return None

        checkpoints = sorted(
            [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if checkpoints:
            latest = checkpoints[-1]
            logger.info(f"최신 체크포인트 자동 탐색 성공: {latest}")
            return str(latest)

        logger.warning("체크포인트를 찾을 수 없음. 처음부터 시작합니다.")

    return None


def _start_vllm_server(cfg: DictConfig) -> subprocess.Popen | None:
    """vLLM 서버를 subprocess로 시작한다 (server 모드 전용).

    colocate 모드에서는 호출하지 않는다.
    TRL GRPOTrainer 내부에서 colocate vLLM이 자동 초기화되기 때문.

    Args:
        cfg: Hydra DictConfig.

    Returns:
        subprocess.Popen 인스턴스. 서버 시작 실패 시 None.
    """
    rl_cfg = cfg.rl
    host = rl_cfg.get("vllm_server_host", "0.0.0.0")
    port = int(rl_cfg.get("vllm_server_port", 8000))
    timeout = float(rl_cfg.get("vllm_server_timeout", 300.0))

    cmd = [
        "trl", "vllm-serve",
        "--model", cfg.model.hub_id,
        "--tokenizer", str(Path(_PROJECT_ROOT) / cfg.model.tokenizer_dir),
        "--host", host,
        "--port", str(port),
    ]

    logger.info(f"vLLM 서버 시작 중: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 서버가 준비될 때까지 대기
    import urllib.request
    server_url = f"http://{host}:{port}/health"
    waited = 0.0
    interval = 5.0

    while waited < timeout:
        try:
            urllib.request.urlopen(server_url, timeout=2)
            logger.info(f"vLLM 서버 준비 완료 ({waited:.0f}초 소요): {server_url}")
            return proc
        except Exception:
            time.sleep(interval)
            waited += interval

    logger.error(f"vLLM 서버가 {timeout}초 내에 준비되지 않음. 서버 모드 중단.")
    proc.terminate()
    return None


@hydra.main(
    config_path=os.path.join(_PROJECT_ROOT, "config", "training", "rl"),
    config_name="pipeline",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """RL 훈련 메인 함수.

    Args:
        cfg: Hydra가 주입하는 DictConfig.
    """
    logger.info("=== RL 훈련 시작 ===")

    # DDP 재시작: nproc_per_node > 1이고 아직 torchrun 하위 프로세스가 아니면
    # os.execvp로 현재 프로세스를 torchrun으로 교체
    nproc = cfg.get("distributed", {}).get("nproc_per_node", 1)
    if nproc > 1 and "LOCAL_RANK" not in os.environ:
        cmd = ["torchrun", f"--nproc_per_node={nproc}"] + sys.argv
        logger.info(f"DDP 모드: torchrun으로 재시작 (nproc_per_node={nproc})")
        os.execvp("torchrun", cmd)

    # 증강 설정 로드 후 cfg.augmentation으로 병합
    aug_config_path = Path(_PROJECT_ROOT) / cfg.data.aug_pipeline_config
    aug_cfg = OmegaConf.load(aug_config_path)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "augmentation", aug_cfg, merge=True)
    OmegaConf.set_struct(cfg, True)
    logger.info(f"증강 설정 로드: {aug_config_path}")

    logger.info(f"설정:\n{OmegaConf.to_yaml(cfg)}")

    # 재현성 시드 고정
    seed: int = cfg.training.seed
    set_seed(seed)
    logger.info(f"시드 고정: {seed}")

    # Resume 체크포인트 경로 결정
    resume_checkpoint = _resolve_checkpoint(cfg)

    # 모델 + 토크나이저 로드 (NF4 base + SFT adapter frozen + RL adapter trainable)
    logger.info("모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # vLLM colocate 모드 전처리: Hub 원본 vocab(151936)과 훈련 모델 vocab(152232) 불일치로
    # sync_weights() 시 embed_tokens assertion 에러가 발생한다. 확장된 vocab을 포함한
    # base 모델을 로컬에 저장(최초 1회)하고, name_or_path를 해당 경로로 교체한다.
    # Mod Record: PeftModel.name_or_path은 nn.Module.__getattribute__가 class property보다
    # instance __dict__를 우선 확인하므로, model.config.name_or_path만 수정하면 반영되지 않는다.
    # model.base_model.model.__dict__['name_or_path']도 직접 업데이트해야 TRL에 반영된다.
    use_vllm = bool(cfg.rl.get("use_vllm", True))
    if use_vllm:
        logger.info("vLLM base 모델 준비 중 (vocab 확장 적용)...")
        vllm_base_dir = prepare_vllm_base_model(cfg, model, tokenizer)
        model.base_model.model.name_or_path = vllm_base_dir
        model.config.name_or_path = vllm_base_dir
        logger.info(f"vLLM name_or_path 설정 완료: {vllm_base_dir}")

    # Vocab 로드 (파서 및 데이터셋에서 필요)
    vocab_path = Path(_PROJECT_ROOT) / cfg.model.vocab_extension
    vocab = load_vocab(vocab_path, cfg.model.tokenizer_dir)
    logger.info(f"Vocab 로드 완료: {len(vocab.token_to_id)}개 커스텀 토큰")

    # RL 프롬프트 데이터셋 로드 (output 없이 prompt + metadata만)
    logger.info("데이터셋 로드 중...")
    train_dataset = RLPromptDataset(cfg, tokenizer, split="train", seed=seed)
    logger.info(f"훈련 데이터셋 크기: {len(train_dataset)}")

    # W&B 프로젝트 설정
    os.environ["WANDB_PROJECT"] = cfg.training.project_name

    # vLLM 설정 파라미터 구성
    from trl import GRPOConfig

    rl_cfg = cfg.rl
    use_vllm = bool(rl_cfg.get("use_vllm", True))
    vllm_mode = str(rl_cfg.get("vllm_mode", "colocate"))

    max_steps = int(cfg.training.get("max_steps", 0))
    rl_config_kwargs = dict(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        seed=cfg.training.seed,
        save_total_limit=cfg.training.get("save_total_limit", 3),
        # RL(GRPO) 설정
        num_generations=rl_cfg.num_generations,
        temperature=rl_cfg.temperature,
        top_p=rl_cfg.top_p,
        beta=rl_cfg.kl_coeff,                    # KL 페널티 계수
        epsilon=rl_cfg.clip_range,               # PPO 클리핑 엡실론
        num_iterations=rl_cfg.num_iterations,    # roll-out 재활용 횟수
        max_completion_length=cfg.data.max_completion_length,
        # <END_OUTPUT>을 추가 EOS로 등록: tokenizer의 기본 EOS(<|endoftext|>)는
        # 커스텀 <END_OUTPUT> 토큰을 stop token으로 인식하지 못하므로,
        # generation_kwargs로 EOS 리스트에 추가하여 생성 즉시 중단.
        # 미등록 시 SFT 모델이 <END_OUTPUT> 이후에도 max_completion_length까지 쓸모없는
        # 토큰을 생성하여 보상 파싱 실패 및 속도 저하가 발생한다.
        # Mod Record: vllm SamplingParams는 eos_token_id 대신 stop_token_ids를 사용한다.
        # use_vllm 여부에 따라 다른 키를 사용해야 한다.
        # Mod Record: vllm은 stop_token_ids에 포함된 토큰을 출력에서 제거(strip)한다.
        # 151643을 stop_token_ids에 넣으면 vllm이 이를 출력에서 제거하여 TRL의 clipped
        # 감지 로직이 항상 True가 되어 clipped_ratio=1 문제가 발생했다.
        # vllm은 config의 eos_token_id(151643)를 자연적 EOS로 처리하여 출력에 포함시키므로,
        # stop_token_ids에는 커스텀 토큰(152214)만 등록한다.
        generation_kwargs=(
            {"stop_token_ids": [vocab.token_to_id["<END_OUTPUT>"]]}         # vllm: 커스텀 토큰만
            if use_vllm else
            {"eos_token_id": [tokenizer.eos_token_id, vocab.token_to_id["<END_OUTPUT>"]]}  # HF: 두 토큰 모두
        ),
        # TRL의 기본 normalize_then_sum 모드 사용 (RLTrainer에서 GDPO로 오버라이드)
        multi_objective_aggregation="normalize_then_sum",
        # 그래디언트 체크포인팅
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        optim=cfg.training.get("optim", "paged_adamw_32bit"),
    )

    if max_steps > 0:
        rl_config_kwargs["max_steps"] = max_steps

    # vLLM 관련 설정 추가
    if use_vllm:
        rl_config_kwargs["use_vllm"] = True
        rl_config_kwargs["vllm_mode"] = vllm_mode
        rl_config_kwargs["vllm_gpu_memory_utilization"] = float(
            rl_cfg.get("vllm_gpu_memory_utilization", 0.45)
        )
        rl_config_kwargs["vllm_enable_sleep_mode"] = bool(
            rl_cfg.get("vllm_enable_sleep_mode", False)
        )
        rl_config_kwargs["vllm_tensor_parallel_size"] = int(
            rl_cfg.get("vllm_tensor_parallel_size", 1)
        )
        rl_config_kwargs["vllm_max_model_length"] = int(
            rl_cfg.get("vllm_max_model_len", 4096)
        )
        logger.info(
            f"vLLM 활성화: mode={vllm_mode}, "
            f"gpu_memory_utilization={rl_config_kwargs['vllm_gpu_memory_utilization']}, "
            f"sleep_mode={rl_config_kwargs['vllm_enable_sleep_mode']}"
        )

        # server 모드일 때만 vLLM 서버 subprocess 자동 시작
        # colocate 모드: GRPOTrainer 내부에서 자동 초기화 (별도 서버 불필요)
        if vllm_mode == "server":
            rl_config_kwargs["vllm_server_host"] = str(
                rl_cfg.get("vllm_server_host", "0.0.0.0")
            )
            rl_config_kwargs["vllm_server_port"] = int(
                rl_cfg.get("vllm_server_port", 8000)
            )
            rl_config_kwargs["vllm_server_timeout"] = float(
                rl_cfg.get("vllm_server_timeout", 300.0)
            )
            vllm_proc = _start_vllm_server(cfg)
            if vllm_proc is not None:
                atexit.register(lambda: vllm_proc.terminate() if vllm_proc else None)
            else:
                raise RuntimeError("vLLM 서버 시작 실패. server 모드를 확인하거나 colocate로 전환하세요.")

    rl_config = GRPOConfig(**rl_config_kwargs)

    # RLTrainer 생성
    logger.info("RLTrainer 생성 중...")
    trainer = RLTrainer(
        model=model,
        args=rl_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_cfg=cfg.rewards,
        advantage_cfg=cfg.advantage,
        vocab=vocab,
    )

    # 훈련 실행
    logger.info("RL 훈련 시작...")
    if resume_checkpoint:
        logger.info(f"Resume from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 훈련 완료 로그
    logger.info("훈련 완료")

    # 최종 저장: RL adapter만 저장 (SFT adapter와 base model 제외)
    # 추론 시 멀티 어댑터 스태킹으로 복원: base + partial_state + SFT adapter + RL adapter
    output_dir = Path(cfg.training.output_dir) / "final"
    logger.info(f"최종 체크포인트 저장 중: {output_dir}")
    trainer.save_model(str(output_dir))
    trainer._save_optimizer_and_scheduler(str(output_dir))
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))

    logger.info("=== RL 훈련 완료 ===")


if __name__ == "__main__":
    main()
