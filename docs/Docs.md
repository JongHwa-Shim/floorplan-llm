# 프로젝트 설계 문서 (Docs)

좌표 기반 평면도 생성 LLM — 상세 설계 및 구현 명세

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [LLM이 평면도를 표현하는 방식](#2-llm이-평면도를-표현하는-방식)
3. [커스텀 토큰 체계](#3-커스텀-토큰-체계)
4. [데이터 형식 정의](#4-데이터-형식-정의)
5. [전체 파이프라인](#5-전체-파이프라인)
6. [Step 1: 평면도 PNG → JSONL 추출](#6-step-1-평면도-png--jsonl-추출)
7. [Step 2: Vocabulary 빌드](#7-step-2-vocabulary-빌드)
8. [Step 3: JSONL → Arrow 변환](#8-step-3-jsonl--arrow-변환)
9. [Step 4: 데이터 증강 + 토크나이징](#9-step-4-데이터-증강--토크나이징)
10. [Pre-Stage: 새 토큰 Embedding 워밍업](#10-pre-stage-새-토큰-embedding-워밍업)
11. [Step 5: LLM 학습 (SFT + GRPO)](#11-step-5-llm-학습-sft--grpo)
12. [Step 6: GRPO 강화학습](#12-step-6-grpo-강화학습)
13. [Step 7: 추론 및 시각화](#13-step-7-추론-및-시각화)

---

## 1. 프로젝트 개요

### 목표

사용자가 자연어 또는 구조화된 조건을 입력하면, LLM이 각 방의 꼭지점 좌표를 포함한 완전한 평면도를 생성하는 모델을 구축한다.

### 핵심 설계 원칙

| 원칙 | 내용 |
|------|------|
| **평면도 = 토큰 시퀀스** | 이미지가 아닌 구조화된 커스텀 토큰 시퀀스로 평면도를 표현 |
| **GPT 스타일 조건부 생성** | MASK 방식이 아닌, 조건을 보고 전체 출력을 순차 생성 |
| **2-Layer 저장 구조** | 사람이 읽을 수 있는 JSONL 원본 + 학습용 Arrow 바이너리 |
| **증강 원칙 (변형 → 출력 → 삭제)** | 변형 증강을 먼저 적용 후 출력 확정, 이후 삭제 증강을 입력에만 적용 |
| **독립적 증강 파이프라인** | DataLoader와 완전 분리된 독립 증강 클래스 설계 |

### 전체 파이프라인 요약

```
① 데이터셋 구축
   평면도 PNG → 정보 추출 → JSONL → Vocab 빌드 → Arrow 변환 → 증강 파이프라인

② LLM 훈련
   Pretrained LLM + 커스텀 토큰 → Pre-Stage → 3-Stage Fine-tune (SFT → DPO → GRPO) → 평면도 생성 모델

③ 생성 결과 시각화
   조건 입력 (4가지 모드) → 증강 (선택적) → 모델 추론 (N출력) → 토큰 파싱 → 평면도 시각화
```

---

## 2. LLM이 평면도를 표현하는 방식

### 평면도 = 토큰 시퀀스

평면도의 각 구성 요소를 커스텀 토큰으로 표현한다. "거실이 (100,200)~(200,300) 영역에 있다"는 정보는 다음과 같이 토큰화된다.

```
<ROOM> <RID:1> <TYPE:livingroom> <X:100> <Y:200> <X:200> <Y:200> <X:200> <Y:300> <X:100> <Y:300> <END_ROOM>
```

좌표는 `<ROOM>` ~ `<END_ROOM>` 사이에 별도 래퍼(`<COORDS>`) 없이 직접 나열된다. 좌표값 `100`을 텍스트로 쓰면 LLM 토크나이저에 의해 여러 서브워드로 쪼개지지만, `<X:100>`이라는 단일 전용 토큰을 사용하면 시퀀스 길이가 대폭 감소한다.

### 조건부 생성 방식

모델은 조건(입력)을 받아 전체 평면도(출력)를 자동회귀적으로 생성한다.

**입력으로 줄 수 있는 조건:**
- 방 종류와 개수 (`<ROOM_SUMMARY>`)
- 일부 방의 좌표 (`<ROOM>` ~ `<END_ROOM>` 내 좌표 토큰)
- 방 간 연결관계와 문 위치 (`<EDGE>` ~ `<END_EDGE>`)
- 방 간 위치관계 (`<SP>` ~ `<END_SP>`)
- 현관문 위치 (`<FRONT_DOOR>` ~ `<END_DOOR>`)

**모델이 출력하는 것:**
- 현관문 정보 (`<FRONT_DOOR>`)
- 모든 방의 종류 + 꼭지점 좌표 리스트 (ROOMS)
- 전체 인테리어 문 정보 (`<DOOR>` ~ `<END_DOOR>` 목록)

조건이 많을수록 거의 재구성에 가깝고, 조건이 적을수록 모델이 자율적으로 판단해야 하는 범위가 넓어진다. 데이터 증강의 "삭제" 전략이 이 다양한 난이도를 학습 데이터로 제공한다.

---

## 3. 커스텀 토큰 체계

기존 LLM의 Vocabulary에 평면도 도메인 전용 토큰 ~1000개를 추가한다.

### 토큰 카테고리

| 카테고리 | 토큰 예시 | 개수 | 초기화 |
|---------|---------|------|--------|
| 좌표 X | `<X:0>` ~ `<X:255>` | 256 | Sinusoidal (연속성 반영) |
| 좌표 Y | `<Y:0>` ~ `<Y:255>` | 256 | Sinusoidal (연속성 반영) |
| 구조 | `<INPUT>` `<END_INPUT>` `<OUTPUT>` `<END_OUTPUT>` `<ROOM>` `<END_ROOM>` `<EDGE>` `<END_EDGE>` `<SP>` `<END_SP>` `<ROOM_SUMMARY>` `<END_ROOM_SUMMARY>` `<TOTAL>` `<COUNT>` `<SEP_DOOR>` `<DOOR>` `<END_DOOR>` `<NO_DOOR>` `<FRONT_DOOR>` 등 | ~19 | 랜덤 |
| 방 종류 | `<TYPE:livingroom>` `<TYPE:bedroom>` `<TYPE:kitchen>` `<TYPE:bathroom>` `<TYPE:entrance>` 등 | ~8 | 랜덤 |
| 방 ID | `<RID:0>` ~ `<RID:15>` | 16 | 랜덤 |
| 위치관계 | `<REL:right>` `<REL:left>` `<REL:above>` `<REL:below>` `<REL:right-above>` `<REL:right-below>` `<REL:left-above>` `<REL:left-below>` | 8 | 랜덤 |

> **숫자 토큰 (`<TOTAL>` / `<COUNT>` 뒤):** `<TOTAL>` 및 `<COUNT>` 레이블 토큰 뒤에 오는 숫자(개수)는 별도의 커스텀 토큰(`<TOTAL:N>` 형태)을 만들지 않고, **LLM 기본 어휘에 이미 존재하는 숫자 토큰**을 그대로 사용한다. 예: `<TOTAL> 7` (7은 LLM 기본 토큰). 이 덕분에 vocab 크기를 줄이고 숫자 표현에 대한 LLM의 기존 이해를 그대로 활용할 수 있다.

> **좌표 토큰 초기화:** 숫자 간의 연속적 관계(100과 101은 가깝다)를 반영하기 위해 Sinusoidal 위치 인코딩 등을 활용한 초기화를 사용한다. 나머지 토큰은 랜덤 초기화 후 학습 중에 의미를 잡아간다.

### 구현 유의사항: LLM별 Tokenizer 호환

1. **ID를 직접 매기지 말 것.** `base_vocab_size=32000` 같은 하드코딩은 LLM마다 vocab 크기가 달라 충돌이 발생한다 (LLaMA2=32K, LLaMA3=128K, Qwen2.5=152K 등). 반드시 `tokenizer.add_tokens()`로 위임한다.
2. **토큰 문자열 목록만 정의하고 ID 매핑은 tokenizer에서 추출한다.** `tokenizer.convert_tokens_to_ids(token)`으로 매핑 구성.
3. **확장된 tokenizer를 `tokenizer.save_pretrained()`로 저장한다.**
4. **`model.resize_token_embeddings(len(tokenizer))` 필수 호출.** Embedding/lm_head 행렬 크기를 맞춰야 한다.
5. **토큰 목록의 순서를 고정한다.** 순서가 바뀌면 동일 토큰에 다른 ID가 부여될 수 있다.

---

## 4. 데이터 형식 정의

### 4.1 Layer 1: JSONL 원본 데이터

하나의 평면도 = 하나의 JSON 라인.

| 필드 | 타입 | 설명 |
|------|------|------|
| `plan_id` | string | 평면도 고유 식별자 |
| `rooms` | array | 모든 방 정보 목록 |
| `rooms[].rid` | int | 방 고유 ID (평면도 내에서) |
| `rooms[].type` | string | 방 종류 (`livingroom`, `bedroom`, `outline` 등) |
| `rooms[].coords` | int[] | 꼭지점 좌표 flat array `[x1,y1,x2,y2,...]` |
| `edges` | array | 인접한 방 쌍 + 문 정보 |
| `edges[].pair` | [int,int] | 연결된 방 ID 쌍 |
| `edges[].doors` | array | 문 정보 목록 (없으면 빈 리스트 `[]`) |
| `edges[].doors[].x` | int | 문 중심 x 좌표 |
| `edges[].doors[].y` | int | 문 중심 y 좌표 |
| `edges[].doors[].w` | int | 문 폭 |
| `edges[].doors[].h` | int | 문 높이 |
| `front_door` | object\|null | 현관문 정보 (`{x, y, w, h}` 또는 `null`) |
| `spatial` | array | 방 간 위치관계 `[[rid_a, rid_b, "direction"], ...]` |

**JSONL 예시:**

```json
{
  "plan_id": "fp_00123",
  "rooms": [
    {"rid": 0, "type": "outline",     "coords": [80,30, 80,220, 210,220, 210,30]},
    {"rid": 1, "type": "livingroom",  "coords": [100,200, 100,300, 200,300, 200,200]},
    {"rid": 2, "type": "bedroom",     "coords": [200,200, 200,300, 300,300, 300,200]},
    {"rid": 3, "type": "kitchen",     "coords": [100,300, 100,400, 200,400, 200,300]}
  ],
  "edges": [
    {"pair": [1,2], "doors": [{"x": 200, "y": 250, "w": 2, "h": 10}]},
    {"pair": [2,3], "doors": [{"x": 250, "y": 300, "w": 10, "h": 2}]},
    {"pair": [1,3], "doors": []}
  ],
  "front_door": {"x": 128, "y": 32, "w": 8, "h": 2},
  "spatial": [[1,2,"right"], [1,3,"below"], [2,3,"right-below"]]
}
```

**설계 결정:**
- **coords 저장 규칙:** `[x1,y1,x2,y2,...]` flat array. `coords[0::2]`=x좌표, `coords[1::2]`=y좌표로 즉시 분리 가능.
- **Edge 정의:** 픽셀이 맞닿아 있는(인접한) 방 쌍 전체. 문 유무와 무관하게 경계 픽셀을 공유하는 모든 방 쌍이 edge로 등록된다.
- **Front Door 독립 필드:** 현관문(G=15)은 방과 방을 연결하지 않으므로 `edges`가 아닌 별도 `front_door` 필드로 관리.

### 4.2 Layer 2: Arrow 데이터 (구조화 저장)

JSONL을 파싱하여 HuggingFace `datasets` 라이브러리의 Arrow 포맷으로 저장. 토크나이징은 이 단계에서 수행하지 않고 증강 단계에서 동적으로 적용한다.

**Arrow가 JSONL보다 나은 이유:**
- Memory-mapped I/O → RAM보다 큰 데이터셋도 처리 가능
- Columnar format → 특정 필드만 선택 로드 가능
- HuggingFace `datasets` 생태계와 완벽 호환
- 매 iteration마다 JSON 파싱을 반복하지 않아 학습 속도 향상

**스키마 정규화:**
- `front_door: null` → `[]` (길이 0 리스트로 정규화, Arrow 스키마 일관성 유지)
- `spatial: [[int,int,str]]` → `[{"rid_a":int, "rid_b":int, "direction":str}]`

**토크나이징을 Arrow 단계에서 하지 않는 이유:** 조건부 생성 (입력/출력 분리) + 삭제 방식 증강을 반영하면, 증강 단계에서 동적으로 토큰 시퀀스를 구성해야 한다. Arrow에는 구조화된 데이터를 저장하고 토크나이징은 증강 후에 수행한다. 이 덕분에 동일한 Arrow 데이터셋을 어떤 LLM 토크나이저에도 적용 가능하다.

### 4.3 입출력 시퀀스 구조

#### ROOM_SUMMARY 형식

```
<ROOM_SUMMARY> <TOTAL> 전체수 <TYPE:종류1> <COUNT> 종류1수 <TYPE:종류2> <COUNT> 종류2수 ... <END_ROOM_SUMMARY>
```

- `<TOTAL>` + 숫자: `<TOTAL>` 레이블 토큰 뒤에 LLM 기본 숫자 토큰으로 전체 방 개수 표기.
- `<TYPE:t> <COUNT>` + 숫자: 해당 종류의 방 개수. 증강 시 `<TOTAL>` 쌍 또는 개별 타입 쌍을 독립적으로 삭제 가능.

증강 예시 — 총 5개 방 중 침실 정보만 유지:
```
<ROOM_SUMMARY> <TOTAL> 5 <TYPE:bedroom> <COUNT> 2 <END_ROOM_SUMMARY>
```
→ 모델은 "총 5개 방인데 침실 2개만 알려줬으니, 나머지 3개는 직접 판단해야 한다"는 상황을 이해한다.

#### 전체 정보 입출력 (증강 미적용)

`<INPUT>` ~ `<END_INPUT>`, `<OUTPUT>` ~ `<END_OUTPUT>` 사이의 토큰은 줄바꿈·공백 없이 연속으로 이어붙인다. 아래 예시는 가독성을 위해 줄바꿈을 표기한 것이다.

**입력 (조건):**
```
<INPUT>
  <ROOM_SUMMARY> <TOTAL> 3 <TYPE:kitchen> <COUNT> 1 <TYPE:livingroom> <COUNT> 1 <TYPE:bedroom> <COUNT> 1 <END_ROOM_SUMMARY>
  <FRONT_DOOR> <X:128> <Y:32> <SEP_DOOR> <X:8> <Y:2> <END_DOOR>
  <ROOM> <RID:1> <TYPE:livingroom> <X:100> <Y:200> <X:100> <Y:300> <X:200> <Y:300> <X:200> <Y:200> <END_ROOM>
  <ROOM> <RID:2> <TYPE:bedroom> <X:200> <Y:200> <X:200> <Y:300> <X:300> <Y:300> <X:300> <Y:200> <END_ROOM>
  <ROOM> <RID:3> <TYPE:kitchen> <X:100> <Y:300> <X:100> <Y:400> <X:200> <Y:400> <X:200> <Y:300> <END_ROOM>
  <EDGE> <RID:1> <RID:2> <DOOR> <X:200> <Y:250> <SEP_DOOR> <X:2> <Y:10> <END_DOOR> <END_EDGE>
  <EDGE> <RID:2> <RID:3> <DOOR> <X:250> <Y:300> <SEP_DOOR> <X:10> <Y:2> <END_DOOR> <END_EDGE>
  <EDGE> <RID:1> <RID:3> <NO_DOOR> <END_EDGE>
  <SP> <RID:1> <RID:2> <REL:right> <END_SP>
  <SP> <RID:1> <RID:3> <REL:below> <END_SP>
  <SP> <RID:2> <RID:3> <REL:right-below> <END_SP>
<END_INPUT>
```

**출력 (정답 — 항상 전체 정보):**
```
<OUTPUT>
  <FRONT_DOOR> <X:128> <Y:32> <SEP_DOOR> <X:8> <Y:2> <END_DOOR>
  <ROOM> <TYPE:livingroom> <X:100> <Y:200> <X:100> <Y:300> <X:200> <Y:300> <X:200> <Y:200> <END_ROOM>
  <ROOM> <TYPE:bedroom> <X:200> <Y:200> <X:200> <Y:300> <X:300> <Y:300> <X:300> <Y:200> <END_ROOM>
  <ROOM> <TYPE:kitchen> <X:100> <Y:300> <X:100> <Y:400> <X:200> <Y:400> <X:200> <Y:300> <END_ROOM>
  <DOOR> <X:200> <Y:250> <SEP_DOOR> <X:2> <Y:10> <END_DOOR>
  <DOOR> <X:250> <Y:300> <SEP_DOOR> <X:10> <Y:2> <END_DOOR>
<END_OUTPUT>
```

**출력 구조 핵심 사항:**
- 출력의 방(`<ROOM>`)에는 `<RID:N>`이 없다. `<TYPE:xxx>`와 좌표만 포함한다.
- 문(`<DOOR>`)은 `<EDGE>` 블록 없이 독립적으로 나열된다. `<SEP_DOOR>`가 중심 좌표(cx, cy)와 크기(w, h)를 구분한다.
- 현관문(`<FRONT_DOOR>`)도 동일한 형식 (`<SEP_DOOR>`로 중심좌표/크기 구분).

**Chat Template (Qwen2.5 형식 기준, LLM 로드 및 훈련 파트에서 구현):**
```
<|im_start|>system
You are a floor plan generator. Given room conditions, generate complete floorplan coordinates.<|im_end|>
<|im_start|>user
<INPUT>...<END_INPUT><|im_end|>
<|im_start|>assistant
<OUTPUT>...<END_OUTPUT><|im_end|>
```

#### 학습 시 Loss 처리

Chat template으로 구성된 전체 시퀀스에서 **system + user 턴(입력) 부분의 loss는 무시 (-100으로 마스킹)**, assistant 턴(`<OUTPUT>` ~ `<END_OUTPUT>`) 부분만 학습한다.

---

## 5. 전체 파이프라인

```
평면도 PNG (RPLAN 데이터셋)
        │
        ▼
┌─────────────────────────┐
│  Step 1: 정보 추출       │  PNG → BGRA 채널 분리 → 방/문/Edge/Spatial 추출
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  JSONL 원본 데이터       │     │  Step 2: Vocab 빌드      │
│  (사람이 읽는 형태)       │     │  토큰 목록 + 토크나이저   │
└──────────┬──────────────┘     └──────────┬──────────────┘
           │                               │
           ▼                               │
┌─────────────────────────┐                │
│  Step 3: Arrow 변환      │                │
│  JSONL → 구조화 데이터    │                │
└──────────┬──────────────┘                │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: 데이터 증강 + 토크나이징                      │
│  변형 증강 → 삭제 증강 → 토큰 ID 시퀀스 생성           │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│  Pre-Stage: 새 토큰 Embedding 워밍업                   │
│  새 커스텀 토큰 embed + lm_head 행만 훈련              │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: LLM 3-Stage Fine-tuning: SFT → DPO → GRPO   │
│  condition + output 토큰 → 평면도 생성 모델            │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│  Step 6: 추론 + 시각화                                │
│  조건 입력 (4가지 모드) → 증강 (선택) → N출력           │
│  토큰 파싱 → 평면도 시각화 + 메타데이터 저장            │
└──────────────────────────────────────────────────────┘
```

| Step | 단계 | 입력 | 출력 |
|------|------|------|------|
| 1 | 평면도 정보 추출 | RPLAN PNG (BGRA) | JSONL 샤드 |
| 2 | Vocabulary 빌드 | Pretrained 토크나이저 | vocab_extension.json + 확장 토크나이저 |
| 3 | Arrow 변환 | JSONL | Arrow DatasetDict (train/val/test) |
| 4 | 증강 + 토크나이징 | Arrow + 증강 설정 | (condition_tokens, output_tokens) |
| Pre-Stage | 새 토큰 Embedding 워밍업 | 토큰 시퀀스 배치 | 워밍업된 embed_tokens + lm_head |
| SFT | DoRA Fine-tuning | pre_stage/final + 토큰 시퀀스 배치 | DoRA 병합된 Fine-tuned 모델 |
| 5 | DPO → GRPO (예정) | 토큰 시퀀스 배치 | Fine-tuned 모델 |
| 6 | 추론 + 시각화 | 조건 텍스트/JSONL/Arrow/텍스트 파일 | 평면도 이미지 + JSON + 메타데이터 |

---

## 6. Step 1: 평면도 PNG → JSONL 추출

### RPLAN 데이터셋 채널 구조

RPLAN PNG는 BGRA 4채널 이미지 (256×256px)이며 각 채널이 다른 정보를 담는다.

| 채널 | 내용 | 주요 값 |
|------|------|--------|
| B (ch0) | 구조 레이블 | 벽=127, 현관문=255 |
| G (ch1) | 공간 타입 | 0~12: 방 종류, 13: 외벽 외부, 14: exterior_wall, 15: 현관문, 17: 인테리어 문 |
| R (ch2) | 방 인스턴스 ID | 방마다 고유한 픽셀값 |
| A (ch3) | 영역 구분 | 외부=0, 내부=255 |

### 10단계 처리 파이프라인

```
0. PNG 로드 + BGRA 채널 분리           (channel_parser.py)
1. 방 타입별 CCL (Connected Component Labeling)  (room_extractor.py)
2. 노이즈 제거 (min_room_area < 30px 제외)
3. 직교 폴리곤 근사 → 꼭지점 좌표 추출
4. 외곽선(outline) 추출 (외벽 외부 차집합)
5. 현관문 추출 (G==15, 가장 큰 컴포넌트)  (door_extractor.py)
6. 인테리어 문 추출 (G==17, L자-형 분해)
7. Raster scan 순서로 정렬 (centroid y→x)   (serializer.py)
8. Edge 구성 (직접 인접 + 문 연결)          (edge_builder.py)
9. Spatial 관계 계산 (8방위)               (spatial_calculator.py)
10. JSONL 직렬화 + 샤드 저장
```

### 핵심 알고리즘: 직교 폴리곤 추출

평면도의 방은 항상 직각으로 이루어진 다각형이다. OpenCV `approxPolyDP`는 일반 근사를 수행하므로 직각 보장이 안 된다. 대신 전용 직교 폴리곤 추출 알고리즘을 구현한다.

```
1. 방 픽셀 마스크에서 외곽선(contour) 추출
2. 방향 전환점(코너)만 추출: 수평 → 수직 또는 수직 → 수평으로 방향이 바뀌는 점
3. Canonical 순서 정규화: 시계방향(CW), top-left 코너 시작
```

### 핵심 알고리즘: L자형 문 분해

인테리어 문(G=17)이 두 방의 경계에서 병합되어 L자형으로 나타날 때 자동으로 두 개의 직사각형 문으로 분해한다.

```
1. 문 컴포넌트의 투영 프로파일 계산 (수평/수직 방향)
2. 프로파일의 Valley(최솟값 구간) 탐지
3. Valley 기준으로 재귀 분해
4. Peak 구간은 직사각형 bbox로 보정
```

### 방 타입 병합 (room_type_merge.json)

RPLAN의 상세 방 타입을 더 일반적인 타입으로 병합한다.

| 원본 타입 | 병합 후 타입 |
|---------|------------|
| `masterroom`, `childroom` | `bedroom` |
| `diningroom` | `livingroom` |
| `walkin` | `storage` |

### 주요 모듈

| 파일 | 주요 클래스/함수 | 역할 |
|------|----------------|------|
| `channel_parser.py` | `ChannelData`, `load_bgra_image()`, `parse_channels()` | BGRA 채널 분리 |
| `room_extractor.py` | `RoomInstance`, `extract_room_instances()`, `extract_polygon_coords()` | 방 분리 + 직교 폴리곤 |
| `door_extractor.py` | `DoorInstance`, `extract_front_door()`, `extract_interior_doors()`, `decompose_door_component()` | 문 추출 + L자 분해 |
| `edge_builder.py` | `EdgeRecord`, `build_edges()` | Edge 구성 |
| `spatial_calculator.py` | `build_spatial_relations()`, `_compute_direction()` | 8방위 공간관계 |
| `serializer.py` | `sort_rooms_raster_order()`, `build_plan_record()`, `append_to_jsonl()` | JSONL 직렬화 |

---

## 7. Step 2: Vocabulary 빌드

### 목적

Pretrained LLM의 토크나이저에 평면도 전용 커스텀 토큰을 추가하고, 추후 학습 및 추론에서 동일하게 사용할 수 있도록 저장한다.

### 처리 흐름

```
1. config의 model.hub_id로 Pretrained 토크나이저 로드 (예: Qwen/Qwen2.5-Coder-7B)
2. token_definitions.py에서 커스텀 토큰 목록 생성
3. tokenizer.add_tokens(custom_tokens) 호출
4. vocab_extension.json 저장 (토큰 → ID 매핑)
5. 확장된 tokenizer 저장 (save_pretrained)
```

**저장 경로:** `data/models/{model.name}/tokenization/`
- 모델명별로 독립된 디렉토리에 저장되어 여러 베이스 모델을 동시에 관리 가능
- `model.name`은 `model.hub_id`의 slash 뒤 부분 (예: `Qwen/Qwen2.5-Coder-7B` → `Qwen2.5-Coder-7B`)

### vocab_extension.json 구조

```json
{
  "coord_x": {"<X:0>": 152000, "<X:1>": 152001, ...},
  "coord_y": {"<Y:0>": 152256, "<Y:1>": 152257, ...},
  "room_id": {"<RID:0>": 152512, ...},
  "room_type": {"<TYPE:livingroom>": ..., ...},
  "spatial_rel": {"<REL:right>": ..., ...},
  "structure": {"<INPUT>": ..., "<END_INPUT>": ..., "<OUTPUT>": ..., "<END_OUTPUT>": ..., ...}
}
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `token_definitions.py` | 카테고리별 커스텀 토큰 목록 정의 (순서 고정) |
| `vocab_builder.py` | HuggingFace 토크나이저 확장 + vocab_extension.json 저장 |

---

## 8. Step 3: JSONL → Arrow 변환

### 목적

사람이 읽는 JSONL 원본을 학습용 고속 Arrow 바이너리로 변환하고, train/val/test로 분리한다.

### 처리 흐름

```
1. JSONL 샤드 파일 목록 수집 (floorplans_*.jsonl)
2. 각 레코드를 Arrow 스키마에 맞게 정규화
   - front_door: null → []
   - spatial: [[a,b,"dir"]] → [{"rid_a":a, "rid_b":b, "direction":"dir"}]
   - doors: 단일 dict → list로 정규화
3. HuggingFace Dataset 생성
4. train/validation/test split 적용
5. 각 split을 Arrow 포맷으로 저장
6. 샘플 단위 검증 수행 (Arrow ↔ JSONL 원본 비교)
```

### 출력 구조

```
data/dataset/processed_dataset/rplan/arrow/
├── train/       (기본 99.4%)
├── validation/  (기본 0.1%)
└── test/        (기본 0.5%)
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `schema.py` | 명시적 Arrow 스키마 정의 (`get_floorplan_features()`) |
| `converter.py` | JSONL 파싱 + 스키마 정규화 + Arrow Dataset 생성 |
| `validator.py` | Arrow ↔ JSONL 원본 비교 검증 |

---

## 9. Step 4: 데이터 증강 + 토크나이징

### 설계 원칙

**증강 적용 순서:** 변형 증강 → 출력 확정 → 삭제 증강 (입력에만)

이 순서 덕분에 출력 정답은 항상 완전한 full information을 유지하면서도, 변형 증강의 효과(예: 재배정된 RID)가 입출력 양쪽에 일관되게 반영된다.

### 증강 전략 목록

#### 표현형 변형 (Shuffle) — 입력만 영향

| 전략 | 대상 | 학습 효과 |
|------|------|---------|
| `ShuffleRID` | 방 ID 번호 재배정 | ID 번호가 아닌 구조와 관계에 집중 |
| `ShuffleVertexOrder` | 꼭지점 리스트 시작점 회전 | 시작점 무관하게 동일한 도형 인식 |
| `ShuffleRoomOrder` | 방 나열 순서 (입력만) | 입력 순서 무관하게 동일한 평면도 생성 |
| `ShuffleEdgeOrder` | 엣지 나열 순서 (입력만) | 엣지 순서 무관하게 연결 구조 인식 |
| `ShuffleSpatialOrder` | Spatial 나열 순서 (입력만) | Spatial 순서 무관하게 위치관계 인식 |
| `ReverseSpatialRelation` | Spatial 방향 반전 | 대칭 위치관계 인식 |

#### 기하학적 변형 (Transform) — 입력 + 출력 모두 반영

| 전략 | 내용 |
|------|------|
| `Translate` | 평행이동 (256×256 경계 보장) |
| `Flip` | 수평/수직/양방향 뒤집기 |
| `ScaleAspect` | 종횡비 변경 (x/y 독립 스케일, 기본 0.7~1.3) |
| `Zoom` | 균일 확대/축소 (기본 0.7~1.3) |

#### 노이즈 — 입력에만 적용

| 전략 | 내용 |
|------|------|
| `GaussianNoise` | σ=3.0px 가우시안 노이즈 (확률 30%) |

#### 삭제 (Drop) — 입력 조건에만 적용

| 전략 | 기본 확률 | 학습 효과 |
|------|---------|---------|
| `DropBlock` | 0.5 | 방 전체 삭제 → 빈 공간에 방 배치 능력 학습 |
| `DropType` | 0.2 | 방 타입만 삭제 → 다른 조건으로 타입 추론 |
| `DropCoords` | 0.2 | 방 좌표만 삭제 → 종류와 관계만으로 좌표 생성 |
| `DropEdge` | 0.5 | 엣지 전체 삭제 → 불완전한 연결관계에서 생성 |
| `DropEdgePair` | 0.2 | 특정 RID 쌍 엣지 삭제 |
| `DropEdgeDoor` | 0.2 | 문 정보만 삭제 (인접 관계는 유지) |
| `DropSpatial` | 0.8 | 개별 Spatial 관계 삭제 |
| `DropFrontDoor` | 0.5 | 현관문 전체 삭제 |
| `DropFrontDoorCoords` | 0.4 | 현관문 좌표만 삭제 |
| `DropRoomSummaryTotal` | 0.5 | `<TOTAL>` + 숫자 쌍 삭제 (샘플 단위) |
| `DropRoomSummaryType` | 0.6 | 개별 `<TYPE:t> <COUNT>` + 숫자 쌍 삭제 (타입별 독립) |

### 토크나이징

증강이 완료된 구조화 데이터를 토큰 ID 시퀀스로 변환한다.

```python
# 조건(입력) 토큰 시퀀스
condition_tokens = build_condition_tokens(augmented_sample, vocab)
# → [<INPUT>, <ROOM_SUMMARY>, ..., <END_INPUT>]

# 정답(출력) 토큰 시퀀스 — 항상 완전한 정보
output_tokens = build_output_tokens(augmented_sample, vocab)
# → [<OUTPUT>, <FRONT_DOOR>, ..., <ROOM>, <TYPE:xxx>, ..., <DOOR>, ..., <END_OUTPUT>]
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `pipeline.py` | 증강 파이프라인 오케스트레이터 (`AugmentationPipeline`) |
| `strategies.py` | 15+ 증강 전략 순수 함수 구현 |
| `tokenizer.py` | `Vocab` 클래스, `build_condition_tokens()`, `build_output_tokens()` |
| `decoder.py` | `decode_tokens()`, `format_sample_report()` (디버깅용 역변환) |

---

## 10. Pre-Stage: 새 토큰 Embedding 워밍업

### 목적

새로 추가된 커스텀 토큰 (~1000개)과 기존 Pretrained 파라미터 간의 **gradient 스케일 차이를 완화**하기 위한 준비 단계. SFT(Stage 1) 이전에 먼저 수행한다.

새 토큰의 embedding 벡터와 lm_head 행이 기존 Pretrained embedding space에 대략적으로 자리잡으면, 이후 SFT 훈련이 더 안정적으로 진행된다.

### 훈련 설정

| 설정 | 값 |
|------|---|
| Freeze 대상 | Transformer 레이어 전체 (attention, FFN, layernorm 등) |
| Train 대상 | `embed_tokens.weight[new_token_ids]` + `lm_head.weight[new_token_ids]` |
| Learning rate | 5e-4 (공격적) |
| Epoch | 2 (기본) |
| 양자화 | 4bit (NF4, Double Quant은 안) — embed_tokens/lm_head는 bfloat16 유지 |
| 혼합 정밀도 | bf16 AMP (forward/backward bf16, optimizer state fp32) |
| 분산 학습 | DDP 지원 (`distributed.nproc_per_node` 설정으로 활성화) |

### 파라미터 동결 전략 (PartialEmbedding / PartialLMHead)

전체 파라미터를 `requires_grad=False`로 동결한 뒤, `embed_tokens`와 `lm_head`를 커스텀 모듈로 교체하여 **새 토큰 567행만 `nn.Parameter`로 분리**한다.

```
수정된 모델
├── embed_tokens: PartialEmbedding
│   ├── base_embed:  nn.Embedding(152232, 3584)  [frozen buffer]
│   └── new_embed:   nn.Parameter(567, 3584)     [훈련 대상]
└── lm_head: PartialLMHead
    ├── base_lm_head: nn.Linear(3584, 152232)    [frozen buffer]
    └── new_lm_head:  nn.Parameter(567, 3584)    [훈련 대상]
```

**forward 시 동작:**
- `PartialEmbedding`: frozen base로 전체 조회 후, 새 토큰 위치에 `index_put`으로 `new_embed` 값 교체 (gradient 흐름 유지)
- `PartialLMHead`: frozen base로 전체 logits 계산 후, 새 토큰 위치에 `scatter`로 `new_lm_head` 재계산 값 교체

**Gradient Hook 방식 대비 이점:**

| | Gradient Hook (기존) | PartialEmbedding (현재) |
|---|---|---|
| optimizer state | ~8.8GB (전체 152232행) | ~16MB (567행만) |
| gradient 계산 | 전체 행 계산 후 마스킹 | 필요한 행만 계산 |

**저장 시:** `merge_and_restore()`로 `new_embed` 값을 `base_embed.weight[new_ids]`에 병합하고 원본 `nn.Embedding`/`nn.Linear`로 복원한 뒤 `model.save_pretrained()`. 이로써 표준 HuggingFace 형식으로 저장되어 다음 Stage에서 `from_pretrained()`로 바로 로드 가능하다.

PEFT 어댑터(LoRA/DoRA)는 이 단계에서 사용하지 않는다.

### 계속 훈련 (Resume)

`resume.enabled=true`로 중단된 훈련을 재개할 수 있다.

**체크포인트 저장 구조 (`PreStageTrainer._save_checkpoint` 오버라이드):**

```
data/models/{model.name}/checkpoints/pre_stage/
└── checkpoint-{step}/
    ├── partial_state.pt      ← new_embed / new_lm_head 가중치만 별도 저장 (model.safetensors 없음)
    ├── optimizer.pt          ← AdamW state (new_embed, new_lm_head 두 파라미터만, ~16MB)
    └── trainer_state.json    ← step, epoch, best_model_checkpoint 등
```

> **model.safetensors를 저장하지 않는 이유:** Transformer 레이어는 항상 HuggingFace에서 새로 로드하므로 저장할 필요가 없다. 기존 방식(중간 체크포인트에서 `merge_and_restore()` 호출)은 `PartialEmbedding`의 `nn.Parameter` 객체를 소멸시키고 `_setup_partial_training()`이 새 객체를 생성하면서 optimizer의 Parameter 참조가 끊어지는 버그가 있었다. 이후 `optimizer.step()`이 소멸된 객체를 업데이트하려 해도 `grad=None`이므로 no-op이 되어 체크포인트 저장 이후의 훈련이 완전히 무효가 된다. `merge_and_restore`는 최종 저장(`run_pre_stage.py`)에서만 호출한다.

체크포인트 저장 흐름:
1. `partial_state.pt` 저장 — 현재 훈련된 new_embed/new_lm_head 값 보존 (DDP 환경에서는 `is_world_process_zero()` 가드로 rank 0만 저장)
2. `self.save_model`을 일시 no-op으로 교체 후 `super()._save_checkpoint()` 호출 → optimizer + trainer_state만 저장 (model.safetensors 건너뜀)

**Resume 로드 흐름:**
- 항상 HuggingFace에서 기본 모델을 새로 로드 (`load_model_and_tokenizer`)
- `_setup_partial_training()` — PartialEmbedding/PartialLMHead 구조 적용
- `trainer.train(resume_from_checkpoint=...)` → `_load_from_checkpoint` 오버라이드 호출
  1. `partial_state.pt` 로드 — new_embed/new_lm_head를 훈련된 값으로 직접 복원 (`super()` 호출 없음 — 중간 체크포인트에 `model.safetensors` 없음)
- optimizer.pt에서 AdamW state / step 복원 (Trainer 내부 `_load_optimizer_and_scheduler`가 자동 처리)

**`_load_best_model` 오버라이드:**
`load_best_model_at_end=true` 시 표준 모델 재로드 대신 `partial_state.pt`에서 new_embed/new_lm_head만 직접 복사한다 (key mismatch 방지).

### 증강 설정 관리 (Hydra config group)

증강 파라미터는 훈련 단계마다 독립된 파일로 관리되며, Hydra **config group** 방식으로 각 파이프라인 yaml에 합성된다.

```
config/training/augmentation/
├── pre_stage.yaml    ← Pre-Stage용 (완료)
├── sft.yaml          ← SFT용 (완료, pre_stage.yaml과 동일한 증강 전략)
└── dpo.yaml          ← DPO용 (추후)
```

`config/training/pre_stage/pipeline.yaml`의 `defaults` 선언:
```yaml
defaults:
  - training/augmentation: pre_stage   # cfg.augmentation으로 병합
  - _self_                             # pipeline.yaml 값이 최우선
```

- config 루트(`config/`)가 탐색 기준이므로 `training/augmentation: pre_stage` →
  `config/training/augmentation/pre_stage.yaml` 탐색
- `pre_stage.yaml` 내부는 `augmentation:` 래퍼 없이 내용만 작성 (group 이름이 키를 자동 생성)
- SFT, DPO, GRPO 파이프라인도 동일한 패턴으로 증강 설정 재사용/오버라이드 가능

### 데이터 구성 및 Chat Template

`AugmentationPipeline`에서 `(condition_tokens, output_tokens)` 쌍을 생성하고, 이를 디코딩하여 Qwen2.5의 Chat Template으로 감싼다.

```
<|im_start|>system
You are a floor plan generator. Given room conditions, generate complete floorplan coordinates.<|im_end|>
<|im_start|>user
<INPUT>...<END_INPUT><|im_end|>
<|im_start|>assistant
<OUTPUT>...<END_OUTPUT><|im_end|>
```

- **Loss 마스킹:** system + user 턴은 `labels=-100`, assistant 턴만 loss 계산
- **Dynamic Padding:** 배치 내 최대 길이로 right-padding (pad 위치도 `labels=-100`)

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/training/pre_stage/model_loader.py` | 4bit 로드 + `prepare_model_for_kbit_training` + `PartialEmbedding`/`PartialLMHead` 교체 + `merge_and_restore` |
| `src/training/pre_stage/dataset.py` | Arrow 로드 → 증강 → Chat Template 적용 → `{input_ids, labels, attention_mask}` |
| `src/training/pre_stage/collator.py` | Dynamic padding + label 마스킹 |
| `src/training/pre_stage/trainer.py` | `TrainingArguments` 구성 + `PreStageTrainer` 빌드 (`_save_checkpoint`, `_load_from_checkpoint`, `_load_best_model` 오버라이드 포함) |
| `scripts/training/run_pre_stage.py` | Hydra 진입점, seed 고정, Resume 분기, 훈련 후 `merge_and_restore` 호출 및 저장 |
| `config/training/pre_stage/pipeline.yaml` | 모델, 양자화, 데이터, 훈련 하이퍼파라미터, resume 설정 |
| `config/training/augmentation/pre_stage.yaml` | Pre-Stage용 증강 파라미터 (Hydra config group, `cfg.augmentation`으로 병합) |
| `tests/training/pre_stage/validate_resume.py` | Resume 체크포인트 검증 스크립트 (partial_state.pt 존재/형태/복원 확인) |
| `tests/training/pre_stage/validate_save_and_load.py` | 저장/로드 후 optimizer 업데이트 정상 동작 검증 (체크포인트 저장 후 훈련이 계속 진행되는지 2-case 검증) |

### 체크포인트 및 출력

```
outputs/training/pre_stage/
└── YYYY-MM-DD/HH-MM-SS/       # Hydra 실행 로그 + 설정 스냅샷

data/models/{model.name}/
└── checkpoints/pre_stage/
    ├── checkpoint-{step}/      # 에폭별 자동 저장 (save_total_limit 초과 시 오래된 것 삭제)
    │   ├── partial_state.pt    # new_embed / new_lm_head 가중치 (model.safetensors 없음)
    │   ├── optimizer.pt        # AdamW state (~16MB)
    │   └── trainer_state.json
    └── final/                  # 최종 병합 모델 (tokenizer 포함, 표준 HuggingFace 형식)
```

---

## 11. Step 5: LLM 학습 (SFT + GRPO)

### 3-Stage Fine-tuning 전략

Pre-Stage에서 워밍업된 모델(`pre_stage/final`)을 기반으로 3단계 fine-tuning을 수행한다. LLM 학습 시 QDoRA(Quantized DoRA)를 사용한다. 혼합 정밀도(bf16 AMP)를 적용한다.

### Stage 1: SFT (Supervised Fine-tuning) — 완료

#### 목적

Pre-Stage 워밍업 이후, DoRA(Weight-Decomposed Low-Rank Adaptation)를 통해 Transformer 전체 레이어를 fine-tuning하여 모델이 평면도 생성 태스크에 적응하도록 한다.

#### Pre-Stage와의 차이점

| 항목 | Pre-Stage | SFT |
|------|-----------|-----|
| 모델 로드 출처 | HF Hub | 로컬 `pre_stage/final` 경로 |
| 훈련 범위 | new_embed/lm_head 행 567개 | DoRA adapter (attention/MLP 전 레이어) |
| 특수 모듈 | PartialEmbedding / PartialLMHead | 불필요 (가중치 이미 병합됨) |
| resize_token_embeddings | 필요 | 불필요 (vocab_size 이미 확장됨) |
| 체크포인트 포맷 | `partial_state.pt` (커스텀) | `adapter_model.safetensors` (표준 PEFT) |
| Resume 처리 | 커스텀 `_load_from_checkpoint` | 표준 PEFT Resume |

#### DoRA (Weight-Decomposed Low-Rank Adaptation)

`LoraConfig(use_dora=True)`로 활성화. 일반 LoRA에서 weight를 magnitude(크기)와 direction(방향)으로 분리하여 **direction만 low-rank로 학습**한다. 이로 인해 일반 LoRA 대비 full fine-tuning에 가까운 학습 품질을 제공하면서도 파라미터 수는 최소화된다.

| 설정 | 값 |
|------|---|
| Train 대상 | DoRA adapter (lora_A, lora_B, lora_magnitude_vector) |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| rank (r) | 32 |
| lora_alpha | 64 (실효 스케일 = alpha/r = 2.0) |
| lora_dropout | 0.05 |
| 학습률 | 2e-4 |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| 양자화 | 4bit NF4 |
| 분산 학습 | DDP 지원 (`distributed.nproc_per_node` 설정으로 활성화) |

**DoRA 파라미터 수 계산 (Qwen2.5-Coder-7B 기준):**
- 28 Transformer 레이어 × 7 target_modules × 3 텐서(lora_A, lora_B, lora_magnitude_vector) = 588개 파라미터 텐서
- scalar 훈련 가능 파라미터: 약 41,760,768개 (~42M)

#### 모델 로드 흐름

```
1. AutoTokenizer.from_pretrained(pre_stage/final)
   → tokenizer.json에서 커스텀 토큰 567개 포함 vocab 로드

2. AutoModelForCausalLM.from_pretrained(
       pre_stage/final,      # 로컬 경로 (인터넷 연결 불필요)
       quantization_config,  # 4bit NF4
       dtype=torch.bfloat16,
       # device_map="auto" 미사용: DDP와 호환되지 않음 (model parallelism vs data parallelism 충돌)
   )
   → model.safetensors에서 전체 가중치 로드 (커스텀 토큰 이미 포함)
   → resize_token_embeddings() 호출 불필요

3. prepare_model_for_kbit_training(model, ...)
   → gradient checkpointing 활성화

4. LoraConfig(..., use_dora=True)
5. get_peft_model(model, lora_config)
   → attention/MLP 레이어에 DoRA adapter 주입
```

#### merge_dora_and_save

훈련 완료 후 DoRA adapter를 base model에 병합하고 **clean bf16**으로 저장한다.

**QLoRA/QDoRA 국룰 적용:**
```python
merged_model = model.merge_and_unload()  # NF4 + bf16 adapter 병합
merged_model = merged_model.to(torch.bfloat16)  # 모든 weights → bf16
merged_model.save_pretrained(str(save_dir))  # quantization_config 제거
```

merge_and_unload() 후 weights는 bf16으로 병합되지만, model object가 여전히 `quantization_config`를 가지고 있어 save_pretrained() 시 이상 동작할 수 있다. 명시적 `to(bfloat16)` 호출로 quantization_config의 간섭을 물리적으로 제거한다.

#### 체크포인트 구조

```
data/models/{model.name}/checkpoints/sft/
├── checkpoint-{step}/
│   ├── adapter_model.safetensors  # DoRA adapter 가중치
│   ├── adapter_config.json        # use_dora: true 포함
│   ├── optimizer.pt               # AdamW state
│   └── trainer_state.json
└── final/                         # merge_and_unload + bf16 변환 후
    ├── model.safetensors          # DoRA 병합된 bf16 모델 (quantization_config 없음)
    ├── tokenizer.json
    └── config.json                # quantization_config 제거됨
```

#### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/training/sft/model_loader.py` | 로컬 pre_stage/final 로드 + DoRA 적용 + `merge_dora_and_save` |
| `src/training/sft/trainer.py` | `TrainingArguments` + 표준 `Trainer` 빌드 (패치 불필요) |
| `scripts/training/run_sft.py` | Hydra 진입점, seed 고정, Resume 분기, 훈련 후 `merge_dora_and_save` 호출 |
| `config/training/sft/pipeline.yaml` | DoRA, 학습률, model_dir 등 SFT 전체 설정 |
| `config/training/augmentation/sft.yaml` | SFT용 증강 파라미터 (pre_stage.yaml과 동일) |
| `tests/training/sft/validate_sft.py` | 로드·DoRA구조·훈련·저장·Resume 통합 검증 |

#### DDP (Data Parallel) 지원

Pre-Stage와 SFT 모두 DDP를 지원한다. `distributed.nproc_per_node` 값이 2 이상이면 `main(cfg)` 진입 직후 `os.execvp`로 torchrun 프로세스를 자동으로 띄운다.

**4bit 양자화 + DDP 호환성:**
- `device_map="auto"`(model parallelism, DDP와 충돌)는 제거됨
- 4bit 양자화(frozen 가중치)는 `requires_grad=False`이므로 DDP all-reduce 대상 제외 → 호환됨
- DoRA adapter(bf16, `requires_grad=True`)만 all-reduce됨

**Pre-Stage DDP 주의사항:**
- `_save_checkpoint`: `is_world_process_zero()` 가드로 rank 0만 `partial_state.pt` 저장
- `_save_checkpoint` / `_load_from_checkpoint` / `_load_best_model`: DDP 래퍼(`DistributedDataParallel`) 내부 실제 모델에 `.module`으로 접근
- 최종 저장 시: `trainer.accelerator.unwrap_model(trainer.model)`로 언래핑 후 `merge_and_restore` 및 `save_pretrained` 호출

**실행:**
```bash
# DDP 2-GPU (config에 저장하거나 override로 지정)
uv run python scripts/training/run_pre_stage.py distributed.nproc_per_node=2
uv run python scripts/training/run_sft.py distributed.nproc_per_node=2

# 단일 GPU (기본값, 동일 명령어)
uv run python scripts/training/run_pre_stage.py
uv run python scripts/training/run_sft.py
```

---

### Stage 2: DPO (Direct Preference Optimization) — 구현 예정

선호/비선호 쌍(preferred/rejected)을 활용하여 생성 품질을 개선한다. 기하학적 제약(방 겹침, 경계 초과 등)을 위반하는 출력을 rejected 샘플로 구성.

### Stage 3: GRPO (Group Relative Policy Optimization) — 완료

RLVR(Rule-based Verifiable Reward) 기반 강화학습으로 규칙 기반 보상 함수를 적용한다. 7개의 보상함수를 통해 평면도 구조 유효성, 겹침 없음, 좌표 정확도, 연결성 등을 평가하고, 토큰 수준의 신용할당으로 오류 토큰에 페널티를 적용하여 생성 정밀도를 극대화한다. 

자세한 내용은 [Step 6: GRPO 강화학습](#12-step-6-grpo-강화학습) 섹션을 참고.

### 학습 데이터 구성 (공통)

- **입력:** condition_tokens (삭제 증강이 적용된 부분 정보)
- **출력:** output_tokens (모든 방 + 모든 Edge의 완전한 정보)
- **Loss 마스킹:** 입력 토큰 구간은 loss 무시 (ignore_index=-100), 출력 토큰 구간만 학습

---

## 12. Step 6: GRPO 강화학습

### 개요

SFT 단계에서 습득한 기본 생성 능력을 바탕으로, 규칙 기반 보상함수(Rule-based Verifiable Reward, RLVR)를 통해 평면도 생성 정밀도를 개선한다. TRL의 `GRPOTrainer`를 상속한 `GDPOTrainer`로 구현했으며, GDPO(Group Relative Policy Optimization + Decoupled Normalization)와 토큰 수준의 신용할당(Token-level Credit Assignment)을 적용한다.

### 핵심 개념

#### GDPO (Group Relative Policy Optimization with Decoupled Normalization)

TRL의 `normalize_then_sum` 모드를 기반으로, 각 보상함수별로 독립적인 그룹 정규화를 수행한 후 가중합을 취하는 방식.

**수식:**
$$A_k^{(i)} = \frac{r_k^{(i)} - \mathbb{E}[r_k]}{\sqrt{\text{Var}(r_k)} + \epsilon} \quad \text{for } k=1,\ldots,K$$
$$A^{(i)} = \sum_{k=1}^{K} w_k \cdot A_k^{(i)}$$

- $r_k$: 보상함수 $k$의 점수 ($K=7$개 보상함수)
- $\mathbb{E}[r_k]$, $\text{Var}(r_k)$: 그룹 내 평균/분산 ($G$개 샘플)
- $w_k$: 보상함수 $k$의 가중치
- $A^{(i)}$: 시퀀스 수준 advantage (정책 그래디언트에 스칼라로 사용)

**장점:**
- 보상함수 간 스케일 차이를 자동으로 정규화 (수동 정규화 불필요)
- 각 보상이 학습에 동등한 영향력을 가짐

#### 토큰 수준 신용할당 (Token-level Credit Assignment)

시퀀스 수준의 advantage를 토큰 수준으로 세분화. 오류 토큰(format violation, 비정상 좌표 등)에 대해 차등 페널티를 부여한다.

**수식:**
$$a_t = A \cdot (1 - m_t) - |A| \cdot \lambda \cdot m_t$$

- $A$: 시퀀스 advantage (스칼라)
- $m_t$: 오류 마스크 (0 또는 1)
- $\lambda$: 페널티 스케일 (기본값 1.5~2.0)
- $a_t$: 토큰 $t$의 advantage

**특징:**
- 정상 토큰: $A$ 그대로 사용
- 오류 토큰: advantage 부호와 무관하게 항상 음수 방향 페널티 ($-|A| \times \lambda$)

#### 하드 게이트 (Hard Gate)

Format reward가 0 (구조 오류)이면 모든 다른 보상을 0으로 설정. 구조 오류 시 다른 기하학적 특성을 평가할 수 없기 때문.

### 7개 보상함수 (RLVR)

| 보상 | 계산 방식 | 토큰 신용할당 |
|------|---------|-------------|
| **R_format** | 구조 유효성 (방/좌표 블록 정상) | ✅ 오류 인덱스 추적 |
| **R_count_total** | 방 총 개수 일치도 (binary) | ❌ 시퀀스 수준 |
| **R_count_type** | 방 종류별 개수 정확도 | ❌ 시퀀스 수준 |
| **R_orthogonality** | 모든 꼭짓점 직각도 | ✅ 비정상 꼭짓점의 X, Y 토큰 |
| **R_no_overlap** | 방 간 겹침 없음 (Shapely polygon) | ✅ 교차점 근처 좌표 토큰 |
| **R_connectivity** | 인접 방 간 Hungarian matching + 문 근접도 | ❌ 시퀀스 수준 |
| **R_spatial** | 방 간 위치관계 정확도 (8방위) | ❌ 시퀀스 수준 |

**주요 특징:**
- 모든 보상은 [0, 1] 범위로 정규화
- R_format = 0이면 다른 모든 보상도 0 (하드 게이트)
- 신용할당이 활성화된 보상(✅)은 오류 토큰 인덱스를 추적하여 target_token_loss 계산에 활용

### 구현 구조

```
src/training/grpo/
├── rewards/
│   ├── parser.py              # 출력 토큰 → ParsedFloorplan 변환
│   │   └── ParsedFloorplan: success, level, rooms[], doors[], error_indices
│   ├── format_reward.py       # R_format 계산
│   ├── count_reward.py        # R_count 계산
│   ├── geometry_reward.py     # R_orthogonality, R_no_overlap 계산
│   ├── connectivity_reward.py # R_connectivity 계산 (Hungarian matching)
│   ├── spatial_reward.py      # R_spatial 계산 (8방위 분류)
│   ├── credit_assignment.py   # 오류 마스크 + 토큰 advantage 계산
│   └── __init__.py            # compute_all_rewards (모든 보상 통합 + 캐싱)
├── trainer.py                 # GDPOTrainer (TRL GRPOTrainer 오버라이드)
├── dataset.py                 # GRPOPromptDataset (프롬프트 + 메타데이터)
├── advantage.py               # GDPO 정규화 + 토큰 advantage 함수
├── model_loader.py            # SFT final 로드 + DoRA + bf16 저장
└── __init__.py                # 공개 API
```

### 훈련 흐름

```
1. SFT final 모델 로드
   - SFT final은 bf16으로 저장됨 (QLoRA/QDoRA 국룰)
   - load_model_and_tokenizer가 quantization_config로 다시 NF4 양자화
   - 결과: NF4 base (frozen) + DoRA adapter 준비

2. GRPOPromptDataset 초기화
   → train.arrow 로드
   → 프롬프트 + 메타데이터(방 개수, 타입, 연결성 등) 추출

3. GDPOTrainer.train() 시작

4. 각 스텝별:
   a. 모델이 N개 완성(num_generations=4, 또는 설정값)
   b. 각 완성에 대해 7개 보상함수 순차 실행
      - compute_all_rewards()로 모든 보상 한 번에 계산 (파싱 한 번)
      - 캐싱으로 중복 파싱 방지
   c. GDPO 그룹 정규화 (모든 프로세스 데이터 수집)
   d. 토큰 신용할당 (로컬 프로세스만)
   e. 정책 그래디언트 계산 (target_token_loss)
   f. Backward + 옵티마이저 업데이트

5. 훈련 완료 후:
   - 중간 체크포인트: adapter_model.safetensors만 저장 (PEFT 기본)
   - 최종 모델: merge_dora_and_save(bf16)
     → DoRA 병합
     → .to(torch.bfloat16) 명시적 캐스팅
     → save_pretrained() (NF4 역변환 패치 불필요)

6. 결과:
   - data/models/{model}/checkpoints/grpo/final/
     → model.safetensors (bf16 clean)
     → config.json, tokenizer.json 등 (bitsandbytes 불필요)
```

### 모델 dtype 관리 (torch_dtype 메타데이터 기반)

**QLoRA/QDoRA + torch_dtype 메타데이터 전략:**

이 프로젝트는 중간 체크포인트에서 다시 훈련이 일어나므로 (SFT → GRPO), **명시적 dtype 변환 대신 torch_dtype 메타데이터를 활용**하여 원본 정밀도를 보존한다.

| 단계 | 저장/로드 방식 | dtype 구성 |
|------|---|---|
| **SFT final 저장** | `merge_and_unload()` → `config.torch_dtype="bfloat16"` → `save_pretrained()` | merge 결과 그대로 (변환 없음), torch_dtype 메타데이터만 포함 |
| **GRPO 로드** | `from_pretrained(..., torch_dtype=bf16, quantization_config=nf4)` | torch_dtype + quantization_config 함께 적용 → NF4 + bf16 adapter |
| **GRPO 훈련 중** | base frozen, adapter trainable | NF4 base + bf16 adapter (혼합) |
| **GRPO final 저장** | `merge_and_unload()` → `config.torch_dtype="bfloat16"` → `save_pretrained()` | merge 결과 그대로, torch_dtype 메타데이터 포함 |
| **추론 로드** | `from_pretrained(..., torch_dtype=bf16)` | torch_dtype 메타데이터에 따라 bf16으로 로드 |

**torch_dtype 메타데이터 방식의 장점:**
1. **저장 오버헤드 최소화** — merge_and_unload() 결과를 그대로 저장 (명시적 변환 없음)
2. **원본 정밀도 보존** — 다양한 dtype이 섞여 있는 상태 유지 (float32 LayerNorm, uint8 NF4, 병합된 가중치 등)
3. **중간 체크포인트 재훈련 최적화** — SFT→GRPO 계속 훈련 시 원본 정밀도 자동 보존
4. **로딩 유연성** — 필요에 따라 다른 torch_dtype으로 로드 가능
5. **표준 준수** — HuggingFace 공식 권장 방식

**구현 상세:**
- `merge_dora_and_save()`에서: `model.config.torch_dtype = "bfloat16"`으로 메타데이터 설정
- 로드 시: `from_pretrained(..., torch_dtype=torch.bfloat16)`로 메타데이터 존중하여 로드
- 양자화 재적용: torch_dtype과 quantization_config는 독립적으로 동작 (로드 시 둘 다 명시 가능)

**왜 torch_dtype 메타데이터 방식인가? (Option 3 선택 근거)**

원래 QLoRA 표준(`merge_and_unload() + .to(bfloat16)`)은 **최종 추론용 모델** 저장을 위해 고안되었다. 하지만 이 프로젝트는:
- **SFT 훈련** → SFT final 저장 (중간 체크포인트)
- **GRPO 훈련 시작** ← SFT final 로드 → GRPO final 저장

이렇게 **중간 체크포인트에서 재훈련이 일어난다**. 이 경우:

| 전략 | 장점 | 단점 | 적합성 |
|------|------|------|--------|
| **Option 1: selective float32** | LayerNorm만 float32 | 구현 복잡, 호환성 불확실 | × |
| **Option 2: full float32** | 최고 정밀도 | 모델 크기 증가 (2배), 슬로우 | △ (과도함) |
| **Option 3: torch_dtype 메타데이터** | 저장 오버헤드 없음, 원본 보존, 로딩 유연성 | 메타데이터 의존성 | ✓ **최적** |

**Option 3이 최적인 이유:**
1. **SFT→GRPO 계속 훈련**: SFT final의 원본 정밀도(float32 LayerNorm + uint8 NF4)가 자동 보존됨
2. **GRPO 로드**: torch_dtype=bf16 + quantization_config=nf4로 명시하면 자동으로 NF4 + bf16 adapter 재구성
3. **GRPO final**: 마찬가지로 torch_dtype 메타데이터만 설정 → 추론 시 bf16으로 로드
4. **저장 비용**: 명시적 .to(bfloat16) 변환 불필요 → GPU 연산 부담 감소
5. **유연성**: 필요시 다른 torch_dtype으로 로드 가능 (예: 추론 최적화 시 float32 로드 고려)

### 설정 파일 (config/training/grpo/pipeline.yaml)

**주요 파라미터:**

```yaml
model:
  model_dir: data/models/Qwen2.5-Coder-7B/checkpoints/sft/final

rewards:
  format:
    enabled: true
    weight: 1.0
    hard_gate: true  # R_format=0 → 모든 다른 보상도 0
    credit_assignment: true
    penalty_scale: 2.0  # 오류 토큰 페널티
  count_total:
    enabled: true
    weight: 0.5
    credit_assignment: false
  orthogonality:
    enabled: true
    weight: 1.5
    credit_assignment: true
    penalty_scale: 1.5
  # ... (나머지 4개 보상)

advantage:
  eps: 1.0e-8  # GDPO 정규화에서 분모 최소값

data:
  max_completion_length: 1024
  aug_pipeline_config: config/training/augmentation/grpo.yaml

training:
  num_train_epochs: 3
  num_generations: 4
  per_device_train_batch_size: 2
  # ... (표준 훈련 파라미터)
```

### 실행 방법

```bash
uv run python scripts/training/run_grpo.py
```

DDP 활성화:
```bash
uv run torchrun --nproc_per_node=2 scripts/training/run_grpo.py
```

---

## 13. Step 7: 추론 및 시각화

### 개요

Pre-Stage 워밍업과 SFT 훈련을 완료한 LLM 모델을 사용하여 사용자 조건으로부터 완전한 평면도를 생성한다. 이 단계는 세 가지 주요 기능을 포함한다:

1. **다양한 입력 모드 지원**: JSONL/Arrow 데이터셋, 텍스트 파일(사전 증강 완료) 직접 입력
2. **1대N 출력 생성**: 동일 입력에서 N개의 서로 다른 출력 생성 (sampling 모드)
3. **구조화된 출력 저장**: 입력/출력 조건, 메타데이터, 시각화 이미지를 계층적 디렉토리에 저장

### 시스템 아키텍처

```
┌─────────────────────────────────────┐
│ 입력 처리 (4가지 모드)               │
├─────────────────────────────────────┤
│ ① JSONL 파일                        │ ─┐
│ ② Arrow 데이터셋                    │  │
│ ③ 텍스트 파일 (txt_dir)             │  │ → load_samples()
│ ④ 특정 plan_id 필터링               │  │   (조건 토큰화)
└─────────────────────────────────────┘ ─┘
                  │
                  ▼
     ┌──────────────────────────┐
     │ 증강 파이프라인 (선택적)  │
     │ txt_dir 모드는 생략      │
     └──────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │ 모델 추론 (N출력 루프)             │
  │ generate_floorplan(              │
  │   condition_tokens,              │
  │   model, tokenizer,              │
  │   generation_config              │
  │ )                                │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │ 토큰 파싱 (양방향)                │
  │ - OUTPUT → 구조화 dict            │
  │ - INPUT → 구조화 dict             │
  │   (txt_dir 모드에서 필요)         │
  └──────────────────────────────────┘
                  │
                  ▼
  ┌──────────────────────────────────┐
  │ 결과 저장                         │
  │ - 토큰 텍스트 (.txt)              │
  │ - JSON 정보 (.json)               │
  │ - 시각화 이미지 (.png)            │
  │ - 메타데이터 (meta.json)          │
  └──────────────────────────────────┘
```

### 입력 모드별 처리

#### 1. JSONL 파일 모드 (`jsonl_file`)

단일 JSONL 파일에서 특정 plan_id들을 읽는다.

```python
# config/inference/pipeline.yaml
input:
  mode: "jsonl_file"
  jsonl_file: "data/dataset/processed_dataset/rplan/jsonl/floorplans_0000.jsonl"
  plan_ids: null  # null이면 전체, ["fp_00001", "fp_00005"]로 지정 가능
  max_samples: 30
```

**처리 흐름:**
- JSONL 파일 로드 → 모든 레코드 파싱
- `plan_ids` 필터 적용 (지정된 plan_id만 추출)
- `max_samples` 제한 적용
- Arrow 스키마 정규화 수행 (Step 3 동일)

#### 2. JSONL 디렉토리 모드 (`jsonl_dir`)

여러 JSONL 파일(glob 패턴)을 처리한다.

```python
input:
  mode: "jsonl_dir"
  jsonl_dir: "data/dataset/processed_dataset/rplan/jsonl"
  jsonl_pattern: "floorplans_*.jsonl"
```

#### 3. Arrow 데이터셋 모드 (`arrow`)

HuggingFace Arrow 포맷에서 특정 split(train/validation/test)을 읽는다.

```python
input:
  mode: "arrow"
  arrow_dir: "data/dataset/processed_dataset/rplan/arrow"
  arrow_split: "test"
```

#### 4. 텍스트 파일 입력 모드 (`txt_dir`) — 신규

**목적**: 이미 증강 완료된 INPUT 토큰 시퀀스 텍스트를 직접 입력 조건으로 사용한다.

```python
input:
  mode: "txt_dir"
  txt_dir: "data/inference/input_txt"
  txt_pattern: "*.txt"
```

**처리 흐름:**

1. **파일 읽기**: `txt_dir` 내 `.txt` 파일 각각이 하나의 입력 조건
   - 파일명 stem이 `plan_id` 됨 (예: `my_test.txt` → `plan_id="my_test"`)
   - 파일 내용 = `<INPUT> ... <END_INPUT>` 형태의 토큰 텍스트

2. **토크나이징**: 
   ```python
   condition_tokens = tokenizer.encode(raw_sample["token_text"], add_special_tokens=False)
   ```

3. **증강 미적용**: txt_dir 모드는 이미 증강 완료된 입력이므로 AugmentationPipeline 우회

4. **구조화 변환** (중요):
   ```python
   input_sample = parse_input_tokens(condition_tokens, vocab, plan_id)
   # → {"plan_id": "my_test", "rooms": [...], "edges": [...], "front_door": {...}, "spatial": [...]}
   ```
   - INPUT 토큰 텍스트를 역변환하여 구조화 dict 생성
   - 이를 `condition.json`, `floorplan.png` 생성에 사용

**설계 의도:**
- txt_dir 모드는 외부에서 생성한 커스텀 입력 조건을 테스트할 때 유용
- 예: 특정 입력 시나리오를 수동으로 구성 → txt 파일로 저장 → txt_dir 모드로 추론
- 토큰 텍스트 형식이므로 tokenizer/vocab에 의존하지만, 기존 JSONL 모드보다 빠름 (JSON 파싱 불필요)

### INPUT 토큰 파서 (`parse_input_tokens`)

**목적**: INPUT 형식의 토큰 ID 시퀀스를 구조화된 dict로 역변환한다.

**상태머신 설계** (src/inference/output_parser.py):

```
상태:
  IDLE                      → <INPUT> 발견 시 PARSING으로 진행
  PARSING
    ├─ ROOM_SUMMARY 블록    → <ROOM_SUMMARY> ~ <END_ROOM_SUMMARY>
    ├─ FRONT_DOOR 블록      → <FRONT_DOOR> ~ <END_DOOR>
    ├─ ROOM 블록           → <ROOM> ~ <END_ROOM> (좌표 포함)
    ├─ EDGE 블록           → <EDGE> ~ <END_EDGE> (nested <DOOR> 포함)
    └─ SPATIAL 블록        → <SP> ~ <END_SP>
  ✓ END_INPUT 발견 시 종료
```

**구체적인 파싱 예:**

```
입력 토큰 텍스트:
<INPUT><ROOM_SUMMARY><TOTAL>3<TYPE:kitchen><COUNT>1<TYPE:livingroom><COUNT>1<TYPE:bedroom><COUNT>1<END_ROOM_SUMMARY>
<FRONT_DOOR><X:128><Y:32><SEP_DOOR><X:8><Y:2><END_DOOR>
<ROOM><RID:1><TYPE:livingroom><X:100><Y:200><X:100><Y:300><X:200><Y:300><X:200><Y:200><END_ROOM>
<EDGE><RID:1><RID:2><DOOR><X:200><Y:250><SEP_DOOR><X:2><Y:10><END_DOOR><END_EDGE>
<SP><RID:1><RID:2><REL:right><END_SP><END_INPUT>

출력 dict:
{
  "plan_id": "fp_00123",
  "rooms": [
    {"rid": 1, "type": "livingroom", "coords": [100, 200, 100, 300, 200, 300, 200, 200]},
    {"rid": 2, "type": "bedroom", "coords": [...]},
    ...
  ],
  "edges": [
    {"pair": [1, 2], "doors": [{"x": 200, "y": 250, "w": 2, "h": 10}]},
    ...
  ],
  "front_door": {"x": 128, "y": 32, "w": 8, "h": 2},
  "spatial": [[1, 2, "right"], ...]
}
```

**핵심 유틸리티:**
- `_extract_rid(token_text)`: `<RID:N>` 토큰에서 N 추출
- `_extract_rel(token_text)`: `<REL:dir>` 토큰에서 direction 추출

### OUTPUT 토큰 파서 (`parse_output_tokens`)

**목적**: 모델이 생성한 OUTPUT 토큰 시퀀스를 평면도 dict로 역변환한다.

**INPUT과의 차이점:**
- `<RID:N>` 없음 (방을 ID로 구분하지 않음, 나열 순서대로 처리)
- 문(`<DOOR>`)이 독립적으로 나열됨 (EDGE 블록 없음)
- `<ROOM_SUMMARY>` 블록 무시 (시각화 불필요)

**구조:**

```
<OUTPUT>
  <FRONT_DOOR> <X:...> <Y:...> <SEP_DOOR> <X:...> <Y:...> <END_DOOR>
  <ROOM> <TYPE:...> <X:...> <Y:...> ... <END_ROOM>
  <ROOM> <TYPE:...> ...
  ...
  <DOOR> <X:...> <Y:...> <SEP_DOOR> <X:...> <Y:...> <END_DOOR>
  ...
<END_OUTPUT>
```

**반환 형식:**

```python
{
  "plan_id": plan_id,
  "rooms": [{"rid": 0, "type": "...", "coords": [...]}, ...],
  "edges": [],        # OUTPUT에는 엣지가 없으므로 빈 리스트
  "front_door": {...} | None,
  "spatial": []       # OUTPUT에는 spatial 정보가 없으므로 빈 리스트
}
```

**파싱 실패 조건** (return None):
- `<OUTPUT>`/`<END_OUTPUT>` 경계를 찾을 수 없음
- 최소 1개 이상의 유효한 `<ROOM>` 블록이 없음
- 좌표 파싱 중 예외 발생

### 1대N 출력 생성 (`generation.num_outputs`)

**목적**: 동일한 입력 조건으로부터 N개의 서로 다른 출력을 생성한다.

```yaml
generation:
  num_outputs: 3        # 1=단일, N>1=다중
  do_sample: true       # sampling 모드 필수
  temperature: 0.8      # 다양성 제어
  top_p: 0.95
  top_k: 50
```

**동작 원리:**

```python
output_results = []
for i in range(num_outputs):
    generated_ids = generate_floorplan(
        condition_tokens, model, tokenizer, cfg.generation
    )
    parsed = parse_output_tokens(generated_ids, vocab)
    output_results.append((generated_ids, parsed, elapsed_time))
```

**주의사항:**
- `do_sample: false` (greedy) 모드에서는 모든 출력이 동일 (deterministic)
- Sampling 모드에서는 RNG 상태가 각 호출마다 진행되어 다양한 출력 생성
- `temperature` 값이 높을수록 더 창의적(다양)하지만 부정확함
- `temperature=0.8, top_p=0.95`는 다양성과 품질 균형의 권장 설정

### 출력 디렉토리 구조

#### 기본 경로 계산

```python
output_dir = Path(cfg.output.dir) / cfg.model.name / cfg.model.training_stage
# → outputs/inference/Qwen2.5-Coder-7B/sft/

# txt_dir 모드는 추가 서브디렉토리
if is_txt_mode:
    output_dir = output_dir / "txt_input"
# → outputs/inference/Qwen2.5-Coder-7B/sft/txt_input/
```

#### N=1 (단일 출력) 구조

```
outputs/inference/{model.name}/{training_stage}/{plan_id}/
├── input/
│   ├── tokens.txt           # 입력 조건 토큰 텍스트
│   ├── condition.json       # 입력 조건 구조화 dict (txt_dir에서도 생성)
│   └── floorplan.png        # 입력 조건 시각화
├── output/
│   ├── tokens.txt           # 생성된 평면도 토큰 텍스트
│   ├── floorplan.json       # 생성된 평면도 구조화 dict (파싱 실패 시 생략)
│   └── floorplan.png        # 생성된 평면도 시각화 (파싱 실패 시 생략)
└── meta.json
```

#### N>1 (다중 출력) 구조

```
outputs/inference/{model.name}/{training_stage}/{plan_id}/
├── input/
│   ├── tokens.txt
│   ├── condition.json
│   └── floorplan.png
├── output_0/
│   ├── tokens.txt
│   ├── floorplan.json       # 파싱 성공 시만
│   └── floorplan.png        # 파싱 성공 시만
├── output_1/
│   └── ...
├── output_2/
│   └── ...
└── meta.json
```

### 메타데이터 구조 (meta.json)

#### N=1 (단일 출력)

```json
{
  "plan_id": "fp_00123",
  "augmentation": "no augmentation",
  "input_token_count": 412,
  "output_token_count": 287,
  "elapsed_sec": 3.42,
  "parse_success": true,
  "timestamp": "2025-04-16T10:30:45.123456+00:00"
}
```

#### N>1 (다중 출력)

```json
{
  "plan_id": "fp_00123",
  "augmentation": "with augmentation",
  "input_token_count": 412,
  "num_outputs": 3,
  "outputs": [
    {
      "index": 0,
      "output_token_count": 287,
      "elapsed_sec": 3.42,
      "parse_success": true
    },
    {
      "index": 1,
      "output_token_count": 301,
      "elapsed_sec": 3.55,
      "parse_success": false
    },
    {
      "index": 2,
      "output_token_count": 295,
      "elapsed_sec": 3.38,
      "parse_success": true
    }
  ],
  "timestamp": "2025-04-16T10:30:45.123456+00:00"
}
```

### 시각화와 DropState 필터링

txt_dir 모드와 증강 미적용 모드에서도 입력 조건을 시각화하려면, 증강으로 인한 샘플 변형을 반영해야 한다.

`_prepare_sample_for_visualization(sample, drop_state)` 함수가 다음을 수행한다:

#### 1. 좌표 노이즈 적용

```python
if drop_state.noise_room_coords:
    for room in sample["rooms"]:
        if room["rid"] in drop_state.noise_room_coords:
            room["coords"] = drop_state.noise_room_coords[room["rid"]]
```

#### 2. 삭제된 요소 필터링

```python
# 방 전체 삭제
sample["rooms"] = [r for r in sample["rooms"] 
                   if r["rid"] not in drop_state.drop_block]

# 방 타입 삭제 (좌표는 유지, 타입만 "unknown")
for room in sample["rooms"]:
    if room["rid"] in drop_state.drop_type:
        room["type"] = "unknown"

# 방 좌표 삭제 (좌표만 빈 리스트)
for room in sample["rooms"]:
    if room["rid"] in drop_state.drop_coords:
        room["coords"] = []

# 엣지 필터링
sample["edges"] = [e for i, e in enumerate(sample["edges"])
                   if i not in drop_state.drop_edge]

# 문 정보 삭제
for idx, mode in drop_state.drop_door.items():
    if idx < len(sample["edges"]) and mode == "all":
        sample["edges"][idx].pop("door", None)
        sample["edges"][idx].pop("doors", None)

# 현관문 삭제
if drop_state.drop_front_door or drop_state.drop_front_door_coords:
    sample["front_door"] = None

# 공간관계 필터링
sample["spatial"] = [s for i, s in enumerate(sample["spatial"])
                     if i not in drop_state.drop_spatial]
```

#### 3. 키 정규화

```python
# _normalize_row_oriented()가 "doors" → "door"로 변환했으므로 복원
for edge in sample["edges"]:
    if "door" in edge and "doors" not in edge:
        edge["doors"] = edge.pop("door")
```

**결과:** 시각화가 모델이 실제로 받은 입력 조건을 정확히 반영한다.

### 추론 파이프라인 실행

#### 기본 실행

```bash
# 기본값: JSONL 파일, 단일 출력, 증강 적용
uv run python scripts/inference/run_inference.py
```

#### 텍스트 파일 입력 (txt_dir)

```bash
mkdir -p data/inference/input_txt

# 기존 outputs/inference/{plan_id}/input/tokens.txt 내용을 복사
# → data/inference/input_txt/my_test.txt 저장

uv run python scripts/inference/run_inference.py input.mode=txt_dir
```

#### N출력 (sampling 모드)

```bash
# Arrow test split에서 2개 샘플, 각 3개 출력 생성
uv run python scripts/inference/run_inference.py \
    input.mode=arrow input.max_samples=2 \
    generation.num_outputs=3 \
    generation.do_sample=true generation.temperature=0.8

# 출력:
# outputs/inference/Qwen2.5-Coder-7B/sft/{plan_id}/output_0/
#                                                   /output_1/
#                                                   /output_2/
```

#### 다른 모델/학습단계

```bash
# GRPO 모델 (완료 후)
uv run python scripts/inference/run_inference.py model.training_stage=grpo

# DPO 모델
uv run python scripts/inference/run_inference.py model.training_stage=dpo
```

#### 증강 미적용

```bash
uv run python scripts/inference/run_inference.py augmentation.enabled=false
```

#### 특정 plan_id만 처리

```bash
uv run python scripts/inference/run_inference.py \
    'input.plan_ids=[fp_00001,fp_00005]'
```

### 주요 모듈

| 파일 | 역할 |
|------|------|
| `src/inference/condition_builder.py` | 입력 로드 (4가지 모드 + plan_ids 필터) |
| `src/inference/output_parser.py` | INPUT/OUTPUT 토큰 → 구조화 dict 파싱 (상태머신) |
| `src/inference/model_loader.py` | 모델/토크나이저 로드 (4bit NF4, 양자화 역변환) |
| `src/inference/generator.py` | 자동회귀 생성 (generate_floorplan) |
| `src/inference/result_saver.py` | 결과 저장 (N출력 서브디렉토리, meta.json 분기) |
| `scripts/inference/run_inference.py` | Hydra 진입점, 추론 루프, N출력 처리 |
| `config/inference/pipeline.yaml` | 모델, 입력 모드, 증강, 생성 파라미터 설정 |
| `tests/inference/validate_inference.py` | 각 입력 모드 + N출력 통합 검증 |

### 설계 결정 및 트레이드오프

#### 1. txt_dir 모드에서 INPUT 파싱이 필수인 이유

**초기 고려**: txt_dir 모드는 이미 증강이 완료되었으므로 입력 JSON/이미지를 건너뛸 수 있을까?

**결정**: 모든 모드에서 입력 조건을 저장하기로 결정.

**근거**:
- **일관성**: 모든 추론 결과가 동일한 입력/출력 구조를 가짐
- **검증**: 입력 JSON을 보고 모델이 받은 조건이 의도한 대로인지 확인 가능
- **디버깅**: 부정확한 생성 결과의 원인이 입력 조건인지, 모델 생성인지 판별 용이

#### 2. N출력 시 output/ vs output_0, output_1, ... 구분

**설계**: N=1일 때만 `output/`, N>1일 때 `output_0/`, `output_1/`, ...

**이점**:
- 기존 단일 출력 스크립트와 완전 하위호환성
- N=1 경우 간결한 디렉토리 구조
- N>1로 전환해도 기존 파일 경로가 변경되지 않음

#### 3. meta.json 구조 분기 (N=1 vs N>1)

**설계**: N=1은 단순 형식, N>1은 `outputs` 리스트

**이점**:
- 단일 출력 사용자: 간단한 JSON 파싱
- 다중 출력 사용자: 각 출력별 통계를 체계적으로 조회 가능
- 도구/스크립트가 `num_outputs` 필드로 형식 판별 용이

#### 4. DropState 필터링이 입력 시각화에 필요한 이유

**문제**: 증강 적용 후 샘플은 변형되었으나, 시각화는 원본 샘플을 사용하면 모델이 받은 입력과 불일치.

**해결**: `_prepare_sample_for_visualization()`이 drop_state를 반영하여 시각화 샘플을 조정.

**결과**: 입력 시각화가 모델의 실제 입력을 정확히 표현함.

### 성능 특성

| 항목 | 값 |
|------|---|
| 평균 추론 시간/샘플 | ~3.4초 (1개 출력, BF16) |
| 입력 토큰 길이 | 300~500 토큰 |
| 출력 토큰 길이 | 250~350 토큰 |
| 총 시퀀스 (Chat Template 포함) | ~800~900 토큰 |
| 메모리 사용 (4bit NF4) | ~7GB GPU + ~2GB CPU |
| N=3 출력 시 추론 시간 | ~10초 (병렬화 미적용) |

### 검증

```bash
# 전체 추론 파이프라인 검증
uv run python tests/inference/validate_inference.py

# 개별 모드 검증
# - JSONL 파일/디렉토리 로드
# - Arrow split 로드
# - txt_dir 파일 읽기 및 파싱
# - 입력 조건 토큰화 및 reverse parsing
# - OUTPUT 파싱
# - N=1 / N=3 출력 메타데이터 검증
# - 디렉토리 구조 확인
```
