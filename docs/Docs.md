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
11. [Step 5: LLM 학습 (구현 예정)](#11-step-5-llm-학습-구현-예정)
12. [Step 6: 추론 및 시각화 (구현 예정)](#12-step-6-추론-및-시각화-구현-예정)

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

③ 생성 결과 시각화 (구현 예정)
   조건 입력 → 모델 추론 → 토큰 시퀀스 → 좌표 복원 → 평면도 시각화
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
│  Step 6: 추론 + 시각화 (구현 예정)                     │
│  조건 입력 → 토큰 시퀀스 → 좌표 복원 → 평면도 이미지    │
└──────────────────────────────────────────────────────┘
```

| Step | 단계 | 입력 | 출력 |
|------|------|------|------|
| 1 | 평면도 정보 추출 | RPLAN PNG (BGRA) | JSONL 샤드 |
| 2 | Vocabulary 빌드 | Pretrained 토크나이저 | vocab_extension.json + 확장 토크나이저 |
| 3 | Arrow 변환 | JSONL | Arrow DatasetDict (train/val/test) |
| 4 | 증강 + 토크나이징 | Arrow + 증강 설정 | (condition_tokens, output_tokens) |
| Pre-Stage | 새 토큰 Embedding 워밍업 | 토큰 시퀀스 배치 | 워밍업된 embed_tokens + lm_head |
| 5 | LLM 학습 (예정) | 토큰 시퀀스 배치 | Fine-tuned 모델 |
| 6 | 추론 + 시각화 (예정) | 조건 텍스트 | 평면도 이미지 |

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
    ├── model.safetensors     ← merge_and_restore 후 표준 HF 형식 저장
    ├── partial_state.pt      ← new_embed / new_lm_head 가중치만 별도 저장
    ├── optimizer.pt          ← AdamW state (new_embed, new_lm_head 두 파라미터만, ~16MB)
    └── trainer_state.json    ← step, epoch, best_model_checkpoint 등
```

체크포인트 저장 흐름:
1. `partial_state.pt` 저장 — merge 전 훈련된 new_embed/new_lm_head 보존
2. `merge_and_restore()` — 표준 HF 형식으로 통합
3. `super()._save_checkpoint()` — 표준 모델 + optimizer + trainer_state 저장
4. `_setup_partial_training()` 재적용 — 훈련 계속을 위해 PartialEmbedding 복원

**Resume 로드 흐름:**
- HF 표준 체크포인트에서 모델 로드 (merge된 embed_tokens.weight 포함)
- `_setup_partial_training()` — 병합된 가중치에서 new_embed 자동 초기화
- `trainer.train(resume_from_checkpoint=...)` → `_load_from_checkpoint` 오버라이드 호출
  1. `super()._load_from_checkpoint()` — transformer 레이어 quantized 가중치 복원 (key mismatch로 embed_tokens/lm_head는 스킵, missing keys 경고는 정상)
  2. `partial_state.pt` 로드 — new_embed/new_lm_head를 훈련된 값으로 직접 복원
- optimizer.pt에서 AdamW state / step 복원

**`_load_best_model` 오버라이드:**
`load_best_model_at_end=true` 시 표준 모델 재로드 대신 `partial_state.pt`에서 new_embed/new_lm_head만 직접 복사한다 (key mismatch 방지).

### 증강 설정 관리 (Hydra config group)

증강 파라미터는 훈련 단계마다 독립된 파일로 관리되며, Hydra **config group** 방식으로 각 파이프라인 yaml에 합성된다.

```
config/training/augmentation/
├── pre_stage.yaml    ← Pre-Stage용 (현재)
├── sft.yaml          ← SFT용 (추후)
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

### 체크포인트 및 출력

```
outputs/training/pre_stage/
└── YYYY-MM-DD/HH-MM-SS/       # Hydra 실행 로그 + 설정 스냅샷

data/models/{model.name}/
└── checkpoints/pre_stage/
    ├── checkpoint-{step}/      # 에폭별 자동 저장 (save_total_limit 초과 시 오래된 것 삭제)
    │   ├── model.safetensors   # 표준 HF 형식 (merge 후)
    │   ├── partial_state.pt    # new_embed / new_lm_head 가중치
    │   ├── optimizer.pt        # AdamW state (~16MB)
    │   └── trainer_state.json
    └── final/                  # 최종 병합 모델 (tokenizer 포함, 표준 HuggingFace 형식)
```

---

## 11. Step 5: LLM 학습 (구현 예정)

### 베이스 모델

Pre-Stage에서 워밍업된 모델을 기반으로 평면도 생성 태스크에 Fine-tuning 한다. 기본 후보 모델은 `Qwen/Qwen2.5-Coder-7B`이나, config의 `model.user`와 `model.name`으로 변경 가능하다.

### 3-Stage Fine-tuning 전략

LLM 학습 시 QDoRA(Quantized DoRA)를 사용한다. 혼합 정밀도(bf16 AMP)를 적용하며, lm_head는 양자화하지 않는다.

#### Stage 1: SFT (Supervised Fine-tuning)

| 설정 | 값 |
|------|---|
| Unfreeze 대상 | DoRA 파라미터 + 새 토큰 embedding + lm_head의 새 토큰 행 |
| LR (Pretrained 파라미터) | 1e-5 |
| LR (새 Embedding) | 2e-4 |
| Warmup | 전체 학습의 5~10% |
| 목적 | 전체 모델이 평면도 생성 태스크에 적응 |

증강된 condition/output 쌍으로 지도 학습. Chat template을 활용하여 user 턴(조건)과 assistant 턴(평면도 출력)으로 구성. DoRA 적용 대상은 config에서 선택 가능하도록 구성 필요. 훈련 후 DoRA 파라미터를 merge하는 기능 필요.

#### Stage 2: DPO (Direct Preference Optimization)

선호/비선호 쌍(preferred/rejected)을 활용하여 생성 품질을 개선한다. 기하학적 제약(방 겹침, 경계 초과 등)을 위반하는 출력을 rejected 샘플로 구성.

#### Stage 3: GRPO (Group Relative Policy Optimization)

RLVR 기반 강화학습으로 규칙 기반 보상 함수를 적용한다. 방 블록 간 겹침 없음, 좌표 경계 준수, 사용자 조건 부합도 등을 보상으로 사용하여 생성 정밀도를 극대화한다.

### 학습 데이터 구성

- **입력:** condition_tokens (삭제 증강이 적용된 부분 정보)
- **출력:** output_tokens (모든 방 + 모든 Edge의 완전한 정보)
- **Loss 마스킹:** 입력 토큰 구간은 loss 무시 (ignore_index=-100), 출력 토큰 구간만 학습

---

## 12. Step 6: 추론 및 시각화 (구현 예정)

### 추론 흐름 (계획)

```
1. 사용자 조건 입력 (자연어 또는 구조화된 조건)
2. 조건을 condition_tokens로 변환
3. 모델 자동회귀 생성
4. 생성된 토큰 시퀀스를 JSONL 구조로 역변환 (decoder.py 활용)
5. 좌표 복원 및 평면도 시각화 (visualize_json 모듈 활용)
```

### 시각화

`visualize_json` 모듈은 JSONL 형식의 평면도 데이터를 OpenCV로 렌더링하는 기능을 이미 구현하고 있으며, 모델 추론 결과 시각화에 재사용될 예정이다.

```bash
# 현재 사용 가능: JSONL 시각화
uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --plan_id 0 1 5
uv run python tests/build_dataset/rplan2json/visualize_jsonl.py --all
```
