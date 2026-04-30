---

name: example-skill # 반드시 부모 폴더 이름(skills명)과 동일해야함
description: 여기에 스킬 설명을 적습니다.

---
이건 클로드 코드에서 skills를 정의하는 방법인데, 왜 api-design.md로 바로 만들지 않고 api-design/SKILL.md 이렇게 만드는거야?간략히 보기오후 7:34디렉토리 구조 설계 의도를 분석했습니다The user is asking about Codex's skill file organization - why use a directory structure like api-design/SKILL.md instead of just api-design.md.완료좋은 질문이에요. 디렉토리 구조를 쓰는 이유는 스킬이 단일 파일로 끝나지 않는 경우가 많기 때문입니다.

예시에서 보면 SKILL.md 안에 이런 참조가 있죠:





비즈니스 로직은 BUSINESS\_RULES.md 참조

데이터 모델은 DATA\_MODELS.md 참조





즉, 실제 구조는 이렇게 됩니다:

.Codex/skills/api-design/

├── SKILL.md            # 메인 진입점 (설명 + description)

├── BUSINESS\_RULES.md   # 비즈니스 로직 규칙

├── DATA\_MODELS.md      # 데이터 모델 정의

└── examples/           # 예제 코드 등

&nbsp;   └── endpoint-example.ts

디렉토리 구조의 장점은:

1\. 관심사 분리 — 하나의 .md 파일에 모든 걸 넣으면 너무 길어집니다. 역할별로 파일을 나누면 관리가 훨씬 수월합니다.

2\. SKILL.md가 진입점 역할 — Codex가 SKILL.md를 먼저 읽고, 필요에 따라 하위 파일을 추가로 참조하는 구조입니다. 일종의 index.html이나 README.md 같은 역할이죠.

3\. 확장성 — 나중에 규칙이 늘어나도 파일만 추가하면 됩니다. 단일 파일이면 계속 비대해지고요.

4\. 컨텍스트 효율성 — Codex가 항상 모든 내용을 로드하지 않고, SKILL.md만 먼저 읽은 뒤 필요한 하위 문서만 선택적으로 읽을 수 있어서 토큰을 절약할 수 있습니다.

반대로 스킬이 정말 간단하다면 api-design.md 한 파일로 만들어도 동작은 합니다. 디렉토리 구조는 스킬이 복잡해질 때를 대비한 확장 가능한 패턴이라고 보시면 됩니다.

