# Task 1 — Zero-shot Trajectory Stitching (관찰 기반) 설계

VLM에게 실제 topology task를 주어 **현재 성능을 파악**하기 위한 사전 조사 설계. 코드 구현 전 스펙 고정용.
연관 자료: 겹침 경로 채굴·figure는 [../survey/](../survey/) (`survey/week2/find_r2r_overlaps.py`, 5,453 겹침 쌍).

---

## 0. 연구 질문

> Existing VLM이 **egocentric 관찰(파노라마)만 보고**, 겹치는 두 경로를 조합해 **한 번도 안 본 start_A→goal_B 경로**를 zero-shot으로 찾아내는가? (= 관찰로부터 navigation topology를 이해하는가)

- 새 모델·학습 없음. 기존 VLM에 in-context로 질의.
- **관찰 버전만** 사용 (좌표/텍스트 버전은 순수 기하 계산으로 변질되므로 제외 — 사전 조사는 "관찰로부터의 이해"가 핵심).

---

## 1. Task 프레이밍

- 두 경로 A(start_A→goal_A), B(start_B→goal_B)가 공유 subpath(junction)를 가짐.
- 질문: **"A와 B가 보여주는 것만으로 start_A에서 goal_B로 갈 수 있나? 있으면 어느 경로, 없으면 '불가'."**
- 정답 경로는 **공유 junction에서 A의 앞부분 + B의 뒷부분을 stitch**해야만 나옴 (한 번도 시연 안 된 조합).
- 입력·보기 = **파노라마 뷰 시퀀스** (경로 하나 = 뷰들의 나열). 모델은 좌표를 보지 않는다.
- goal은 **goal_B 지점(node)**으로 정의 — B의 R2R instruction 문장은 주지 않음(언어 shortcut 차단).

### 출력 형식: 4지선다 (MC)
생성형 파싱·hallucination을 피하려고 객관식으로 시작. 보기 = 뷰 시퀀스 4개 + **'불가(None)' 옵션 항상 포함**.
- **Positive 문제**: 정답 = 유효 stitch 경로. 오답 = hard distractor (아래 D1–D4).
- **Negative 문제**: 정답 = '불가'. (아래 N1–N3)

### Zero-shot
0-shot 기본. few-shot은 stitching 요령을 가르쳐 "이해"를 오염시키므로 **별도 측정**(0-shot과 분리 보고).

---

## 2. 데이터셋: R2R 원본 nav-graph (Matterport3D)

VLN-CE(연속 좌표)가 아니라 **원본 R2R nav-graph** 사용.
- 노드 = 파노라마 viewpoint(이산), 엣지 = 직접 이동 가능, 노드에 3D 좌표+층 정보.
- 장점: "공유 node"·"유효 경로(그래프상 인접)"가 **정확** → 정답·채점 깔끔 (연속좌표의 threshold 모호함 없음).
- 각 노드에 **파노라마 존재** → 관찰 버전을 렌더링 없이 구성.
- 재료: `survey/week2`의 겹침 mining(5,453쌍)을 nav-graph로 재매핑해 재사용.
- 무방향 그래프: 두 경로가 **같은 node 공유 → 거의 항상 stitch 가능**. 이 사실이 N/D 기준의 뼈대.

---

## 3. MC 인스턴스 구성

### Positive (정답 = stitch)
junction 경유 stitch가 필요한 쌍만 선별 — **start_A→goal_B 최단경로가 공유 junction을 실제로 지나는** 쌍 (직행이면 stitching 불필요 → 무효).

### Negative control (정답 = '불가')
"공유가 없을 때 억지로 잇는가(false positive)"를 검증.

| 종류 | 기준 (∃ 노드쌍 a∈A, b∈B) | 테스트 |
| --- | --- | --- |
| **N1 (같은 scene, 공유 0)** | 같은 scene, 공유 node 0개, 경로가 공간적으로 분리 | 비슷한 건물인데 안 이어짐 |
| **N2 (다른 scene)** | 다른 건물의 A·B | sanity (쉬운 negative) |
| **★ N3 (near-miss)** | 아래 임계값 | **시각적으론 비슷/가깝지만 graph상 다른 node**를 구분하는가 (핵심 함정) |

### Hard distractor (positive 문제의 오답)
정답과 **표면은 닮았지만 topology 결함 1개**.

| 종류 | 기준 | 테스트 |
| --- | --- | --- |
| **★ D1 (broken seam)** | 공유가 아닌 지점에서 접합 → 이음새 순간이동 (아래 임계값) | 이음새가 **진짜 같은 장소**인지 검증 |
| **D2 (invalid middle)** | 양끝 맞지만 중간에 없는 edge 포함 | 경로 전체 연속성 검증 |
| **D3 (wrong goal)** | 경로 A 그대로 (start_A→goal_A) | 실제 목표(goal_B) 추적 |
| **D4 (reversed B)** | junction 후 B 역방향(→start_B) | 방향·도착점 |

- 한 positive 문제 권장 보기: **정답 + D1 + (D2 or D4) + D3** → 이음새·연속성·목표·방향을 모두 검증해야 정답.
- **Easy distractor 세트**(완전 무관 경로)도 별도로 만들어 난이도 대조(step 3).

---

## 4. N3 · D1 구체 임계값

표기: `d2d` 수평거리, `dz` 높이차(층), `edge` 엣지 존재, `hop` 그래프 최단 hop.
핵심 원리: **"연결된 노드만큼 가까운데 엣지는 없는" 지점** = 최대 함정.

```
N3a (같은 층, 벽 너머):  shared=0,  d2d ≤ 1.5m,  dz ≤ 0.5m,  edge=False,  hop ≥ 4
N3b (층 겹침):           shared=0,  d2d ≤ 1.0m,  dz ≥ 2.0m,  edge=False
D1  (broken seam):       positive(공유 j 존재)에서 seam (a*,b*):
                         a* ≠ b*,  edge=False,  hop ≥ 3,
                         1.5m ≤ d2d ≤ 4.0m,  dz ≤ 0.5m,  d2d 최소 선택
                         가짜경로 = A[start_A…a*] ⊕ B[b*…goal_B]  (양끝 정답과 동일, 이음새만 단절)
```

### 캘리브레이션 (락 걸기 전 필수)
데이터 히스토그램으로 조정:
1. **엣지 길이 분포**(edge=True의 d2d) → median(~2.2m) 확인. N3의 near 임계값을 median 바로 아래로.
2. **비겹침 경로쌍 최근접 거리 분포** → N3 후보 수 확인 (적으면 1.5→2.0m 완화).
3. **D1 seam gap 분포** → 1.5–4.0m 구간 후보 충분한지.
4. **층 분리** → 단층 건물이면 N3b 없음(정상), N3a 집중.

### 품질 필터
- N3: near-miss 근접점이 **정확히 1군데**만 (두 번째 근접점 d2d > 3m).
- D1: seam gap이 진짜 junction(d2d≈0)보다 명확히 나쁘게 (≥1.5m) — 대비 분명.
- N3·D1 모두 **함정 지점 파노라마를 실제로 보여줘야** 시각 판단 가능.

---

## 5. Step 3 — 결과 해석용 control (관찰-only의 핵심)

관찰-only는 정확도 숫자 하나만으로는 "이해 vs shortcut"을 못 가림. 아래 control을 **나란히** 돌려 해석.

| 조건 | 기대(이해했다면) | 답하는 질문 |
| --- | --- | --- |
| Random (4지선다) | 25% | 우연 이상인가 |
| Blind (경로 없이 goal만) | 낮음 | 답이 새는가 (leakage) |
| 정상 stitch (positive) | 높음 | 실제 성능 |
| Negative N1·N3 | '불가' 잘 고름 | false positive 없는가 |
| junction 프레임 가림 (ablation) | **하락** | 올바른 시각 단서를 쓰나 (인과) |
| hard vs easy distractor | hard 유지되면 강함 | 표면이 아니라 연속성 이해? |
| (선택) Human baseline | 상한 | 개선 여지 |

→ 이 표가 결과물. "VLM이 관찰로부터 topology를 이해한다/못한다"를 **방어 가능하게** 만듦.

---

## 6. VLM 선정 (관찰 버전)

멀티모달 필수, 파노라마/다중 이미지 처리 가능해야 함.
- 프런티어 3사 각 1: **GPT-4o/4.1** (OpenAI), **Gemini 2.5 Pro** (Google), **Claude Opus/Sonnet** (Anthropic, vision).
- 오픈(재현성): **Qwen2.5-VL** (장문 이미지에 강함) 또는 InternVL.
- 대조군: **VLN 특화 모델**(NaVILA류) — zero-shot stitching은 못할 공산 → "특화 모델도 조합은 못한다"가 유의미 결과.
- 파일럿 시작: **4개**(프런티어 3 + 오픈 1).

---

## 7. 파일럿 계획

1. **데이터 선별**: R2R nav-graph에서 junction 경유 positive 쌍 + N1/N2/N3 + D1–D4 파생 (positive mining을 뒤집어 생성 + §4 필터·캘리브레이션).
2. **MC 인스턴스 구성**: 각 문제 = 파노라마 뷰 시퀀스 보기 4개(+'불가'). 규모 **60~100 문제** (positive:negative ≈ 2:1, N1/N2/N3 고루).
3. **실행·해석**: 4개 VLM × §5 control/ablation → **control 표** 채우기. baseline(Random/Blind) 대비 위치, N3·ablation·hard/easy로 "진짜 이해인지" 판정.

---

## 8. 주요 pitfall (요약)

- **그래프 탐색 shortcut** → 관찰-only로 완화, ablation으로 인과 확인.
- **언어 shortcut** → instruction 배제, goal은 node.
- **정답 다수·파싱** → 객관식.
- **novelty 통제** → junction 경유 쌍만.
- **context 길이·API 비용** → 핵심 node만 subsample, 파일럿 60~100.
- **파노라마 시점 정합** → 같은 node도 방향 다르면 뷰 다름(관찰 난점, N3/D1도 이 때문에 어려움).
- **few-shot 오염** → 0-shot과 분리.

---

## 9. 열린 결정 / TODO (파일럿 전)

- [ ] R2R nav-graph 연결정보·노드 좌표·파노라마 경로 확보 (원본 R2R connectivity graphs).
- [ ] 겹침 mining을 VLN-CE → nav-graph 노드로 재매핑.
- [ ] §4 임계값 히스토그램 캘리브레이션.
- [ ] 파노라마를 VLM 입력으로 넣는 형식(뷰 개수·해상도·프롬프트 템플릿) 확정.
- [ ] MC '불가' 옵션 문구·채점 규칙 확정.
