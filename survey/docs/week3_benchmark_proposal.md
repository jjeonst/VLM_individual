# Week 3 — Topology benchmark proposal (deliverable #4)

과제: [individual_researcher_topology_survey.md](individual_researcher_topology_survey.md) · Week 1: [week1_dataset_survey.md](week1_dataset_survey.md) · Week 2 코드/결과: [analysis/week2/](../analysis/week2/)

## 이 문서가 답하는 질문 (benchmark-first framing)

> Existing VLM/VLA가 **explicit topology 정보를 받았을 때**, shared subpath, branch point, valid/invalid option, stitchable trajectory를 이해하는가?

새 model을 제안하는 게 아니라, 위를 물어볼 수 있는 **probe question 4개**를 정의합니다. 각 probe는 Week 2에서 실제 데이터로 확보한 예시로 **구성 가능함이 증명**되어 있습니다 (`analysis/week3/build_probe_examples.py`가 실제 인스턴스를 생성).

## 설계 원칙

0. **Decision-making이 자격 요건(gate)이다.** 과제 원칙: *"graph가 있다는 이유만으로 topology라 부르지 말고, 그 graph가 decision-making에 어떤 영향을 주는지 설명하라."* 따라서 probe는 단순히 구조를 **탐지**하는 데 그치지 않고, **그 구조가 선택을 만들고 그 선택이 미래 경로·성공을 바꾸는지**를 물어야 한다. 이 gate를 먼저 통과한 뒤에 아래 원칙(1–3)과 easiest·convincing·least-training으로 순위를 매긴다. (주의: Probe 1 shared-subpath는 단독이면 탐지 과제라 이 gate가 약하다 → Probe 2 stitching 결정과 묶어 보완한다. 아래 "첫 benchmark 추천" 참고.)
1. **Topology를 보되 generic language를 보지 않게 한다.** Input은 자연어 설명이 아니라 **직렬화된 좌표/그래프**(waypoint 시퀀스, cell traffic)로 준다. 정답이 좌표 관계에서만 나오고, instruction 문장의 표면적 유사도로는 못 맞히게 설계한다.
2. **Train-free probe와 needs-eval probe를 구분한다.** 좌표를 텍스트로 주고 물으면 지금 바로 VLM/LLM에 물을 수 있는 것(train-free)과, agent가 실제로 움직여 성공 여부를 봐야 하는 것(policy rollout 필요)을 나눈다.
3. **정답을 데이터에서 기계적으로 얻는다.** correctness criterion은 Week 2 파이프라인이 계산한 gold(공유 구간 index, stitched route, bottleneck cell, dead-end 여부)로 자동 채점 가능해야 한다.

> **중요 개정 (관찰 기반)**: 아래 Probe 1–4는 입력을 **좌표**로 줬는데, 그러면 probe가 순수 기하 계산(겹침 매칭·argmax·splice·loop 검출)으로 변질되어 "topology 이해"가 아니라 "좌표 계산"을 테스트한다. 그래서 **모델에는 agent의 egocentric RGB 관찰만 주고 좌표·위치는 gold 생성용 oracle로만 쓰는** 관찰 기반 버전으로 재설계했다 (아래 "관찰 기반 재설계" 및 첫 benchmark 추천 참고). 좌표 Probe 1–4는 폐기가 아니라 **gold 라벨을 만드는 oracle**로 남는다.

---

## Probe 1 — Shared-subpath detection (R2R)

- **동기**: 두 route가 같은 물리 구간을 공유하는지 인식하는가.
- **Input**: 한 scene의 두 trajectory를 **waypoint 좌표 시퀀스**로 직렬화 (예: `A: [(x0,z0), (x1,z1), ...]`, `B: [...]`). instruction 문장은 주지 않는다(또는 대조군으로만).
- **Expected output**: 공유 subpath가 있는가(yes/no) + 공유 구간의 **index 범위**(A[i0:i1], B[j0:j1]).
- **Correctness**: gold 공유 구간(Week 2 `nearest_dists`/`longest_run_under`로 계산)과 index IoU ≥ 0.5, yes/no 일치.
- **Likely failure mode**: 두 경로의 시작/끝만 비교하거나, 좌표 스케일을 무시하고 "비슷해 보인다"로 답함. instruction을 같이 주면 문장 유사도로 편법.
- **왜 topology?**: 정답은 좌표 근접의 **연속 구간**에서만 나온다. 단어 유사도·상식으론 index 범위를 못 맞힌다.
- **Decision-making gate (주의)**: 이 probe는 단독이면 **탐지(perception)** 과제다 — 공유 구조를 인식할 뿐, 그 구조가 만드는 선택·결과가 없다. 즉 원칙 0(gate)에는 **약하다**. 따라서 첫 benchmark에서는 Probe 2(stitching 결정)와 묶어, 인식한 공유 junction이 "연결 가능/불가"라는 결정과 "연속 경로/끊김"이라는 결과로 이어지게 한다.
- **분류**: **train-free** (직렬화 좌표를 LLM/VLM에 바로 질의).
- **데이터 출처**: R2R `find_r2r_overlaps.py` 후보쌍. 예시: `QUCTc6BB5sX` ep 94↔1114.

## Probe 2 — Trajectory stitching feasibility (R2R)

- **동기**: 공유 junction을 이용해 fragment를 이어 붙여 **connectivity를 보존한 새 route**를 만들 수 있는가.
- **Input**: 두 trajectory A, B(좌표 시퀀스)와 "start_A에서 goal_B로 가는 route를 이어붙일 수 있는가? 가능하면 그 시퀀스를 써라".
- **Expected output**: 가능/불가 + stitched waypoint 시퀀스.
- **Correctness**: (a) 시퀀스가 start_A에서 시작해 goal_B에서 끝나고, (b) 연속한 waypoint 간 거리 ≤ step 임계(끊김 없음=connectivity 보존), (c) 실제 공유 junction을 경유. gold는 Week 2 `draw_r2r_stitch.build_stitch`의 stitched route.
- **Likely failure mode**: 공유 junction 확인 없이 A와 B를 단순 concat → 이음새에서 **순간이동(벽 통과)** 발생.
- **왜 topology?**: 두 경로가 **같은 지점을 지나는지**를 판단해야만 유효한 stitch가 나온다. 문장 조합으론 connectivity를 보존 못 한다.
- **분류**: **train-free** (구성/채점 모두 좌표로 가능). 단 "정말 navmesh상 통행 가능한가"의 엄밀 검증은 sim 필요(→ 확장).
- **데이터 출처**: R2R 헤드라인 stitch 예시(94↔1114) + `overlap_pairs.csv`의 both-ends 후보.

## Probe 3 — Branch validity judgment (ObjectNav)

- **동기**: 한 junction에서 어떤 다음 branch가 goal로 이어지고 어떤 게 dead-end인지 판단하는가.
- **Input**: target object category + 현재까지의 경로 prefix(좌표) + 두 개의 candidate 다음 segment(하나는 goal로 이어지는 실제 expert 연속, 하나는 dead-end excursion). 
- **Expected output**: 각 candidate를 `goal-route` / `dead-end(되돌아와야 함)`로 분류(또는 valid branch 선택).
- **Correctness**: Week 2 spatial 분석의 dead-end 라벨(≥2.5m 우회 후 복귀)과 일치. dead-end=invalid, expert 연속=valid.
- **Likely failure mode**: object 이름 상식("sofa는 거실")으로 찍고, 실제 scene connectivity를 무시.
- **왜 topology?**: 같은 target이라도 정답 branch는 **그 scene의 연결 구조**에 의존한다. 일반 언어 능력으론 판별 불가.
- **분류**: **부분 train-free.** 두 candidate를 주고 라벨을 묻는 것은 train-free. 하지만 "이 branch가 정말 성공으로 이어지는가"를 **새 상황에서 검증**하려면 policy rollout(sim) 필요 → needs-eval.
- **데이터 출처**: `objectnav_topology_positions.py`의 dead-end 예시(1208개) + expert 정답 연속.

## Probe 4 — Bottleneck / doorway identification (ObjectNav)

- **동기**: 여러 route가 공유하는 connectivity chokepoint(문/병목)를 찾는가.
- **Input**: 한 scene의 N개 trajectory(좌표 시퀀스) + "가장 많은 route가 통과하는 지점(cell)은 어디인가?".
- **Expected output**: 병목 지점 좌표(cell 중심 xz).
- **Correctness**: gold bottleneck cell(distinct-trajectory traffic 최대, Week 2 계산)과 거리 ≤ cell 크기(0.5m) 또는 top-3 cell 안.
- **Likely failure mode**: 좌표 평균(무게중심)이나 시각적 중앙을 답함 — 실제 chokepoint가 아님.
- **왜 topology?**: 정답은 **경로들의 교집합이 몰리는 곳**이지 기하 중심이 아니다. traffic 개념을 이해해야 한다.
- **분류**: **train-free** (좌표 주고 질의/채점). 인스턴스 **구성**은 위치 필요(Week 2에서 sim replay로 확보).
- **데이터 출처**: `spatial_branch.json`의 top bottleneck cell (scene `00440`, 41/51 routes=80%).

---

## 요약 표

| # | Probe | Input | Output | Correctness | 왜 topology(≠language) | 분류 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Shared-subpath detection | 두 route 좌표 | 공유 index 범위 | index IoU≥0.5 | 좌표 연속 구간에서만 정답 | train-free |
| 2 | Trajectory stitching | 두 route 좌표 | stitched 시퀀스 | 연속성+endpoints+junction | 같은 junction 판단 필요 | train-free (엄밀검증은 sim) |
| 3 | Branch validity | prefix+2 candidate | goal / dead-end | Week2 dead-end 라벨 | scene 연결구조 의존 | 부분 train-free / needs-eval |
| 4 | Bottleneck ID | N개 route 좌표 | 병목 cell | gold cell 근접 | traffic 집중점=정답 | train-free (구성은 위치 필요) |

## Train-free vs needs-eval 구분 (과제 Week 3 요구)

- **지금 바로 물을 수 있음 (train-free, no rollout)**: Probe 1, 2, 4 + Probe 3의 "라벨 판단" 부분. 좌표를 텍스트/이미지로 직렬화해 기존 VLM/LLM에 질의하고 gold로 자동 채점.
- **나중에 policy/eval 필요 (needs-eval)**: Probe 3의 "이 branch를 실제로 택하면 성공하는가"를 새 episode에서 검증하는 부분, Probe 2의 navmesh 엄밀 통행성 검증. 둘 다 Habitat sim rollout이 필요.

## 관찰 기반 재설계 (rev — 좌표 probe의 한계 보완)

**왜**: 좌표 입력은 probe를 순수 기하 계산으로 만든다 → 모델이 topology를 이해하지 않아도 숫자만 처리하면 풀린다. **모델에는 egocentric RGB 관찰만 주고, 좌표·위치는 gold oracle로만** 쓰면, 서로 다른 시점에서 같은 장소를 인식(place recognition)하고 연결 구조를 목표와 함께 추론해야만 풀린다.

| Obs-Probe | 입력 = agent 관찰 | 질문 → 출력 | 묻는 후보 | topology 종류 | Gold (좌표 oracle) |
| --- | --- | --- | --- | --- | --- |
| 1. Same-connector 인식 | 두 route의 egocentric RGB (+distractor) | 같은 connector인가 / 어느 뷰가 같은 장소 → yes·grouping | R2R shared subpath · ObjectNav doorway | dataset→task (place recog.) | 좌표 cell 근접 = 같은 장소 |
| 2. Continuation match | route A 뷰 시퀀스 + 후보 continuation 뷰 | 어느 continuation이 같은 지점에서 이어지나 → 선택 | R2R stitching junction | task (연결) | junction 프레임 매칭 |
| 3. Branch choice (view+goal) | junction 뷰 + target + L/R 후보 뷰 | 목표로 가는 branch는? dead-end는? → 선택·분류 | ObjectNav dead-end/fork | task (선택→성공) | expert 정답 / dead-end 라벨 |
| 4. Connector-role 인식 | 단일 egocentric 프레임 | 여러 route가 지나는 doorway/bottleneck인가 → 분류 | ObjectNav doorway/bottleneck | dataset→task | high-traffic cell 프레임 |

- **원칙**: 모델은 RGB만 본다. 좌표·위치는 oracle(모델 비공개) → 기하 shortcut 제거.
- **구성 가능**: ObjectNav PR2L RGB 캐시(6,134 세트, (T,480,640,3))가 위치와 정렬 → 실제 프레임으로 생성. 예시: `analysis/week3/build_observation_probe_example.py` → `results/probes/obs_same_place_00440.png` (같은 bottleneck junction을 지나는 3 route의 뷰 + distractor). R2R은 egocentric 렌더링 필요(sim, 확장).
- **정직한 난이도**: 같은 junction도 접근 방향에 따라 뷰가 크게 달라진다 → 그래서 좌표 계산이 아니라 진짜 이해를 요구한다.

## 첫 benchmark 추천 (Week 4 미리보기)

**추천 근거(한 줄)**: 모델에 좌표 대신 egocentric 관찰만 주어 기하·언어 shortcut을 모두 없애고, 목표-조건 branch 선택으로 "선택이 성공을 바꾸는" decision-with-consequence를 물어 진짜 topology 이해를 측정하기 때문이다.

**추천**: 첫 benchmark = **Obs-Probe 3 (Branch choice from view+goal)**, 동반 = **Obs-Probe 1 (Same-connector 인식)**.
- Obs-Probe 3 — junction의 egocentric 뷰 + 목표 물체 + 좌/우 후보 뷰를 주고 목표로 가는 branch(vs dead-end)를 고르게 한다. decision-making을 가장 순수하게 보고(선택→성공), 관찰 기반이라 좌표·언어 shortcut을 모두 차단. 라벨 판단은 train-free, 성공 완전검증은 sim rollout(확장).
- Obs-Probe 1 — 여러 route의 뷰가 같은 physical connector를 지나는지 인식(place recognition = topology의 공유 node). 완전 train-free, 실제 RGB로 바로 구성(위 그림).
- 좌표 기반 Probe 1·2·4는 **폐기가 아니라 gold 라벨을 만드는 oracle**로 남는다.

> 이전 좌표-기반 추천(Probe 1+2)의 한계: 좌표를 주면 겹침 매칭·splice 등 순수 기하 계산으로 환원되어 topology 이해를 측정하지 못한다. 관찰 기반 재설계로 이를 교정했다.

## 구성 가능성 증명

`analysis/week3/build_probe_examples.py`가 위 probe들의 **실제 인스턴스**(prompt + gold answer)를 Week 2 산출물에서 생성 → `analysis/week3/results/probes/`. 즉 이 benchmark는 개념이 아니라 **오늘 데이터로 만들 수 있는 것**이다.
