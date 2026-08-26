# TopoVLM — "현재 VLM 표현은 navigation topology를 담는가?" 전체 실험 보고서

> 이 문서 하나로: **왜 이 실험을 하는지(목적) → 각 테스트가 던진 질문 → 방법 → 결과의 각 숫자 뜻 → 해석**까지 이해되도록 정리.
> 대상 코드/데이터: PR2L(frozen Prismatic VLM → 표현 → RL 정책) · HM3D ObjectNav.

---

## 0. 배경 & 연구 논지 (왜 하나)

**PR2L 파이프라인**: `RGB 화면 → Prismatic VLM → 표현(embedding) → RL 정책 → 행동 → navigation.`
정책이 실제로 보고 판단하는 것은 VLM이 뽑은 **표현**이다.

**연구 논지**: *"이 표현이 topology(공간 구조: 어디가 같은 장소인지, 어디서 갈라지는지)를 담으면 정책 능력이 크게 오른다."*
→ 그러려면 먼저 **현재 표현이 topology를 잘 못 담는다**는 것을 보여, 연구의 출발점(문제 제기)을 세운다.

**핵심 시험대 = stitching**: 경로 A(start_A→goal_A)와 B(start_B→goal_B)가 **공유 junction(갈림길)** 을 지날 때, 한 번도 시연 안 된 **start_A→goal_B**에 도달하는 능력. 이게 되려면 표현이 "A와 B가 지나는 그 junction은 **같은 장소**"임을 알아야 한다.

이 논지를 **3단계 질문**으로 쪼갰다:

| 단계 | 수준 | 질문 | 상태 |
| --- | --- | --- | --- |
| **B** | 표현 | 표현이 공유 junction을 "같은 곳"으로 인식하는가? | ✅ 완료 |
| **C-1** | 국소 행동 | 정책이 junction(결정점)에서 실패하는가? | ✅ 완료 |
| **C-2** | 궁극 능력 | 정책이 stitching(novel goal 조합)을 못 하는가? | 📄 계획만 |

---

## 1. Task B — 표현이 공유 junction을 아는가

### 질문
> Prismatic 표현은, **A가 junction을 지날 때의 화면**과 **B가 같은 junction을 지날 때의 화면**(다른 경로·다른 접근방향)을, **다른 장소보다 더 비슷하다**고 인식하는가?

### 목적
stitching의 **전제조건**을 표현 수준에서 검증. 표현이 공유 junction을 못 알아보면, 이후 어떤 정책도 그 junction을 활용할 수 없다.

### 방법
HM3D ObjectNav에서 공유 subpath를 갖는 stitch 쌍 24개. 각 쌍마다:
- **positive**(같은 장소) = 유사도(A@junction, B@junction)
- **negative**(다른 장소) = 유사도(A@junction, B@start), (A@junction, B@goal)
- 표현이 topology를 알면 positive > negative. **AUC**(같은 곳을 다른 곳보다 높게 매기는 확률)로 측정.
- **두 표현**으로 교차검증: ① last-token hidden state, ② **정책이 실제 먹는 표현**(visual token 2층→4×4 pool→PCA 1024). **raw-pixel**(32×32 회색 코사인)을 바닥선으로.

### 결과

| 지표 | last-token | **정책 표현(PR2L 정확)** | raw-pixel |
| --- | --- | --- | --- |
| 같은 장소 유사도 | 0.961 | 0.359 | 0.114 |
| 다른 장소 유사도 | 0.929 | 0.071 | 0.036 |
| **AUC** | 0.743 | **0.758** | 0.627 |
| junction 최상위 비율 | 62.5% | **62.5%** | — |
| 같은 장소 유사도 범위 | 0.82~0.999 | −0.112~0.969 | — |

### 해석
- **[정정]** 1차의 "표현이 다 0.93~0.96으로 포화" 비판은 **last-token 특성**일 뿐. 실제 정책 표현은 잘 퍼져 있다(0.359 vs 0.071).
- 그러나 **핵심 지표(AUC ~0.75, 최상위 62.5%)는 두 표현이 동일** → 결론은 표현 선택에 **강건**.
- **여전히 약함**: AUC 0.758 → **37.5%는 다른 장소를 더 비슷하다고 오판**. 일부 같은-junction 쌍은 **코사인이 음수**(두 경로에서 본 같은 지점을 무관하게 봄).
- **→ 답**: 표현은 일반적 장소 구분은 하지만 **공유 junction 인식이 불안정**. topology 신호가 약하고 노이즈가 크다.

*(상세: `results/B_probe_report.md`)*

---

## 2. Task C-1 — 정책이 junction(결정점)에서 실패하는가

### 질문
> 학습된 BC 정책이 **결정점(방향을 바꾸는 갈림길)** 에서 특히 행동을 틀리는가?

### 목적
B(표현이 약함)가 **행동으로 새어나오는지**를 싸게 확인. 표현→행동 인과의 중간 고리.

### 방법 & 지표 읽는 법 (숫자 뜻)

**세팅**: 행동 = 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT. 정책을 각 node에 돌려 **예측 vs 정답 행동** 비교.

**두 그룹으로 나눔(stratify)**:
- **결정점(inflection)** = 직전과 행동이 **바뀌는** node (예: 직진→좌회전) = 갈림길 판단. (학습 목적함수가 쓰는 정의와 동일.)
- **복도(non-inflection)** = 직전과 **같은** 행동 (예: 계속 직진) = 이어가기.

**두 조건**:
- **bidirectional** = 예측 시 **미래 node까지** 봄(전체 궤적).
- **causal** = 예측 시 **과거만** 봄 → **실제 navigation/rollout과 동일 조건.**

**각 지표 뜻**:
| 지표 | 뜻 |
| --- | --- |
| **policy_accuracy** | 그 그룹에서 정책이 맞힌 비율 |
| **majority_baseline** | "입력 무시하고 그 그룹 최빈 행동만 always 찍기"의 정확도 = 멍청한 기준선 |
| **policy − baseline** | 멍청한 기준선보다 얼마나 나은가(양수↑ = 입력을 실제로 씀) |
| **pct_predicted_FORWARD** | 결정점에서 정책이 FORWARD를 출력한 비율(실제보다 높으면 "직진 과다") |
| **recall (LEFT/RIGHT)** | 정답이 LEFT인 node 중 정책이 LEFT라 맞힌 비율(=회전을 실제로 실행한 비율) |

### 결과 (6000 에피소드 / 71,060 node)

| 지표 | bidirectional | **causal (실제 조건)** |
| --- | --- | --- |
| 결정점 정확도 | 0.994 | **0.776** |
| 결정점 멍청한 기준선 | 0.476 (FORWARD) | 0.476 |
| 결정점 − 기준선 | +0.518 | +0.301 |
| 복도 정확도 | 0.960 | 0.821 |
| 복도 − 기준선 | +0.121 | −0.018 |
| **복도 − 결정 격차** | −0.033 (결정이 더 쉬움) | **+0.045 (결정이 더 어려움)** |
| 결정점 FORWARD 출력률 | 47.5% (실제 47.6%와 일치) | **56.6% (직진 과다)** |
| **LEFT recall** (→FORWARD) | 99.4% (0.6%) | **68.0% (27.7%)** |
| **RIGHT recall** (→FORWARD) | 99.5% (0.5%) | **62.1% (31.9%)** |

**읽는 예시**: causal에서 **정답이 LEFT인 8,293개 node 중 정책은 68%만 LEFT라 맞히고, 27.7%(2,299개)를 FORWARD로 틀렸다** → 즉 갈림길에서 **회전해야 하는데 직진으로 뭉갠다.**

### 해석
1. **offline(bidirectional)은 문제를 숨긴다**: 결정점 99.4%는 정책이 **미래를 훔쳐보고 + 학습 데이터를 재현**하기 때문. 한 궤적을 그대로 따라가는 데는 topology가 필요 없어 자명하게 쉽다 → **offline 정확도로는 논지를 검출 불가.**
2. **causal(실제 조건)에서 junction 실패가 드러난다**: 과거만 주면 결정점 정확도가 복도보다 낮고, **회전의 ~30%를 직진으로 놓친다.** "junction에서 헤맨다"는 행동 신호.
3. **연결**: B(표현이 junction을 약하게만 인식) → **C-1 causal(행동이 junction에서 회전을 놓침).** 표현→행동 고리의 1차 증거.

### 한계 (정직하게)
- **원인 미분리**: causal에서 회전이 어려운 것은 (i) 표현의 topology 결함 때문일 수도, (ii) **과거만으로 미래 회전을 예측하는 본질적 난이도**(어떤 정책이든 발생) 때문일 수도. C-1만으로는 못 가른다.
- **학습 데이터·단일 goal**: held-out이 아니고 단일 궤적·단일 goal → **진짜 stitching은 아님.** 확정은 C-2.

*(상세 수치: `results/taskC1_junction_bc.json`, `results/taskC1_junction_bc_causal.json`)*

---

## 3. 종합 해석 — 지금까지 말할 수 있는 것

- **B**: 표현 수준에서 공유 junction 인식이 **불안정**(AUC~0.76, 일부 음의 상관).
- **C-1**: 그 약함이 **rollout 조건의 행동으로 이어져**, 결정점에서 **회전의 ~30%를 직진으로 실패**.
- **두 결과의 정합성**: "다른 경로의 같은 junction을 표현이 잘 못 묶는다"(B) ↔ "갈림길에서 올바른 분기를 못 고르고 직진 default"(C-1 causal).
- **중요한 방법론적 교훈**: **평가 조건이 결론을 가른다.** 미래를 보는 offline 지표(99.4%)로는 topology 결함이 안 보이고, 과거만 주는 실제 조건에서만 드러난다. → **closed-loop rollout(C-2)이 진짜 테스트인 이유.**

---

## 4. 아직 안 한 것 (질문·목적만)

### C (multi-object 재선별) — 데이터 정비
- **목적**: 지금 데이터는 scene당 1물체로 붕괴돼 있어(원본은 4~6물체) 발산-goal stitching이 어렵다. 원본에서 **scene당 ≥2물체**를 재선별해 깨끗한 stitching 데이터를 만든다.
- **최대 게이트(G0)**: 재선별이 실제로 scene당 ≥2물체를 내는지 검증. *(계획: `taskC_stitching_prep.md`)*

### C-2 (stitching rollout) — 궁극 증명
- **질문**: AB만 배운 정책이, 한 번도 안 배운 **start_A→goal_B(다른 물체)** 에 Habitat에서 **실제로 도달**하는가?
- **목적**: "표현에 topology 없음 → stitching 능력 실패"를 **직접** 증명. closed-loop(에러 누적)라 표현 결함이 가장 가혹하게 드러남.
- **필요 3요건**: (a) goal-조건 온라인 인코딩+증분 추론, (b) 정책-인-더-루프 rollout harness, (c) 누수 없는 (start_A,goal_B) 분할 + SR/SPL 판정. 인프라 대부분 재사용 가능. *(계획: `taskC2_rollout_plan.md`)*

---

## 5. 다음 단계 (권장 순서)
1. **C multi-object 재선별** → 발산-goal 데이터 확보(G0 통과).
2. **C-1 clean**: 재선별 후 **goal-swap 반사실**(같은 junction·같은 과거, goal만 A→B로 바꿔 정책이 분기를 바꾸는지)로 "표현 결함 vs 본질적 causal 난이도" 분리.
3. **C-2 rollout**: novel start_A→goal_B 실제 도달률(SR/SPL)로 stitching 능력 직접 측정. 비교군 ①in-dist ②stitch ③follower 상한.

---

## 부록 — 파일 & 재현

| 산출물 | 경로 |
| --- | --- |
| B 표현 probe 보고서 | `vlm_topology_test/results/B_probe_report.md` |
| B 결과(JSON) | `results/pr2l_stitch_probe.json`, `results/pr2l_stitch_probe_exact.json` |
| C-1 결과(JSON) | `results/taskC1_junction_bc.json`, `results/taskC1_junction_bc_causal.json` |
| C 데이터 정비 계획 | `vlm_topology_test/taskC_stitching_prep.md` |
| C-2 실행 계획 | `vlm_topology_test/taskC2_rollout_plan.md` |
| 코드 | `phase1_pr2l_stitch_probe(_exact).py`, `taskC1_junction_bc.py` |

```bash
# 모두 slurm(bml-head 제출). B/exact=rtx4090(7B VLM), C-1=rtx2080(경량).
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/pr2l_stitch_probe_exact.slurm'  # B 정책표현
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/taskC1_junction_bc.slurm'       # C-1 bidir+causal
```
