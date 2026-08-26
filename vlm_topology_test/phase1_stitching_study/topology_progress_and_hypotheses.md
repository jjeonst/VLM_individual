# TopoVLM — 전체 진행 결과 & 가설 종합

> "현재 PR2L 표현(frozen VLM)은 navigation topology를 처리하는가?"
> 이 문서 = **지금까지의 모든 실측 결과 + 근본원인 분석 + 논문 대비 + 핵심 가설/다음 실험**.
> 표기: **[확정]** = 실측 데이터 · **[해석]** = 근거 있는 추론 · **[가설]** = 검증 필요.

---

## 0. 연구 논지 (한 줄)

PR2L: `RGB → frozen VLM → 표현 → RL/BC 정책 → 행동 → navigation`.
논지: *"표현이 topology를 담으면 정책 능력이 오른다"* → 먼저 **현재 표현이 topology를 잘 못 담음**을 보인다.
핵심 시험대 = **stitching**(경로 A·B의 공유 junction을 이어 novel start_A→goal_B 도달).

기존 baseline 구성: HM3D **최단경로(geodesic follower)** 전문가 데이터 · **bidirectional** GraphTransformerPolicy BC · **offline 평가만**(closed-loop 미측정).

---

## 1. B — 표현이 공유 junction을 인식하는가 (표현 수준, 교란 없음)

**질문**: A가 junction 지날 때 프레임 vs B가 같은 junction 지날 때 프레임(다른 경로), 다른 장소보다 더 비슷하다고 보는가?

**[확정] 결과** (HM3D 24 stitch 쌍):

| 지표 | last-token | **정책 표현(PR2L 정확)** | raw-pixel |
| --- | --- | --- | --- |
| 같은 장소 유사도 | 0.961 | 0.359 | 0.114 |
| 다른 장소 유사도 | 0.929 | 0.071 | 0.036 |
| **AUC** | 0.743 | **0.758** | 0.627 |
| junction 최상위 비율 | 62.5% | 62.5% | — |
| 같은 장소 유사도 범위 | 0.82~0.999 | −0.112~0.969 | — |

**[해석]**
- 두 표현 모두 **AUC~0.75, 최상위 62.5%** → 결론 강건. (last-token 포화는 아티팩트, 정정됨)
- **37.5%는 오판, 일부 같은-junction 쌍은 코사인 음수** → **표현이 공유 junction을 불안정하게만 인식.**
- 이건 정책·데이터와 **무관한 직접 증거** → topology 신호가 약함.

---

## 2. C-1 — 정책이 결정점(junction)에서 실패하는가 (행동, offline)

결정점 = inflection(`action[i]≠action[i-1]`, 학습 정의). 평가 = 캐시 그래프에 정책 돌려 per-node 정확도.

**[확정] 결과** (train 6000 에피소드, 71,060 노드):

| | bidirectional 모델·bidir 평가 | bidirectional 모델·**causal 평가** | **causal 재학습 모델·causal 평가** |
| --- | --- | --- | --- |
| 결정점 정확도 | 0.994 | **0.776** | **0.992** |
| 복도 정확도 | 0.960 | 0.821 | 0.952 |
| LEFT recall | 99.4% | 68.0% (→FWD 27.7%) | **99.6%** |
| RIGHT recall | 99.5% | 62.1% (→FWD 31.9%) | **99.4%** |
| 결정점→FWD | 47.5% | 56.6%(과다) | 47.3% |

**[확정] 참고 — held-out offline 일반화**: 기존 bidirectional baseline의 **held-out val action 정확도 = 0.526**, 반면 **majority(무조건 FORWARD) = 0.602** → **held-out에서 무조건 직진보다도 못함**(심각한 과적합/일반화 실패).

**[해석]**
- **미래 훔쳐보기(bidir)는 문제를 숨김**(train 99%). **과거만(causal)** 주면 bidir 모델은 회전 30% 놓침.
- **causal 재학습** → 과거만으로도 **회전 99% 복원** → "always-forward"는 **표현이 아니라 양방향 학습 결함**이 주범이었음.

---

## 3. C-2 — closed-loop rollout (실제 navigation, 이번 세션 최초)

**[확정] 결과** (5 scene × 5 ep = 25 에피소드, success_dist 1.0m):

| | bidirectional 모델 | **causal 모델** | follower(상한) |
| --- | --- | --- | --- |
| **성공률(SR)** | **0.0** | **0.12** | 1.0 |
| SPL | 0 | 0.044 | 1.0 |
| stop_rate | — | 0.88 | 1.0 |
| 종료 시 목표거리 | 3.3m | 4.93m | 0.88m |

**[확정] 진단**: 학습 그래프의 **16%가 "1-node=STOP" 퇴화 샘플** → 정책이 "node 1개=STOP" 학습 → rollout 첫 스텝(항상 1노드)에서 **즉시 STOP**. STOP-masking으로 우회.

**[해석]**
- causal 재학습이 rollout을 **되살림(SR 0→12%)**. 하지만 **12% ≪ follower 100%.**
- 실패 패턴: **목표서 ~5m 앞에서 성급히 STOP** = "도착 여부"를 못 앎.
- 남은 원인은 ①(누수)이 아니라 ②(복구불가)·③(약한 goal/표현)·배포 불일치.

---

## 4. 근본원인 분석 — 왜 정책이 closed-loop을 못 하나

| 원인 | 성격 | 증거 | 상태 |
| --- | --- | --- | --- |
| **① 양방향 학습(미래 누수)** | 설계 결함 | C-1: 99% vs 77% | **[해결]** causal 재학습 |
| **② BC 복구불가(covariate shift)** | 최단경로 데이터 태생 | SPL 0.044, 5m 앞 정지 | 미해결 |
| **③ 약한 goal/topology 표현** | 연구 논지 | B AUC 0.75, 성급한 STOP | 미확정(교란) |
| ④ 직진편향·1-node-STOP 오염 | 데이터 | C-1 직진과다, 16% STOP | 부분우회 |

---

## 5. 논문(2402.02651) 대비 — 왜 논문이 훨씬 높나

**[해석] 영향 큰 순서:**

1. **🔴 데이터 종류**: 논문=**Habitat-Web 인간 시연**(탐색·복구 포함) / 여기=**최단경로**(복구 0). **양은 비슷(7550 vs 6000), 종류가 다름.** → BC covariate shift의 근본.
2. **정책 인과성**: 논문=배포용 causal / 기존 baseline=bidirectional(누수). → **이번 세션에서 수정.**
3. **평가**: 논문=closed-loop SR / 기존=offline 정확도만. → 애초에 비교 대상이 다른 숫자.
4. **[가설] 표현 추출**: 논문=이미지+프롬프트+**생성 텍스트** hidden state(의미적 추론) 가능성 / 여기=**visual token** pool. (원문 재확인 필요)
5. 씬(MP3D vs HM3D)·frame_stride=4 서브샘플 등 부차적. **VLM 백본은 대체로 일치** → 격차 원인 아님.

---

## 6. 핵심 가설 — "topology 처리 실패" vs "데이터 부족" (연구의 심장)

**[가설/논증]**
- 데이터 문제는 **"양 부족"이 아니라 "최단경로(복구 없음)"** 라는 **종류** 문제.
- **중요 논증**: *정말로 목표-상대 topology를 담은 표현이면, 최단경로만으로도 "목표 방향으로 내려가라"라는 매끄러운 정책을 배워 **경로 이탈에도 일반화**(복구가 공짜로 따라옴).*
  → **최단경로로 일반화가 안 됨 = 표현이 깨끗한 목표-상대 topology를 안 준다는 신호.** (연구 논지와 정합)
- **단 교란**: BC covariate shift는 표현이 완벽해도 부분적으로 물림 → **행동 실패 단독으론 "표현 약함"을 확정 못 함.**
- **이미 있는 직접 증거**: B probe(교란 없음)가 junction 인식 AUC 0.75 → 표현 약함을 독립적으로 지지.

**결론**: "현재 표현이 topology를 잘 못 처리한다"는 **B(직접 증거)로 뒷받침**되고 rollout 실패와 **정합**하지만, rollout만으론 **데이터 교란 때문에 확정 불가.**

---

## 7. 결정적 다음 실험 (가설 판정용)

1. **[권장] 데이터 고정 + 표현 교체** — 최단경로 BC를 그대로 두고 **PR2L 표현 vs oracle/강한 표현(예: goal-상대 위치)** 로 정책 학습→rollout SR 비교.
   - oracle 성공 & PR2L 실패 → **표현이 병목 = 논지 확정.**
   - 둘 다 실패 → 최단경로 BC 자체 한계(데이터).
   → **표현을 데이터에서 격리하는 유일한 방법.**
2. **frame_stride 배포 불일치 수정 rollout** (구현완료, 테스트대기) — SR 12%가 과압축 아티팩트였는지 판정.
3. **DAgger**(follower로 정책 방문 상태 라벨) — "복구 데이터 넣으면 되나"(데이터 가설) 검증. (표현 격리는 안 됨)

---

## 8. 산출물 & 재현

| 항목 | 경로 |
| --- | --- |
| B 결과 | `results/pr2l_stitch_probe.json`, `pr2l_stitch_probe_exact.json`, `B_probe_report.md` |
| C-1 결과 | `results/taskC1_junction_bc.json`(bidir), `_causal.json`, `_causalmodel.json` |
| C-2 결과 | `results/taskC2_rollout.json` |
| causal 정책 | `/data/topovlm/checkpoints/pr2l_hm3d_bc_causal/seed_42/model.pt` (train_loss 0.129) |
| 코드 | `phase1_pr2l_stitch_probe(_exact).py`, `taskC1_junction_bc.py`, `taskC2_rollout.py` |
| 정책 causal mask | `policies/graph_policy.py` (causal 플래그) |

```bash
# 전부 slurm(bml-head). 경량=rtx2080, 7B VLM in-loop=rtx4090.
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/train_causal.slurm'          # causal 재학습
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/taskC1_causalmodel.slurm'    # C-1 (causal 모델)
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/taskC2_rollout.slurm'        # C-2 rollout
```

---

## 9. 한 문단 요약

이번 세션은 **closed-loop rollout을 최초로 측정**해 현재 정책이 navigation을 못 함을 확인하고, 원인을 **① 양방향 학습(→causal 재학습으로 해결, SR 0→12%) ② 최단경로 데이터(복구 없음) ③ 약한 표현(B: junction AUC 0.75)** 로 분해했다. 논문 대비 격차의 최대 원인은 **인간 시연 vs 최단경로**(종류)이며, "표현이 topology를 못 처리한다"는 **B로 직접 뒷받침되나 rollout만으론 데이터 교란 탓에 미확정** — 이를 가르려면 **데이터 고정·표현 교체(oracle vs PR2L)** 실험이 결정적이다.
