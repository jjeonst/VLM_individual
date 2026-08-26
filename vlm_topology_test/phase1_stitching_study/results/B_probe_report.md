# B — PR2L VLM 표현이 stitching의 공유 junction을 인식하는가?

**한 줄 결론:** PR2L 정책이 실제로 먹는 표현(visual-token 2층 → 4×4 pool → PCA 1024)은 **공유 junction을 불안정하게만 인식**한다 — AUC 0.758, junction 최상위 62.5%(37.5%는 오판), 일부 같은-junction 쌍은 코사인이 **음수**. raw 픽셀(AUC 0.627)보다 낫지만, **다른 경로에서 본 같은 junction을 "같은 결정지점"으로 robust하게 묶지 못함** → 표현의 topology(같은 장소=같은 node) 신호가 약하다.

**두 표현으로 교차검증**했고, **랭킹 결론은 표현 선택에 강건**하다(아래 정정 참고).

실행: slurm 5442(last-token, rtx4090) / **5446(정확-재현, rtx4090)** · 데이터: HM3D ObjectNav 24 stitch 쌍 · 코드: `phase1_pr2l_stitch_probe.py`(last-token), `phase1_pr2l_stitch_probe_exact.py`(정확-재현) · 결과: `pr2l_stitch_probe.json`, `pr2l_stitch_probe_exact.json`

---

## 1. 왜 이걸 묻나 (연구 배경)

PR2L 파이프라인: **RGB → Prismatic VLM → 표현 → RL 정책 → 행동 → navigation.** 정책이 보는 건 VLM의 **표현**.
연구 논지: *"이 표현이 topology 정보를 담으면 정책 능력이 크게 오른다."* → 먼저 **현재 표현이 topology를 잘 못 담는다**를 보여 연구를 시작.

stitching(A·B 경로를 이어 novel start_A→goal_B 도달)이 가능하려면, 표현이 **A와 B가 지나는 공유 junction을 "같은 곳"으로 인식**해야 한다. 그 전제조건을 표현 수준에서 검증한 것이 B.

## 2. 질문 (Task B)

> **표현은 A가 junction을 지날 때의 프레임과, B가 같은 junction을 지날 때의 프레임(다른 접근방향)을, 다른 장소보다 더 "비슷하다"고 인식하는가?** (= 표현이 공유 junction을 아는가)

## 3. 방법

- **데이터**: HM3D ObjectNav 위치 복원본에서 stitch 쌍 24개(같은 scene, 공유 subpath, 발산 start/goal).
- **비교**:
  - **positive** = rep(A@junction) vs rep(B@junction) — 같은 장소, 다른 궤적.
  - **negative** = rep(A@junction) vs rep(B@start), rep(B@goal) — 다른 장소.
- **두 가지 표현으로 각각 측정**:
  1. **last-token**(1차): 마지막 층·마지막 토큰 hidden state(4096d), L2 정규화.
  2. **정확-재현(PR2L 동일)**: `include_generated_text=True` 프롬프트 → **마지막 2층 visual token → 4×4 adaptive-avg pool(16토큰) → bank mean → PCA(8192→1024, 배포된 `pr2l_hm3d_bc`)** → 16토큰 평균 → L2 정규화. **정책 node와 동일 표현.** junction 프레임을 직접 추출(캐시 frame_stride 정렬문제 회피).
- **지표**: junction/비junction 유사도, 분리도, **AUC**(같은 장소를 다른 장소보다 높게 매기는 확률), junction 최상위 비율. **raw-pixel baseline**(32×32 회색 코사인)과 대조.

## 4. 결과

| 지표 | last-token (1차) | **정확-재현 (PR2L 표현)** | raw-pixel baseline |
| --- | --- | --- | --- |
| junction 유사도 (같은 장소) | 0.961 | **0.359** | 0.114 |
| non-junction 유사도 (다른 장소) | 0.929 | **0.071** | 0.036 |
| **분리도 (separation)** | 0.032 | **0.288** | 0.078 |
| **AUC** | 0.743 | **0.758** | 0.627 |
| junction 최상위 비율 | 62.5% (15/24) | **62.5% (15/24)** | — |
| junction 유사도 범위 | 0.82 ~ 0.999 | **−0.112 ~ 0.969** | — |

## 5. 해석 (정정 포함)

1. **[정정] "표현 포화(0.93~0.96)"는 last-token 아티팩트였다.** 실제 PR2L 표현은 잘 퍼져 있다(같은장소 0.359 vs 다른장소 0.071, 범위 −0.11~0.97). 1차 보고서가 "임베딩 공간이 이방적이라 다 비슷하다"고 한 비판은 **정확-재현에서 성립하지 않는다.** (last-token의 anisotropy 특성일 뿐.)
2. **핵심 지표는 두 표현이 사실상 동일**: AUC 0.743→0.758, junction 최상위 62.5%→62.5%. 코사인 스케일이 완전히 바뀌었는데도 **랭킹 품질이 그대로** → **"공유 junction 인식"이라는 결론은 표현 선택에 강건**하다. 아티팩트가 아님.
3. **그럼에도 신호가 약하고 불안정**:
   - AUC 0.758 → **37.5%의 경우 다른 장소를 더 비슷하다고 오판**.
   - junction 유사도 **분산이 큼**(−0.112~0.969): 일부 같은-junction 쌍은 **코사인 음수** = 표현이 두 경로에서 본 같은 지점을 **무관하게** 봄.
   - raw-pixel(0.627) 대비 낫지만, 발산-goal(multi-object) 데이터가 아니라 **N=24 single-object subset**의 한계 위 결과.
4. **행동적 함의**: 표현이 "route A/B의 같은 junction = 같은 결정지점"을 **불안정하게** 인코딩 → 이 표현을 먹는 정책이 공유 junction을 활용한 stitching을 **robust하게 수행하기 어렵다**(→ Task C에서 행동으로 검증).

→ **답**: PR2L 표현은 일반적 장소 구분은 하지만, **다른 접근방향의 같은 junction 인식은 불안정**(AUC~0.76, 1/3 오판, 일부 음의 상관). 표현에 "같은 장소=같은 node"의 topology 신호가 **약하게·노이즈와 함께** 존재한다.

## 6. 한계 & 다음

- **데이터**: single-object subset이라 goal이 수렴하고 N=24로 작음. **multi-object 재선별**(→ `taskC_stitching_prep.md`, 원본은 scene당 4~6물체 확인)로 발산-goal junction을 만들면 더 깨끗하고 규모 확대 가능.
- **표현 교차검증 완료**: last-token / PR2L-정확 두 표현 모두 AUC~0.75, 최상위 62.5% → 결론 강건.
- **다음**: (1) multi-object 재선별, (2) **C-1**(junction에서 BC action 정확도 급락)으로 "표현→행동" 인과, (3) (선택) **C-2** rollout stitching(novel start_A→goal_B 실제 도달).

## 7. 재현
```bash
# last-token
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/pr2l_stitch_probe.slurm'        # -> pr2l_stitch_probe.json
# 정확-재현 (PR2L 동일 표현: visual token 2층 + 4x4 pool + PCA 1024, gen_text=True)
ssh bml-head 'sbatch ~/Projects/VLM_individual/slurm/pr2l_stitch_probe_exact.slurm'  # -> pr2l_stitch_probe_exact.json
```
