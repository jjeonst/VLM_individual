# Task C — 행동(BC) stitching 사전준비 · **multi-object 재선별 방식** (미실행)

목적: B(임베딩 probe)가 "PR2L 표현이 공유 junction을 못 담는다"를 보이면, C는 그 **행동적 결과**(정책이 junction/stitch에서 실패)를 보인다.
**결정: C는 multi-object 재선별로 진행.** 이 문서는 설계·작업목록·게이트만 — 아직 실행 안 함.

---

## 0. 검증된 사실 (2026-07-22, 실측)

| 항목 | 값 | 근거 |
| --- | --- | --- |
| **원본 ObjectNav scene당 물체 수** | **4~6종** (각 ~8333 eps) | `datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/train/content/*.json.gz` 8개 scene 실측 |
| **현재 선별(7550)·에피소드(6000) scene당 물체 수** | **1종** (145/124 scene 전부) | `episode_selections/.../train_scene_object_balanced_7550.jsonl`, `episodes/.../manifest.jsonl` |
| 선별 빌더 | `(scene_id, object_category)` 버킷 round-robin | `topovlm_data/habitat_objectnav.py:123` |
| dedup 키 | `source_trajectory_id = scene_id:episode_id` | `habitat_objectnav.py:236` |

→ **핵심 진단**: 원본은 물체가 다양한데 **현재 선별 산출물이 1물체/scene으로 붕괴**돼 있다(구버전 빌더 또는 `max_episodes` cap 추정). 따라서 **재선별이 C의 첫 작업이자 최대 게이트**. 재선별이 실제로 scene당 ≥2물체를 내는지 **검증 통과 전엔 이후 단계로 진행 금지**.

## 1. 핵심 아이디어 (multi-object stitching)

같은 scene에서 서로 다른 물체로 가는 두 궤적:
- 궤적 A: start_A → **bed** (goal_A)
- 궤적 B: start_B → **sofa** (goal_B)
- 둘이 복도(junction) 공유 → **stitch: start_A → junction → sofa(goal_B)** = 한 번도 시연 안 된 novel route.
- bed≠sofa → **목표 발산 확실 + goal 뷰 시각 구분 자연 해결**(다른 물체=다른 뷰, 기존 DINOv2 goal 필터 불필요).

---

## 2. 작업 목록 (순서 = 의존순, 각 단계 뒤 **게이트** 통과해야 다음)

### C0. 재선별 — scene당 ≥2물체 보장  ⟵ **최우선 / 최대 리스크**
- **무엇**: `train.py --mode build_selection`을 **전체 에피소드 이터레이션(no max cap)**으로 재실행. 새 manifest 예: `episode_selections/pr2l_hm3d_objectnav/train_scene_multiobject.jsonl`.
- **파라미터**: `balanced_subset_size`를 scene당 최소 2물체가 뽑히도록 설정(버킷=scene×object이므로 subset이 `2 × #버킷` 이상이면 round당 각 물체 최소 1개). 목표 규모 예: 145 scene × ~5물체 → subset ~2000이면 물체당 ~2~3 eps.
- **왜 지금 것이 붕괴했는지 먼저 확인**: 재실행 전 `objectnav_source_trajectory_id` dedup와 `iter_episodes` cap을 점검(episode_id가 물체별로 유니크한지). 필요 시 dedup 키에 `object_category` 포함하도록 수정.
- **slurm**: CPU/가벼움 → **rtx2080**(또는 CPU 노드). 
- **게이트 G0**: 새 manifest에서 `scenes with ≥2 objects ≥ 30`, `mean objects/scene ≥ 2.5`. (검증 스크립트: scene별 object set 히스토그램 — 0절 실측에 쓴 것 재사용.) **미달 시 빌더 수정 후 재실행.**

### C1. expert 경로 + RGB 생성
- **무엇**: `train.py --mode build_episodes`로 재선별 궤적의 HM3D shortest-path 액션 + 에고 RGB 생성 → `rgb/…/<episode_id>.npy`, `actions/…`.
- **slurm**: habitat-sim, GPU 렌더 → **rtx2080**로 충분(위치 replay와 동급).
- **게이트 G1**: 선별된 모든 episode에 대해 rgb/action 파일 존재 + 길이>MIN_LEN. 누락률 <5%.

### C2. 위치 복원(replay)
- **무엇**: `survey/week2/objectnav_replay_positions.py` 방식으로 per-step (x,y,z) 위치 → `positions/train/<scene>.npz`. (frame index t ↔ position index t 정렬 유지.)
- **slurm**: habitat-sim, **rtx2080**.
- **게이트 G2**: position 스텝 수 == RGB 프레임 수(에피소드별). (0절 정렬검증과 동일 체크.)

### C3. cross-object stitch 쌍 채굴
- **무엇**: `phase1_build_instances.py`의 `find_pairs`를 **objA≠objB 필터**로 확장(이제 원리적으로 존재). 공유 subpath(junction) + 발산 goal 쌍 추출. junction 스텝 인덱스 `tJ_A/tJ_B` 산출.
- **의존**: C2 positions + manifest object 매핑(`obj_of`).
- **게이트 G3**: cross-object 쌍 ≥ 30개 확보, 각 쌍 공유 subpath ≥ MIN_RUN_M, goal 발산 ≥ MIN_DIVERGE.

### C4. VLM 토큰 캐시 (C 학습용)
- **무엇**: `train.py --mode build_cache`로 재선별 궤적의 PR2L 표현(visual token 2층 → 4×4 pool → PCA 1024) graph `.npz` 생성. 기존 6173 single-object 캐시는 **재사용 불가**(다른 선별).
- **slurm**: 7B VLM → **rtx4090/titanrtx(24GB)**. rtx2080 불가.
- **게이트 G4**: 모든 선별 episode에 graph `.npz` 생성, node dim=1024, frame_ranges 정합.

---

## 3. C의 두 갈래 (C3·C4 이후)

### C-1 (가벼움, 권장 1차) — Junction-stratified BC 정확도
- **무엇**: 재선별 데이터로 BC 정책 학습(`--mode train`) → **junction/결정 지점에서 offline action 정확도가 비-junction 대비 급락하는가**를 측정.
- **인과 사슬**: B(표현에 junction 신호 약함) → **C-1(junction에서 행동 실패)**. rollout 불필요, 기존 `evaluation/offline_eval.py`(offline action-accuracy) 재사용.
- **산출**: junction vs non-junction BC accuracy 표 + Δ. 이게 "표현→행동" 인과의 핵심 증거.
- **slurm**: 학습 rtx4090, 평가 rtx2080.

### C-2 (무거움, 궁극 stitching rollout)
- **무엇**: **AB만 배운 정책이 novel start_A→goal_B(다른 물체)에 실제 도달하나** — offline GCRL stitching의 진짜 정의.
- **추가 필요**: (a) **goal-조건부 재구성**(goal=object 조건 또는 goal-image), (b) **Habitat rollout eval harness**(현재 없음 — 신규 구현), (c) held-out stitch 쌍 + 도달 성공 판정(geodesic<threshold).
- **위험**: harness 신규 구현 + rollout 비용 큼. C-1 결과 확인 후 착수 판단.

---

## 4. 게이트 요약 (Go/No-Go)

| 게이트 | 조건 | 실패 시 |
| --- | --- | --- |
| **G0** 재선별 | scene당 ≥2물체(≥30 scene) | 빌더/dedup 수정 후 재실행 — **여기 막히면 multi-object C 전체 보류** |
| G1 episodes | rgb/action 누락<5% | build 로그 점검 |
| G2 replay | position==frame 정합 | replay 스텝 재확인 |
| G3 채굴 | cross-object 쌍 ≥30 | subset 확대(C0 규모↑) |
| G4 캐시 | 전 episode graph .npz | 캐시 재빌드 |

## 5. 현재 자산 재사용성

| 요소 | 상태 |
| --- | --- |
| `train.py` 4모드(selection/episodes/cache/train) | ✅ 재사용 |
| 선별 빌더 round-robin | ⚠️ 1물체 붕괴 — **C0에서 검증/수정 필요** |
| 기존 single-object 캐시(6173)+체크포인트 | ❌ C엔 재사용 불가(참고만) |
| B 정확-재현 probe(`phase1_pr2l_stitch_probe_exact.py`) | ✅ C3 후 divergent-goal로 재실행 |
| offline action-accuracy 평가 | ✅ `evaluation/offline_eval.py` (C-1) |
| rollout harness | ❌ 없음 (C-2에서 신규) |

## 6. 권장 순서
1. **(진행 중) B 정확-재현 결과 확인** — 1차 신호.
2. **C0 재선별 + G0 검증** ← 최우선. 통과 못 하면 나머지 무의미.
3. C1→C2→C3→C4 (게이트 순차).
4. **C-1**(junction-stratified BC) → "표현→행동" 인과 완성 (rollout 없이 여기까지가 현실적 1차 목표).
5. (선택) **C-2** rollout → 궁극 stitching 증명 (harness 신규 구현 필요, 규모 큼).

> 요지: 원본은 scene당 4~6물체라 multi-object stitching이 **원리적으로 성립**하나, **현재 선별 산출물이 1물체/scene으로 붕괴**돼 있다. 따라서 C의 실질 첫 작업은 **재선별이 정말 ≥2물체를 내는지 검증(G0)** 이고, 그게 통과돼야 episodes→replay→채굴→캐시→BC(C-1)로 진행한다.
