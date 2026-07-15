# Week 1 — Dataset Source Survey (TopoVLM topology)

과제 문서: [individual_researcher_topology_survey.md](individual_researcher_topology_survey.md)

이 문서는 Week 1 산출물입니다. 목표는 새 모델/loss가 아니라, 이 repo에서 **실제로 접근 가능한 dataset**을 계열별로 구분하고, 각 dataset의 observation / action / graph 유무 / instruction 유무 / topology 후보를 표로 정리하는 것입니다. 아래 내용은 모두 디스크의 실제 파일 스키마를 열어 확인한 것입니다 (추정 아님).

## 0. 요약: 세 계열 모두 이미 확보됨

과제가 요구한 3계층(VLN 중심 → Habitat ObjectNav 비교 → robotics 확장 1개)이 전부 `/data/topovlm` 에 실물로 존재합니다. 새로 다운로드할 필요가 없습니다.

| 우선순위 | 계열 | Dataset | 디스크 위치 |
| --- | --- | --- | --- |
| 1순위 | VLN | R2R-VLN-CE v1.3 (preprocessed) | `/data/topovlm/r2r_vlnce_v1_3_preprocessed/extracted/R2R_VLNCE_v1-3_preprocessed` |
| 2순위 | Navigation (graph 복원) | HM3D ObjectNav v2 (Habitat) | `/data/topovlm/habitat/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2` |
| 3순위(확장) | Robotics | D4RL Franka Kitchen (compositional v2, Minari) | `/data/topovlm/d4rl_kitchen_compositional_v2/hf_snapshot/kitchen` |

## 1. 메인 비교 표

| Environment / dataset | Observation space | Action space | Graph 있음? | Instruction 있음? | Branching / topology 후보 | 데이터 접근 | Risk / missing info |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **R2R-VLN-CE v1.3** | egocentric RGB-D (VLN-CE 연속 환경) + `instruction_text` | continuous / waypoint (VLN-CE); 원본 R2R은 adjacent viewpoint 선택 + `STOP` | 부분적 — 원본 R2R은 MP3D nav-graph(node=panorama viewpoint). 이 CE 버전은 node id 없이 `reference_path`(연속 3D 좌표열)로 제공 | ✅ `instruction{instruction_text}` | 같은 `scene_id` 위 여러 GT path의 overlap → shared subpath / stitching, junction, doorway | ✅ train 10,819 ep + val_seen/val_unseen/test | node id가 없어 "같은 node 공유"인지 "좌표만 근접"인지 직접 판정 필요 (threshold 결정 이슈) |
| **HM3D ObjectNav v2** | egocentric RGB (640×480, hfov 79) + target `object_category` (예: bed) | Habitat discrete: `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT` (turn_angle 30°) | 명시적 graph 없음 → navigable-space / shortest-path 에서 복원 필요 | ❌ language instruction 없음 (목표는 object category뿐) | 같은 scene 내 여러 shortest-path가 공유하는 doorway/corridor, room entry 분기, target 근처 STOP 결정, wrong-turn recovery | ✅ train + val + val_mini, scene별 content json.gz | 이 v2 episode에는 `shortest_paths: None`, `goals: []` → expert path/goal 좌표를 별도(sim rollout or goals_by_category)로 재구성해야 함 |
| **D4RL Franka Kitchen (mixed/partial/complete v2)** | proprioception + object/achieved-goal state (연속 벡터, no RGB in dataset) | 9-DoF continuous end-effector/joint command | physical map 아님 → object-receptacle / precondition / skill-transition graph | ❌ (task는 완료할 subtask 집합으로 지정) | subtask 순서 = task-stage graph. `tasks_to_complete`: microwave, kettle, bottom burner, light switch → choice-point / valid-invalid option | ✅ HDF5 3 variants (mixed 621 ep / 156k steps 등) | navigation topology와 성격이 다름(physical route X). 곁다리 확장 예시로만 사용. RGB 없음 |

## 2. Dataset별 실측 스키마 노트

### 2.1 R2R-VLN-CE v1.3  (1순위, VLN)

`train/train.json.gz` → `{episodes: [...10819], instruction_vocab}`. episode 키:

```
episode_id, trajectory_id, scene_id (mp3d/<scene>/<scene>.glb),
start_position[x,y,z], start_rotation,
info{geodesic_distance},
goals[{position[x,y,z], radius}],
instruction{instruction_text, ...},   # language O
reference_path[[x,y,z], ...]          # GT 경로 = 3D 좌표 시퀀스
```

- `train_gt.json.gz` 별도 존재 (GT action/path 상세).
- **topology 재료**: 같은 `scene_id`로 episode를 묶으면 여러 `reference_path`가 한 scene 위에 겹침 → shared subpath / stitching / branching 후보 탐색 가능 (Week 2 핵심 입력).
- **핵심 주의**: `reference_path`는 graph node id가 아니라 연속 좌표. "정말 같은 physical subpath인가"를 좌표 근접도로 판정해야 함 → 과제 문서가 강조한 "graph상 같은 node인가 vs 시각적으로만 비슷한가" 구분 이슈가 여기서 발생.

### 2.2 HM3D ObjectNav v2  (2순위, graph 복원 필요)

`.../objectnav_hm3d_v2/val/content/<scene>.json.gz` → scene별 episodes (예: 한 scene에 28개). episode 키:

```
episode_id, scene_id (hm3d_v0.2/<split>/<scene>/<scene>.basis.glb),
start_position, start_rotation,
info{geodesic_distance, euclidean_distance, closest_goal_object_id},
goals: [],            # 이 v2 파일에선 비어있음
start_room: None,
shortest_paths: None, # expert path 미포함 → 재구성 필요
object_category       # 목표 (예: bed)
```

- Observation/Action은 `configs/habitat/pr2l_objectnav.yaml`에서 확인: RGB agent, turn_angle 30°, max 500 steps.
- top-level `goals_by_category`, `category_to_task_category_id`로 goal 좌표를 category 단위로 회수 가능.
- **repo 자산**: `analysis/code/hm3d_*` (notebook + `hm3d_branch_structure.py`)가 이미 HM3D top-down trajectory / branch 구조를 시각화. → Week 2 재사용 후보.

### 2.3 D4RL Franka Kitchen  (3순위, robotics 확장)

`hf_snapshot/kitchen/{complete,mixed,partial}-v2/data/main_data.hdf5` + `metadata.json`.

```
env_spec: FrankaKitchen-v1, max_episode_steps 450,
tasks_to_complete: [microwave, kettle, bottom burner, light switch],
total_episodes(mixed): 621, total_steps: 156560,
action_space / observation_space 명시 (연속 벡터)
```

- topology = physical map이 아니라 **subtask 완료 순서 그래프** (precondition / skill-transition / task-stage).
- `analysis/code/predictive_branching_datasets.py`가 이미 Kitchen achieved-goal state로 branching 진단을 수행 → 참고용.
- **역할**: navigation 논의를 보조하는 확장 예시 1개로만 배치 (과제 성공기준 6).

## 3. 이미 존재하는 repo 분석 자산 (Week 2에서 재사용)

| 자산 | 위치 | 용도 |
| --- | --- | --- |
| R2R + Kitchen branching 진단 | `analysis/code/predictive_branching_datasets.py` | R2R GT path / Kitchen state 로딩 + branching 지표 |
| HM3D branch 구조 | `analysis/code/hm3d_branch_structure.py` | ObjectNav branch 후보 추출 |
| HM3D top-down 시각화 노트북 | `analysis/code/hm3d_01/02/03_*.ipynb`, `hm3d_trajectory_notebook.py` | scene top-down trajectory overlay |
| Scene top-down export | `analysis/code/export_hm3d_scene_topdown_pngs.py` | scene 배경 렌더 |
| R2R overlay 예시 그림 | `docs/assets/navigation_r2r_topdown_overlay_00_17DRP5sb8fy.png` | 과제 문서에 첨부된 overlay 예시 (이 repo 생성물) |

## 4. Source list (초안)

- **R2R / VLN-CE**: Anderson et al., *Vision-and-Language Navigation* (R2R), CVPR 2018; Krantz et al., *Beyond the Nav-Graph (VLN-CE)*, ECCV 2020. 로컬: `r2r_vlnce_v1_3_preprocessed`.
- **HM3D ObjectNav**: Habitat ObjectNav challenge; HM3D-Semantics (Ramakrishnan et al., 2021). 로컬: `habitat/datasets/objectnav/hm3d/v2`. Habitat-Lab 문서.
- **PR2L**: repo 구현 근거 문서 `docs/pr2l_implementation_notes.md` (ObjectNav 기반 prompt-conditioned latent).
- **D4RL Franka Kitchen**: Gupta et al., *Relay Policy Learning*; D4RL (Fu et al., 2020); Minari 포맷. 로컬: `d4rl_kitchen_compositional_v2`. metadata code_permalink 포함.

## 5. Week 1 결론 & Week 2 진입점

1. 세 계열 dataset 확보 완료, observation/action/graph/instruction 축 정리 완료.
2. **Stitching(공유 subpath)에 가장 유망한 것은 R2R-VLN-CE** — 같은 scene에 다수 GT path가 있고 instruction까지 있어 과제의 핵심 질문에 바로 답할 수 있음.
3. **Branching 후보는 HM3D ObjectNav** — 같은 scene 여러 shortest-path의 doorway 공유 / wrong-turn / STOP 결정. 단 expert path 재구성 필요.
4. Kitchen은 choice-point graph 확장 예시로만.
5. Week 2 첫 작업: R2R에서 같은 `scene_id`로 path를 묶어 좌표 근접 기반 shared-subpath 후보를 하나 뽑고, 기존 overlay 코드로 그림화.

### 남은 확인 필요 (open questions)

- R2R CE `reference_path` 좌표 → 같은 node 판정 threshold를 얼마로 둘지.
- HM3D ObjectNav expert shortest-path를 어디서 얻을지 (sim rollout vs 캐시된 `habitat/actions/pr2l_hm3d_objectnav`).
- `habitat/episodes/pr2l_hm3d_objectnav` 캐시가 이미 재구성된 path/action을 담고 있는지 (담고 있으면 재구성 생략 가능).
