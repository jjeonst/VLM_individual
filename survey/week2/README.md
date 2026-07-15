# Week 2 — Navigation examples (stitching & branching)

과제: [docs/individual_researcher_topology_survey.md](../../docs/individual_researcher_topology_survey.md) · Week 1 조사: [docs/week1_dataset_survey.md](../../docs/week1_dataset_survey.md)

Week 2 목표는 navigation 데이터 안에서 **topology가 실제로 보이는 구체적 예시**를 찾아 그림으로 보여주는 것입니다. Week 2에 해당하는 코드는 전부 이 폴더(`analysis/week2/`)에 모읍니다.

## 실행 환경

이 repo의 conda 환경(`topovlm`)에 numpy/matplotlib가 있습니다. 시스템 `python3`가 아니라 아래 인터프리터로 실행하세요.

```bash
PY=/home/jonghoon/miniconda3/envs/topovlm/bin/python
```

데이터 경로는 `TOPOVLM_DATASET_ROOT`(기본 `/data/topovlm`)로 잡혀 있습니다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `find_r2r_overlaps.py` | **Step 1** — R2R-VLN-CE에서 같은 scene 안 겹치는(shared subpath) trajectory 쌍을 채굴해 랭킹 |
| `draw_r2r_stitch.py` | **Step 2** — 고른 쌍 하나로 shared subpath + merge/branch junction + stitched route 그림 생성 (산출물 #2) |
| `objectnav_branching.py` | **Step 5–7 (action-space)** — action 시퀀스에서 fork / STOP / spin 후보 표 + 그림 |
| `objectnav_replay_positions.py` | **Step 5–7 (positions, slurm)** — Habitat sim으로 action replay → per-step 위치 복원 |
| `objectnav_topology_positions.py` | **Step 5–7 (spatial)** — 복원된 위치에서 doorway/bottleneck/wrong-turn 탐지 + top-down 그림 (산출물 #3 완성) |
| `results/` | 스크립트 산출물 (JSON/CSV/PNG/MD). git에 커밋하지 않아도 재생성 가능 |

## 공유 overlap 정의 (Step 1·2 공통)

두 스크립트는 **동일한** 정의를 씁니다 (`find_r2r_overlaps.py`의 `nearest_dists` / `longest_run_under`를 Step 2가 import). 겹침은 **격자가 아니라 좌표 거리**로 판정합니다: 한 경로의 점이 상대 경로와 `match_dist`(기본 0.5 m) 이내면 "같은 복도 위"로 봅니다.

> 왜 격자를 안 쓰나: 고정 격자로 양자화하면 불과 몇 cm 떨어진 두 점이 셀 경계로 갈라져 "다른 셀"이 되고, 연속 공유 구간이 토막나 merge point가 실제보다 뒤로 밀립니다. 좌표 거리 기반은 이 아티팩트가 없습니다.

## Step 1 — R2R 겹치는 경로 쌍 채굴 (`find_r2r_overlaps.py`)

### 무엇을 하나
- R2R GT 경로 로딩: episode 메타(`{split}.json.gz`) + dense 경로(`{split}_gt.json.gz`의 `locations`).
- R2R은 한 물리 경로(`trajectory_id`)에 지시문을 ~3개 붙이므로, 기본적으로 **trajectory_id 기준 중복 제거**.
- 같은 scene 안 모든 경로 쌍에 대해 (거리 기반):
  - `longest_shared_run_m`: 상대 경로와 `match_dist` 이내로 **연속** 붙어있는 최장 구간의 **arc length(m)** (= 문서의 `B→C` shared subpath)
  - `min_divergent_tail_m`: 공유 구간 앞뒤로 A·B가 각자 갖는 4개 tail의 최소 길이 (클수록 깨끗한 `X→B→C→Y`)
  - `same_direction`: 두 경로가 공유 복도를 같은 방향으로 지나는가
  - `both_ends_diverge`: 시작·도착이 둘 다 달라 stitch 시 **새 경로**가 생기는가
- 랭킹 두 가지 (`--rank-by`):
  - `run`(기본) — 가장 강한 overlap(공유 복도 최장). shared subpath가 존재한다는 **증거**로 가장 강함(대신 거의 동일한 경로가 상위에 옴).
  - `balance` — 가장 깨끗한 stitching **그림**용(양쪽 tail이 모두 긴 것 우선).

### 실행
```bash
$PY -m analysis.week2.find_r2r_overlaps --split val_unseen --top 20            # 증거용
$PY -m analysis.week2.find_r2r_overlaps --split val_unseen --rank-by balance   # 그림용
# 옵션: --match-dist 0.5  --min-shared-run 2.0
```

### 산출물 (`results/r2r_overlaps/<split>/`)
- `overlap_pairs.json` — 상위 쌍 상세(+ 두 지시문 원문)
- `overlap_pairs.csv` — 스프레드시트용 flat 표 (run_m, min_tail, dir 포함)
- `summary.json` — 설정, scene별 경로 수, 후보 쌍 통계

### val_unseen 실행 결과 (match_dist=0.5 m, min_run=2.0 m)
- 613 physical trajectories (중복 제거 후), 11 scenes.
- 5,453 candidate pairs, 4,620개가 both-ends diverge.
- `--rank-by run` 최상위는 거의 동일한 경로(overlap≈전체)라 shared subpath **증거**로 강함.

## Step 2 — Stitching figure (`draw_r2r_stitch.py`)

### 무엇을 하나
경로 한 쌍을 받아 top-down(바닥 평면 x–z) 그림 2장:
- **왼쪽**: A·B 두 경로 + 공유 subpath(보라 하이라이트) + **merge junction**(복도 합류)·**branch junction**(goal로 분기).
- **오른쪽**: **stitched route** = A prefix + 공유 복도 + B의 goal-tail → `start_A → goal_B`, 어느 원본도 안 밟은 새 경로.

진행 방향(same/opposite)을 자동 판별합니다. stitched route는 **방향과 무관하게 항상 goal_B(=B[-1])로 끝나도록** B의 goal-tail(공유 구간을 B[j1]에서 벗어나는 쪽)을, 그에 대응하는 A의 junction에 이어붙입니다. same-direction이면 stitched route가 공유 복도를 눈에 보이게 재사용하고, opposite면 junction에서만 이어집니다.

### 실행
```bash
$PY -m analysis.week2.draw_r2r_stitch --episode-a 94 --episode-b 1114   # 현재 headline 예시
# 옵션: --match-dist 0.5
```

### 산출물 (`results/r2r_overlaps/<split>/figures/`)
- `stitch_<scene>_<epA>_<epB>.png` — 그림
- `stitch_<scene>_<epA>_<epB>.json` — junction 좌표, shared-run 인덱스, same_direction, 두 지시문

### Headline 예시 (`QUCTc6BB5sX`, ep 94 ↔ 1114)  — `--rank-by balance` 상위의 same-direction 후보
- shared run: A[18:39] / B[17:35], **same direction**, match≤0.5 m.
- 공유 복도 5.25 m, 네 tail 모두 ~4 m (preA/postA/preB/postB), 시작·도착 모두 다름.
- merge (-5.02, -11.95) → 공유 복도 → branch (0.20, -12.21).
- 지시문 A: *"Walk forward, take a left around the corner and walk all the way to the end of the hallway, stop at the door before the bedroom."*
- 지시문 B: *"Walk straight across the wine cellar and exit the door on the other side. Turn left, enter the door to your left in the corner..."*
- → 두 경로가 가운데 복도를 공유하고 양끝(시작·도착)이 모두 갈라지는 교과서적 `X→B→C→Y`. stitched route `start_A → goal_B`가 공유 복도를 재사용하며 goal_B에서 종료.
- 그림: `results/r2r_overlaps/val_unseen/figures/stitch_QUCTc6BB5sX_94_1114.png`

> 다른 예시로 재생성하려면 `overlap_pairs.csv`(또는 `--rank-by balance` 출력)에서 `both_ends_diverge=yes`, `same_direction=same`, `min_divergent_tail_m` 큰 쌍을 골라 `--episode-a/--episode-b`로 넘기면 됩니다.

## Step 5–7 — ObjectNav branching (`objectnav_branching.py`) → 산출물 #3

### 무엇을 하나 & 왜 action-space인가
Week 1에서 확인했듯 HM3D ObjectNav는 R2R 같은 language route graph가 없고, PR2L 캐시에는 episode당 **expert action 시퀀스**(0=STOP,1=FWD,2=LEFT,3=RIGHT) + RGB + target object만 있고 **world position은 없습니다**(top-down은 Habitat sim replay 필요). 과제 문서가 *"어렵다면 action sequence와 branch 후보를 작은 graph로"* 허용하므로, branch를 **action space**에서 특성화합니다. (sim/training 불필요.)

### 실행
```bash
$PY -m analysis.week2.objectnav_branching --split train
```

### 산출물 (`results/objectnav_branching/<split>/`)
- `branch_candidates.md` — **branch 후보 표**(과제 산출물 #3 형식: Candidate / Observation / Action choices / Why topology / valid·invalid / visualize / evidence)
- `objectnav_branching.png` — 3-panel 그림: (A) start-junction fork, (B) option-transition graph, (C) spawn-orientation spin strip
- `objectnav_branching.json` — 전체 통계 + 예시 episode id

### train 결과 (6000 episodes)
- **Start-junction fork** (junction/observation-conditioned): 같은 scene+object인데 첫 turn이 L/R로 갈리는 그룹 **102개**. top: scene `00757-LVgQNuK8vtv`, sofa, 50 routes = 25L/25R. → 초기 branch가 전체 route를 결정, 올바른 쪽은 target instance 위치(관찰)에 의존.
- **Terminal STOP decision**: 모든 episode가 STOP으로 종료, **961개(16.0%)**가 즉시 STOP(spawn=goal). STOP 위치가 success를 가름.
- **Spawn-orientation spin**: **376개(6.3%)**가 ≥180° 스핀으로 시작, **100%가 step 0**.
- **Inflection junctions**: episode당 평균 **14.86회** FORWARD→TURN.
- Option-transition (from FORWARD): FORWARD 55% · LEFT 21% · RIGHT 21% · STOP 2%.

### action-space의 한계 (그리고 그걸 어떻게 넘었나)
action 시퀀스만으로는 **doorway·bottleneck·wrong-turn을 식별 불가** — 전부 "여러 경로가 같은 물리 지점을 지나는가 / 이전 위치로 되돌아왔는가"라는 **공간(위치) 개념**이기 때문. 실제로 ≥180° 연속 turn run은 **100% step 0**(스폰 스핀)이라 mid-route backtrack = 0개였음. 그래서 아래 position 파이프라인으로 위치를 복원해 이 세 후보를 진짜로 찾음.

## Step 5–7 (positions) — Habitat sim replay로 위치 복원 → 공간 branch

### Phase 1: 위치 복원 (`objectnav_replay_positions.py`, slurm)
PR2L 캐시엔 action만 있으므로, Habitat sim에서 각 episode의 action을 **headless replay**(RGB 센서 없이)해 per-step (x,y,z)를 복원. start pose는 원본 ObjectNav episode(`source_trajectory_id`의 `:N` → episode_id N)에서 join.

**slurm 실행 (중요 — 파일시스템 구조):**
- `bml-dev01`의 `/home`은 **로컬**이라 컴퓨트 노드가 못 봄. `/data`만 `bml-head:/data`로 NFS 공유.
- 따라서 **코드를 `bml-head:/home`(88T 공유 FS)로 rsync한 뒤 `bml-head`에서 sbatch**해야 컴퓨트 노드가 파일을 봄. (bml-dev01에서 직접 sbatch하면 로그도 없이 죽음.)

```bash
# 1) 코드 sync
rsync -a --exclude=__pycache__ --exclude=results \
  analysis/week2/ bml-head:/home/jonghoon/Projects/VLM_individual/analysis/week2/
rsync -a slurm/objectnav_replay.slurm bml-head:/home/jonghoon/Projects/VLM_individual/slurm/
# 2) bml-head에서 제출 (rtx5090 partition; rtx3090 노드는 down이었음)
ssh bml-head 'sbatch /home/jonghoon/Projects/VLM_individual/slurm/objectnav_replay.slurm'
# 3) 완료 후 위치 파일 회수
rsync -a bml-head:/home/.../results/objectnav_branching/positions/ \
  analysis/week2/results/objectnav_branching/positions/
```
결과: **5039 trajectories / 124 scenes, ~280s** (rtx5090). 산출물: `positions/train/<scene>.npz` + `index.jsonl`.

### Phase 2: 공간 branch 탐지 (`objectnav_topology_positions.py`, 로컬)
```bash
$PY -m analysis.week2.objectnav_topology_positions --split train
```
- **Doorway/bottleneck**: scene별 floor(x,z)를 0.5m 격자로, cell당 **통과하는 distinct trajectory 수**를 셈. 많이 몰리는 cell = 병목/문.
- **Wrong-turn/dead-end**: 경로가 ≥2.5m 우회 후 이전 지점 0.75m 이내로 **되돌아옴** = there-and-back backtrack.

산출물 (`results/objectnav_branching/train/`): `spatial_branch_candidates.md`(표), `objectnav_topdown.png`(2-panel top-down), `spatial_branch.json`.

**결과 (진짜 doorway/bottleneck/wrong-turn):**
- Doorway/bottleneck: scene `00440-wPLokgvCnuk`에서 한 cell이 **41/51 routes(80%)** 통과 = 방↔방 연결 병목.
- Wrong-turn/dead-end: **1208 routes(24.0%)**가 되돌아오는 backtrack. top: detour 16.5m 후 0.64m 이내 복귀.
- → action-space에선 0개였던 dead-end를 위치 복원으로 실제 탐지. 과제의 doorway/bottleneck/wrong-turn/stop 4개 후보를 모두 커버.

## 남은 확장 (선택)
- **R2R Step 3 검증**: stitched route가 navmesh connectivity를 보존하는지(현재는 좌표 연속성으로 근사).
