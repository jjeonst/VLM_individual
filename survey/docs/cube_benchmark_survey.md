# Cube benchmark (OGBench) — topology survey (robotics expansion)

과제: [individual_researcher_topology_survey.md](individual_researcher_topology_survey.md) · 코드/결과: [analysis/cube_ogbench/](../analysis/cube_ogbench/)

과제 문서는 robotics를 navigation 논의를 **보조하는 확장 예시**로 두라고 했고, kitchen/RoboCasa/CALVIN 외 다른 VLA benchmark도 가능하다고 했다. 여기서는 **OGBench의 cube benchmark**(Park et al., *OGBench: Benchmarking Offline Goal-Conditioned RL*)를 실제 데이터로 분석한다. Week 1 kitchen 항목의 확장이다.

## Cube benchmark이 뭔가

로봇 팔이 N개의 큐브를 집어 goal 배치로 pick-and-place 하는 goal-conditioned manipulation. `cube-single`(1) / `cube-double`(2) / `cube-triple`(3) / `cube-quadruple`(4). 여기서 topology는 **물리 지도가 아니라 task 구조**다: 어떤 큐브를 조작할지(choice point), 큐브를 다른 큐브 위에 쌓는 관계(precondition), pick→place→stack skill 순서(task-stage).

## Observation / Action / Topology 축 (Week 1 표 형식)

| 항목 | 내용 |
| --- | --- |
| **Observation** | 37-D state (arm proprioception + 각 cube의 pose). 이 dataset엔 이미지 없음. state layout: `qpos[0:14]`=arm, cube i xyz = `qpos[:, 14+i*7 : 14+i*7+3]` |
| **Action** | 5-D 연속 (end-effector Δxyz + wrist + gripper open/close) |
| **Graph 있음?** | 물리 map 아님 → **object-on-object(stacking) / precondition / skill-transition 그래프**로 나타남 |
| **Instruction 있음?** | ❌ (goal은 목표 cube 배치로 지정; 언어 없음) |
| **데이터** | OGBench `.npz` (observations/actions/terminals/qpos/qvel). val ~30 MB, train 297 MB(double) |

## 분석 결과 (`cube-double-play-v0` val, 실측)

`analyze_cube_topology.py`로 100 play episodes(각 1001 step)를 분석:

- **Choice point (어떤 큐브 먼저)**: first-move cube0 **56** / cube1 **44**, 둘 다 들어올려진 episode **99/100** → 어떤 큐브를 먼저 잡을지가 실제 결정 지점.
- **Precondition / stacking**: **87%** episode에 한 큐브를 다른 큐브 위에 쌓는 이벤트. cube0-on-cube1 **62**, cube1-on-cube0 **64** → play 데이터는 **양쪽 순서를 모두 탐색**.
- **Task-stage / skill**: 한 episode가 pick→lift(최대 0.35 m)→place→(stack)을 여러 번 반복. 평균 **9.1%** step이 stacked 상태.

표와 그림: [analysis/cube_ogbench/results/cube_branch_candidates.md](../analysis/cube_ogbench/results/cube_branch_candidates.md), `cube_topology.png`(3-panel: choice point / episode 높이 timeline / stacking 그래프).

## 왜 이게 topology인가 (그리고 무엇과 구분되나)

- **choice point·precondition·skill order가 미래 결과를 바꾼다**: 밑 큐브를 먼저 놓지 않으면 위 큐브를 쌓을 수 없다(precondition). 어떤 큐브를 먼저 잡느냐가 이후 계획을 결정한다. → "graph가 있다"가 아니라 "선택이 성공을 바꾼다"는 과제의 topology 기준을 만족.
- **정직한 프레이밍**: play(task-agnostic) 데이터라 양쪽 stacking 순서가 다 나온다. 특정 goal-conditioned task는 한 순서를 고정 → 그때 precondition 그래프가 방향성을 가진다. 즉 **dataset은 옵션 공간**을, **task는 그 안의 valid branch**를 고른다.
- **route topology와 구분**: cube의 topology는 R2R/ObjectNav의 물리 경로 topology와 성격이 다른 **task/precondition topology**다. 과제가 강조한 "dataset/task/representation topology를 섞지 마라"에 따라 별도로 둔다.

## Benchmark 연결 (Week 3 probe 확장)

cube는 과제 benchmark probe 목록의 **"Robotics choice-point labeling"**(task 설명 + candidate action → 현재 choice point와 valid/invalid option 구분)에 바로 대응한다. 예:

```
choice point: 어떤 큐브를 먼저 stack 할까
  valid  : 목표에서 base가 되는 큐브를 먼저 놓기
  invalid: 위 큐브를 base 놓기 전에 stack (precondition 위반)
```

이 probe는 cube state(각 큐브 xyz)와 candidate action을 주고 valid/invalid를 묻는 형태로, navigation probe들이 정리된 뒤 **확장**으로 붙이면 된다.

## Risk / 남은 것

- 더 긴 precondition chain(3–4단 탑)은 `cube-triple`(1 GB)/`cube-quadruple`(1.9 GB)에 있음 — 이번엔 double만 받음.
- 이미지 관찰은 이 state dataset에 없음(visual variant는 별도). VLM 실험엔 visual observation이 필요할 수 있음.
- stacking 탐지는 xy<4.5cm·Δz≈cube size 휴리스틱 — 정밀 grasp 라벨은 env replay로 보강 가능.
