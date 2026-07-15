# Cube benchmark (OGBench) — manipulation topology

Survey 문서: [docs/cube_benchmark_survey.md](../../docs/cube_benchmark_survey.md)

OGBench `cube` 태스크의 **task/precondition topology**를 실제 데이터로 분석. 과제의 robotics 확장 예시(navigation 논의 보조).

## 실행
```bash
PY=/home/jonghoon/miniconda3/envs/topovlm/bin/python
$PY -m analysis.cube_ogbench.analyze_cube_topology --dataset cube-double-play-v0
# 없으면 val split(~30MB) 자동 다운로드. train 쓰려면 --use-train (297MB)
```
설치: `pip install ogbench --no-deps`면 충분(데이터는 파일로 받아 numpy로 직접 분석; MuJoCo/env 불필요). 데이터는 `~/.ogbench/data/`.

## 파일
| 파일 | 역할 |
| --- | --- |
| `analyze_cube_topology.py` | cube-double play 데이터에서 choice-point / stacking / skill-stage 분석 + 그림 |
| `results/cube_branch_candidates.md` | robotics topology 표 (실측) |
| `results/cube_topology.png` | 3-panel: choice point / episode 높이 timeline / stacking 그래프 |
| `results/cube_topology.json` | 전체 통계 |

## cube-double val 결과 (100 play eps × 1001 step)
- **Choice point**: first-move cube0 56 / cube1 44; 둘 다 lift 99/100.
- **Precondition/stacking**: 87% eps에 stack; cube0-on-1 62, cube1-on-0 64 (양쪽 순서 탐색).
- **Skill/stage**: pick→lift(≤0.35m)→place→stack 반복, 평균 9.1% step이 stacked.

## 데이터 레이아웃 (참고)
`qpos[0:14]`=arm, cube i xyz = `qpos[:, 14+i*7 : 14+i*7+3]` (cube당 7 = xyz+quaternion).
출처: `ogbench/relabel_utils.py`. num_cubes: single=1, double=2, triple=3, quadruple=4.

## 정직한 한계
- **play(task-agnostic) 데이터** → 양쪽 stacking 순서가 다 나옴. 특정 goal task는 한 순서 고정.
- 긴 precondition chain은 triple(1GB)/quadruple(1.9GB)에 있음(미다운로드).
- 이미지 관찰 없음(state only).
