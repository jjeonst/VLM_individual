# TopoVLM Topology Survey — Claude 작업 모음

과제([docs/individual_researcher_topology_survey.md](../docs/individual_researcher_topology_survey.md))에 대해 Claude로 진행한 조사·분석·benchmark 제안을 한 폴더에 모았다. Navigation(R2R·HM3D ObjectNav) 중심, robotics(Kitchen·Cube)는 확장.

## 폴더 구조

```
survey/
├── docs/                     최종 문서 (읽는 사람용)
│   ├── TopoVLM_navigation_benchmark_report.docx   ★ 최종 정리 리포트 (6섹션, 그림 4)
│   ├── TopoVLM_topology_deliverables.docx         이전 산출물 종합본 (Week1–3 + Cube)
│   ├── week1_dataset_survey.md                     Week 1 dataset 조사
│   ├── week3_benchmark_proposal.md                 benchmark probe 설계
│   └── cube_benchmark_survey.md                    OGBench cube 조사
├── week2/                    Navigation 분석 코드 + 결과
├── week3/                    Benchmark probe 생성 코드 + 결과
├── cube_ogbench/             Robotics(cube) 분석 코드 + 결과
└── build_scripts/            docx 생성 스크립트
```

## 실행 환경
```bash
PY=/home/jonghoon/miniconda3/envs/topovlm/bin/python   # conda env 'topovlm'
# repo 루트에서 모듈로 실행 (survey 는 패키지):
$PY -m survey.week2.find_r2r_overlaps --split val_unseen
```
데이터: `/data/topovlm`(공유 NFS). 무거운 작업(ObjectNav 위치복원)은 bml-head로 코드 sync 후 rtx5090 slurm.

## 코드 맵

### week2 — Navigation topology (R2R + ObjectNav)
| 파일 | 역할 | 산출물 |
| --- | --- | --- |
| `find_r2r_overlaps.py` | R2R 겹치는 경로쌍 채굴 (거리 기반) | `results/r2r_overlaps/.../overlap_pairs.*` |
| `draw_r2r_stitch.py` | shared subpath + stitched route 그림 | `.../figures/stitch_*.png` |
| `objectnav_branching.py` | ObjectNav action-space branch 분석 | `results/objectnav_branching/train/branch_candidates.md` |
| `objectnav_replay_positions.py` | Habitat sim으로 action→위치 복원 (slurm) | `.../positions/train/*.npz` |
| `objectnav_topology_positions.py` | 위치에서 doorway/bottleneck/dead-end | `.../objectnav_topdown.png` |
| `objectnav_junctions.py` | decision junction(fork) 탐지·시각화 | `.../objectnav_junction.png` |

### week3 — Benchmark probes
| 파일 | 역할 | 산출물 |
| --- | --- | --- |
| `build_probe_examples.py` | 좌표 기반 probe 인스턴스(oracle 겸용) | `results/probes/probe*.json` |
| `build_observation_probe_example.py` | 같은 doorway 관찰 몽타주 | `results/probes/obs_same_place_*.png` |
| `build_junction_obs_example.py` | Obs-Probe 3 (junction 뷰+branch) 몽타주 | `results/probes/obs_branch_choice_*.png` |

### cube_ogbench — Robotics 확장
| 파일 | 역할 | 산출물 |
| --- | --- | --- |
| `analyze_cube_topology.py` | OGBench cube-double manipulation topology | `results/cube_topology.png` + 표 |

## 과제 산출물 ↔ 위치 매핑
| 산출물 | 위치 |
| --- | --- |
| 1. Comparison table | `docs/week1_dataset_survey.md`, 리포트 §2 |
| 2. Stitching figure | `week2/results/r2r_overlaps/val_unseen/figures/stitch_QUCTc6BB5sX_94_1114.png`, 리포트 §3 |
| 3. Branching figure/table | `week2/results/objectnav_branching/train/` (objectnav_topdown.png · objectnav_junction.png), 리포트 §4 |
| 4. Benchmark question | `docs/week3_benchmark_proposal.md` + `week3/results/probes/`, 리포트 §5 |
| 5. Source list | `docs/week1_dataset_survey.md` §Source, 리포트 각주 |
| 부록. Robotics(Cube) | `docs/cube_benchmark_survey.md` + `cube_ogbench/results/` |

## 핵심 결론 (리포트 요약)
- Navigation topology 중 robust한 것은 **공유 구조(bottleneck 80%, junction 다수 수렴)**. 물리적 dead-end는 expert 데이터에서 rare해 포착 어렵고, **junction+목표(‘목표 아닌 branch = 이 goal의 dead-end’)로 대체**한다.
- Probe는 좌표를 주면 기하 계산으로 변질되므로 **관찰(egocentric RGB) 기반**으로 재설계. 유의미한 둘: **Obs-Probe 3 (뷰+목표→branch 선택)**, **Stitching-via-policy (AB만 학습→startA→goalB 도달)**.
- 추천 첫 benchmark: **A) ObjectNav Branch-choice(관찰·경량)** + **B) R2R Stitching-via-policy(needs-eval, stitching 직접 측정)**.

## 문서 재생성
```bash
$PY survey/build_scripts/build_report_docx.py   # → docs/TopoVLM_navigation_benchmark_report.docx
```
(docx는 그림이 내부에 임베드된 자립 파일이라 그대로 열람 가능.)
