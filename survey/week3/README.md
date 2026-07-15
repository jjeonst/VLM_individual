# Week 3 — Topology benchmark instances

설계 문서: [docs/week3_benchmark_proposal.md](../../docs/week3_benchmark_proposal.md)

Week 3의 산출물 #4(benchmark question proposal)를 **실제 인스턴스로 구성 가능함**을 증명하는 코드. Week 2에서 확보한 R2R overlap 쌍과 replay된 ObjectNav 위치에서 (prompt, gold) 쌍을 기계적으로 생성한다.

## 실행
```bash
PY=/home/jonghoon/miniconda3/envs/topovlm/bin/python
$PY -m analysis.week3.build_probe_examples --split val_unseen
# 옵션: --pairs 94:1114 52:1807  (R2R probe 1-2용 episode 쌍)
```

## 파일
| 파일 | 역할 |
| --- | --- |
| `build_probe_examples.py` | 4개 probe의 concrete 인스턴스 생성 |
| `results/probes/*.json` | 생성된 (prompt, gold) 인스턴스 |

## 생성물 (`results/probes/`)
| probe | 파일 | input | gold | 분류 |
| --- | --- | --- | --- | --- |
| 1. Shared-subpath detection | `probe1_shared_subpath.json` | 두 route 좌표 | 공유 index 범위 | train-free |
| 2. Trajectory stitching | `probe2_stitching.json` | 두 route 좌표 | junction + stitched 시퀀스 | train-free |
| 3. Branch validity | `probe3_branch_validity.json` | segment 좌표 | dead-end / goal-route | 부분 train-free |
| 4. Bottleneck id | `probe4_bottleneck.json` | N개 route 좌표 | 병목 cell xz | train-free |

각 인스턴스는 `input`(직렬화 좌표), `question`, `gold`(정답), `correctness_criterion`(자동 채점 규칙), `source`(원본 데이터)를 포함.

## 핵심 설계
- Input은 **instruction 문장이 아니라 좌표**로 직렬화 → generic language가 아니라 topology 이해를 본다.
- Gold는 Week 2 파이프라인이 계산한 값(공유 index, stitched route, bottleneck cell, dead-end 라벨)이라 **자동 채점** 가능.
- 실제 확인: probe2의 gold stitched route는 start_A→goal_B이고 연속 gap ≤ 1.0m로 connectivity 보존.
