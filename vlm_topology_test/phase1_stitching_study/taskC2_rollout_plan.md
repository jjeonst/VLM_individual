# Task C-2 — Stitching rollout 실행계획 (novel start_A → goal_B 실제 도달)

목적: **AB만 배운 goal-조건 정책이, 한 번도 시연 안 된 start_A→goal_B(다른 물체)에 Habitat에서 실제로 도달하는가**를 closed-loop rollout으로 측정. offline GCRL/OGBench의 "stitching = 새 목표로의 zero-shot 일반화" 정의를 그대로 구현.

> 이 문서는 **설계·시그니처·slurm·게이트**만 (미실행). C-1은 건너뛰되, 실패 원인 분해용으로 **junction-정확도 진단(§7)** 은 rollout과 함께 뽑는다.

**선행조건**: `taskC_stitching_prep.md`의 **C0~C4(multi-object 재선별→episodes→replay→cross-object 쌍→VLM 캐시)** 완료 + 아래 (a)의 goal-조건 정책 학습. B(정확-재현)로 표현의 약한 junction 신호는 이미 확인됨.

---

## 재사용 자산 (이미 존재 — 확인 완료)

| 자산 | 위치 | C-2에서 역할 |
| --- | --- | --- |
| rollout 루프 골격 | `topovlm_data/hm3d_objectnav_render.py:294` `_rollout_shortest_path_episode` | follower→policy 교체의 원형 |
| goal viewpoint 선택 | 동 `_select_goal_position`, `_goal_position_candidates` | 성공 판정 목표 좌표 |
| geodesic 거리 | `env.sim.geodesic_distance(...)` | DTG·성공 판정 |
| shortest-path follower | 동 `_build_shortest_path_follower` (`goal_radius=cfg.eval.success_distance`) | **상한 baseline** |
| Habitat env 오픈 | 동 `_open_habitat_env`, `_configure_habitat_dataset` | RGB 센서 + pathfinder |
| RGB 프레임/액션 id | 동 `_rgb_frame`, `_action_id` (0=STOP,1=FWD,2=LEFT,3=RIGHT) | 관측·행동 매핑 |
| PR2L 표현 추출+PCA | `encoders/prismatic.py:encode_image_goal_tokens` + `phase1_pr2l_stitch_probe_exact.py:PR2LExactRep` | 루프 내 온라인 인코딩 |
| 정책 | `policies/graph_policy.py:GraphTransformerPolicy` | 행동 예측 |
| eval 설정 | `configs/schema.py` `EvalConfig.success_distance=0.2`, `max_steps`, `episode_timeout_seconds` | rollout 파라미터 |

신규 모듈 3개: `encoders/pr2l_online.py`, `evaluation/rollout_eval.py`, `evaluation/stitch_commands.py`.

---

## 요건 (a) — goal-조건부 정책 (온라인 인코딩 + 증분 추론)

**문제**: `GraphTransformerPolicy`는 goal을 명시 입력으로 안 받음(goal은 VLM 프롬프트로만 조건화). start_A 궤적에 goal_B(sofa)를 명령하려면 표현이 sofa로 조건화돼야 하고, 정책은 자라나는 graph에 대해 인과적으로 추론해야 함.

### (a-1) 온라인 goal-조건 인코더 — `encoders/pr2l_online.py`
`PR2LExactRep`를 rollout용으로 감쌈. **매 스텝 새 RGB를 goal_B 프롬프트로 인코딩 → PCA → (16,1024) node 반환** (정책이 먹는 표현과 동일).
```python
class PR2LOnlineEncoder:
    def __init__(self, weights=WEIGHTS, pca_path=PCA_PATH,
                 include_generated_text=True): ...
    def encode_node(self, rgb: np.ndarray, goal_text: str) -> np.ndarray:
        """RGB(H,W,3) → (16,1024) PR2L node (2층 visual token→4x4 pool→PCA). goal_text=명령 목표."""
```
- **핵심**: `goal_text`는 **명령된 goal_B**(예: "sofa"), 학습 궤적의 goal이 아님. PR2L이 프롬프트로 조건화하므로 이게 goal-조건화의 실체.
- `include_generated_text=True`면 스텝마다 생성 호출 → 느림. **rollout 1차는 False 권장**(속도), 최종은 True로 배포 표현과 일치.

### (a-2) 증분 추론 래퍼 — `evaluation/rollout_eval.py` 내 `OnlinePolicyRunner`
```python
class OnlinePolicyRunner:
    def __init__(self, policy: GraphTransformerPolicy, device, max_positions: int): ...
    def reset(self): self._nodes = []            # 누적 graph
    def act(self, node: np.ndarray) -> int:
        """node(16,1024) append 후, graph[0..t] 전체를 정책에 넣어 마지막 node의 argmax action."""
        # graph_nodes: [1, t+1, 16, 1024], graph_mask: [1, t+1] all-True
        # logits[0, -1] → argmax  (과거만 입력 → 미래 누수 없음)
```
- 정책은 양방향 encoder지만 **과거 프레임만** 넣으므로 안전. 시퀀스 길이 > `max_positions`면 슬라이딩 윈도우.
- 비용: 스텝당 정책 forward는 저렴(VLM 인코딩이 지배적).

### (a-3) goal-조건 정책 **학습**
- multi-object 캐시(C4)로 BC 학습(`train.py --mode train`). 각 에피소드 node는 **자기 goal 프롬프트로 인코딩**돼 있어 goal-조건 신호가 표현에 내재 → 정책이 goal-민감 분기를 학습.
- **(선택) 강한 조건화 상한판**: node에 object-id 임베딩 또는 goal-이미지 DINOv2 특징 concat(구조 변경+재학습). "표현 결함 vs 조건화 부족" 분리용. 1차는 프롬프트-only.

**게이트 A**: 학습 정책이 **in-distribution (start→goal)** rollout에서 SR ≥ 0.4 (정책이 애초에 작동함). 미달이면 C-2 무의미 → 학습/데이터 점검.

---

## 요건 (b) — rollout harness — `evaluation/rollout_eval.py` (신규)

`_rollout_shortest_path_episode`를 정책 버전으로 재작성:
```python
def run_policy_rollout(cfg, env, encoder: PR2LOnlineEncoder,
                       runner: OnlinePolicyRunner, goal_text: str) -> dict:
    obs = env.reset_to(command)          # start_A 포즈 + goal_B goals 주입 (§c)
    runner.reset()
    traj = []
    for t in range(cfg.eval.max_steps):
        rgb = _rgb_frame(cfg, obs)
        node = encoder.encode_node(rgb, goal_text)     # 온라인 goal_B 인코딩
        action = runner.act(node)                       # 증분 추론
        traj.append(env.sim.get_agent_state().position.copy())
        if action == 0:                                 # STOP
            break
        obs = env.step(action)
    dtg = min geodesic(agent_pos, goal_B viewpoints)    # _select_goal_position 재사용
    return {"success": bool(action==0 and dtg <= cfg.eval.success_distance),
            "stopped": action==0, "dtg": dtg, "steps": t+1, "traj": traj}
```
**신규 부분**:
1. **정책-인-더-루프 + 온라인 VLM 인코딩** (매 스텝 Prismatic forward) — **무거움**: 스텝 100~500 × 에피소드 수 × VLM.
2. **합성 명령 주입** `env.reset_to(command)` — §(c)에서 start_A/goal_B 에피소드 구성.
3. **로깅**: success/DTG/steps/traj/SPL.

**구현 선택 (env)**: 재사용 극대화를 위해 `habitat.Env`에 **커스텀 에피소드 데이터셋**(start_A+goal_B)을 주입(`_configure_habitat_dataset` 패턴). 그러면 `geodesic_distance`·goals·follower(baseline)가 그대로 동작. (대안: `habitat_sim` 직접 + `PathFinder.geodesic_distance` — 가볍지만 follower baseline 별도 필요.)

**게이트 B**: 단일 명령 **smoke rollout** 1건이 끝까지 돌고(STOP 또는 max_steps) success/DTG 로그 정상.

---

## 요건 (c) — held-out stitch 명령 + 성공 판정 — `evaluation/stitch_commands.py` (신규)

### (c-1) 명령 집합 = (start_A, goal_B)
- C3의 **cross-object stitch 쌍**(objA≠objB, junction 공유)에서, 각 쌍의 실제 포즈/목표를 원본 에피소드에서 취함:
  - `start_A` = A 에피소드 `start_position`, `start_rotation`.
  - `goal_B` = B 에피소드 `object_category` + `goals[].view_points[].agent_state.position`.
```python
@dataclass
class StitchCommand:
    scene_id: str; start_position; start_rotation
    goal_object: str; goal_view_points: list[np.ndarray]
    provenance: dict   # epA, epB, junction 위치 (누수감사·재현용)

def build_stitch_commands(cross_object_pairs, orig_objectnav) -> list[StitchCommand]: ...
def to_habitat_episodes(commands) -> list[ObjectGoalNavEpisode]:  # env 주입용
```
### (c-2) 누수 차단 (가장 중요한 설계점)
- 원본엔 거의 모든 (start, object) 조합이 있어 **start_A 근처에서 sofa로 가는 에피소드가 train에 들어가면 "novel"이 아님.**
- **분할 규칙**: train 궤적 집합과 각 stitch 명령을 대조해, **(start_A와 근접한 start) → goal_B object** 궤적을 train에서 **제외**. `provenance`로 감사(누수율 리포트).

**게이트 C0**: 명령 ≥ 30개, 각 명령에 대해 train에 직접 궤적 없음(누수 0), goal_B viewpoint 존재.

### (c-3) 지표 & 비교군 (증거의 핵심)
- **지표**: SR(성공률), **SPL**(=성공×최단경로/실제경로), 종료 DTG. 성공 = STOP 시 goal_B viewpoint geodesic ≤ `success_distance`(+엄밀히 물체 가시).
- **필수 비교 4종** (이 대비가 논지 증거):
  | 조건 | 기대 | 의미 |
  | --- | --- | --- |
  | ① in-dist (start→goal) 정책 | 높음 | 정책 작동 확인 (게이트 A) |
  | ② held-out stitch (start_A→goal_B) 정책 | **급락** | **stitching 실패 = 논지** |
  | ③ 같은 명령에 shortest-path follower | 높음 | **상한** — 과제 자체는 가능(실패는 정책/표현 탓) |
  | ④ (선택) 강한-goal(oracle 이미지) 정책 | ②보다↑면 조건화 문제 | 표현 결함 vs 조건화 부족 분리 |

**게이트 C1**: ①·③ 높고 ②가 유의하게 낮음 → "표현에 topology 없음 → stitching 실패" 인과 성립. (②≈③이면 논지 반증 → 재검토.)

---

## slurm 실행 순서 (최저 적합 GPU 준수)

| 단계 | 명령/모듈 | GPU | 산출 |
| --- | --- | --- | --- |
| S0 | C0~C4 (taskC) | rtx2080 / rtx4090(캐시) | multi-object 데이터+VLM 캐시 |
| S1 | `train.py --mode train` (goal-조건 BC) | rtx4090 | 체크포인트 |
| S2 | `stitch_commands.py` (명령+누수감사) | CPU/rtx2080 | 명령 jsonl + 감사 |
| S3 | `rollout_eval.py --smoke 1` | rtx4090 | 게이트 B |
| S4 | `rollout_eval.py` ①②③(④) 전체 | rtx4090 | SR/SPL 표 |

- rollout은 **VLM(7B, bf16≈14GB) 루프 내 + Habitat 렌더** → 24GB 필요 → **rtx4090/titanrtx**. rtx2080 불가.
- 에피소드 병렬: 명령을 shard로 나눠 array job. `episode_timeout_seconds`로 hang 방지.

## 마일스톤 & 게이트 요약
1. **A**: in-dist SR ≥ 0.4 (정책 작동). 미달 → C-2 중단.
2. **B**: smoke rollout 1건 정상 종료.
3. **C0**: 명령 ≥30, 누수 0.
4. **C1**: ②(stitch) ≪ ①·③ → 논지 증거 확보.

## 리스크
- **비용**: VLM-인-더-루프 rollout이 지배적(스텝마다 forward). `include_generated_text=False`·해상도·max_steps로 완화, 규모는 array job.
- **누수(c-2)**: 가장 미묘 — 감사 자동화 필수.
- **조건화 약함**: 프롬프트-only가 너무 약하면 ②가 in-dist에서도 낮아 해석 모호 → ④(강한-goal) 또는 C-1(junction 정확도)로 분해.
- **rollout 에러 누적**: closed-loop라 초반 한 번의 오분기가 실패로 → ③(follower 상한)이 이를 통제.

> 요지: sim·goal·geodesic·follower·표현추출은 **전부 재사용**. 신규는 (a) 온라인 goal 인코더+증분 추론, (b) 정책-인-더-루프 harness+합성명령 주입, (c) 누수 없는 (start_A,goal_B) 분할+SR/SPL 4비교군. C-2는 "표현 topology 결함 → stitching 능력 실패"를 직접 증명하는 최종 실험.
