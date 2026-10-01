# PR2L Habitat 재현 — 구현 사양

이 문서는 **실제로 무엇이 코드에 들어갔는가**를 기록한다. "무엇을 왜 이렇게 하기로 했는가"는
같은 폴더의 `PLAN.md`에 있다. 단계가 끝날 때마다 여기에 확정값과 실측치를 덧붙인다.

재현 대상은 Chen, Mees, Kumar, Levine, *Vision-Language Models Provide Promptable
Representations for Reinforcement Learning*, [arXiv:2402.02651](https://arxiv.org/abs/2402.02651)
의 Habitat ObjectNav 모방학습 실험(본문 Table 3)이다. 저장소 사본은
`docs/papers/PR2L_arxiv_2402.02651.pdf`.

각 항목은 세 가지 중 하나로 분류한다.

- **[일치]** 논문이 값을 명시했고 그대로 따랐다.
- **[이탈]** 논문과 다르게 했다. 이유를 함께 적는다.
- **[미명시]** 논문에 값이 없어 원 출처를 찾거나 사용자와 합의해 정했다. 근거를 함께 적는다.

---

## 0단계 — 데이터 확보와 부표본 선정

코드: `subsample.py`. CPU만 사용하며 slurm이 필요 없다.
(파일 이름을 `select.py`로 두면 표준 라이브러리의 `select` 모듈을 가려 `huggingface_hub`
임포트가 깨진다. 처음에 이 이름으로 만들었다가 그 오류를 만나 바꾸었다.)

### 0.1 학습 데이터의 출처

**[일치]** 논문은 사람이 직접 조작해 만든 시연을 쓴다. 부록 C.2 첫 문단이 "Habitat-Web human
demonstration dataset of 77k trajectories (12M steps)"라고 밝힌다. 이 데이터는 PIRLNav 저자들이
[huggingface.co/datasets/axel81/pirlnav](https://huggingface.co/datasets/axel81/pirlnav)의
`objectnav_hm3d_hd`("HD" = human demonstrations)로 배포한다. 라이선스는 CC BY-NC 4.0이다.

받은 위치: `/data/topovlm/habitat/sources/pirlnav_hf/objectnav_hm3d_hd/train/content/`
(80개 장면 파일, 86.9 MB).

이 데이터에는 **이미지가 없다.** 시연 하나는 시작 자세와 행동 열로만 저장돼 있고, 사람이 무엇을
보았는지는 그 행동을 시뮬레이터에서 재생해야 복원된다. Habitat의 관찰·행동·동역학이 결정적
이라고 부록 C.1이 밝히고 있으므로, 재생 결과는 원래 화면과 정확히 같다. 그 재생이 1단계다.

받은 것이 논문이 쓴 데이터가 맞는지 다음과 같이 확인했다.

| 항목 | 논문/원출처 | 실측 | 차이 |
|---|---|---|---|
| 총 궤적 | 77 k (PR2L 부록 C.2) | 76,394 | 0.8 % |
| 총 스텝 | 12 M (PR2L 부록 C.2) | 12,156,643 | 1.3 % |
| 평균 궤적 길이 | 159 (VC-1 부록 A.3) | 159.1 | — |
| 학습 장면 | 80 (PR2L 부록 C.1) | 80 | 일치 |
| 목표 물체 | 6종 (PR2L 본문 4.2) | 6종 | 일치 |

논문이 반올림해 적은 값과 소수점 단위까지 맞는다.

### 0.2 부표본 추출 규칙

**[일치]** 부록 C.2 항목 1의 문장을 그대로 구현했다.

> "we used a subset of the dataset, built by dividing the dataset by both target object and
> scene, then sampling every tenth demo. This would ensure that our training data still
> contained examples from every training scene + target object combination that existed."

묶는 단위는 장면 하나도 물체 하나도 아닌 **(장면, 목표 물체) 쌍**이고, "every tenth"는 무작위
추출이 아니라 **고정 간격**이다. 따라서 시드가 없고 몇 번을 돌려도 같은 결과가 나온다. 그룹
안의 순서는 장면 파일에 적힌 순서를 그대로 쓴다.

조합별로 먼저 나누는 이유는 논문이 바로 다음 문장에 적어 두었다. 전체를 대상으로 그냥 10분의
1을 뽑으면 예시가 하나도 남지 않는 조합이 생기고, 그러면 정책은 학습 때 본 적 없는 물체를 그
건물에서 찾으라는 요구를 평가에서 받게 된다. 그래서 **"조합 누락 0"을 보고용 통계가 아니라
통과 조건으로 두었다.** 어긋나면 선정 파일을 쓰지 않고 멈춘다.

### 0.3 부표본 결과

산출물: `/data/topovlm/habitat/episode_selections/pr2l_habitat_web_hd/train_every_tenth.jsonl`

| 항목 | 논문 | 실측 | 판정 |
|---|---|---|---|
| 선정 궤적 | 약 7,550 | **7,824** | 3.6 % 많음 |
| 선정 스텝 | 약 1.1 M | **1,236,438** | 12 % 많음 |
| 장면 | 80 | 80 | 일치 |
| 목표 물체 | 6종 | 6종 | 일치 |
| (장면 × 물체) 조합 | 누락 0 | **403/403, 누락 0** | 통과 |

목표별 궤적 수: chair 1,503 · bed 1,477 · toilet 1,418 · tv_monitor 1,375 · sofa 1,246 ·
plant 805. 길이 0인 궤적은 원본에도 선정분에도 없다.

**궤적 수가 274개 많은 이유는 설명된다.** 76,394 ÷ 10 = 7,639이고, 그룹마다 고정 간격 추출이
올림으로 뽑히므로 403개 그룹에서 평균 0.5개씩 더해져 약 +185가 된다. 합이 7,824로 맞는다.

**논문의 7,550이 더 적은 이유는 알 수 없다.** 논문의 부표본 평균 길이를 역산하면
1.1 M ÷ 7,550 = 145.7 스텝으로 전체 평균 159보다 8 % 짧은데, 고정 간격 추출은 평균 길이를
바꾸지 않으므로(우리 부표본 평균은 158.0으로 전체와 거의 같다) 논문은 어딘가에서 긴 궤적을
덜 담았거나 단순히 어림수를 적은 것이다. 논문이 "approximately"라고 썼고 차이가 3.6 %이므로
**논문 문장을 그대로 구현한 7,824를 쓴다.** 억지로 7,550에 맞추려면 논문에 없는 규칙을 새로
만들어야 해서 오히려 멀어진다.

예상 RGB 용량은 1,236,438 스텝 × 900 KB = **1,061 GB**다.

### 0.4 평가 데이터

**[일치]** 논문은 학습에서 본 적 없는 검증 장면 20개에서 2000 에피소드로 평가한다
(부록 C.1, 본문 Table 3). Habitat의 공식 ObjectNav HM3D **v1** 데이터셋을 받아
`/data/topovlm/habitat/datasets/objectnav/hm3d/v1/`에 두었다(138 MB).

받은 검증 집합의 에피소드 수가 논문 Table 3의 물체별 표본 수와 **여섯 항목 모두 정확히**
일치한다. 즉 논문이 평가에 쓴 바로 그 에피소드 집합이다.

| 목표 물체 | 논문 Table 3 | 받은 v1 val |
|---|---|---|
| 침대 (bed) | 433 | 433 |
| 의자 (chair) | 428 | 428 |
| 변기 (toilet) | 398 | 398 |
| 소파 (sofa) | 376 | 376 |
| 텔레비전 (tv_monitor) | 281 | 281 |
| 화분 (plant) | 84 | 84 |
| **전체** | **2000** | **2000** |

검증 장면도 20개로 같다. 저장소에 이미 있던 ObjectNav **v2**판(검증 36장면)은 쓰지 않는다.
사람 시연이 v1 장면을 참조하고 논문의 평가 설정도 v1 기준이기 때문이다.

### 0.5 시연 파일의 형식 — 1단계에 영향을 주는 세 가지

부표본을 고른 뒤 시연 파일을 전수 조사하면서 확인한 사실이다. 셋 다 1단계 렌더링의 설계를
바꾸므로 여기에 남긴다.

**(1) 모든 에피소드가 STOP으로 시작해 STOP으로 끝난다.** 전체 STOP이 152,788개로
76,394 × 2와 정확히 같아, 에피소드마다 둘뿐임이 확인된다. 첫 번째는 "아직 행동하지 않음"을
뜻하는 표시이고 마지막이 실제 정지 행동이다. **첫 항목을 버리지 않으면 정책은 0스텝에서 즉시
정지하도록 배운다.** 기존 `topovlm_data/habitat_web_render.py`의 `_drop_leading_initial_stop`이
이미 이 처리를 하고 있고, 이 구현도 같게 한다.

**(2) 매 스텝의 자세가 기록돼 있지 않다.** `agent_state`가 채워진 스텝은 표본 5장면 기준
29.5 %뿐이고 그마저 불규칙하다(에피소드 5,003개 중 1,573개는 첫 스텝만 채워져 있다).

기존 `HabitatSimReplayRenderer`는 매 스텝 기록된 자세로 에이전트를 **순간이동**시키고 그 값이
없으면 예외를 던지므로 이 데이터에 쓸 수 없다. 그 코드는 매 스텝 전체 자세가 들어 있는 MP3D판을
겨냥해 쓰인 것으로 보인다. 따라서 **시작 자세에서 출발해 행동을 실제로 시뮬레이터에 먹이는
방식**으로 재생한다. Habitat의 동역학이 결정적이므로(부록 C.1) 결과는 사람이 본 것과 같다.
`nav_baseline/env.py`의 `make_sim`·`reset_to`가 이 방식에 더 가까워 그쪽을 재사용한다.

**(3) 시연에 시선 행동이 들어 있다. [미명시 → 사용자와 합의]**

논문은 부록 C.1에서 "pitch를 바꾸는 행동을 제거해 네 개만 남긴다"고 밝히지만, **시연에 든 시선
행동을 어떻게 했는지는 적지 않았다.** 실제 분포는 다음과 같다.

| 행동 | 스텝 수 | 비율 |
|---|---|---|
| MOVE_FORWARD | 7,549,277 | 62.10 % |
| TURN_RIGHT | 2,209,186 | 18.17 % |
| TURN_LEFT | 2,139,199 | 17.60 % |
| STOP | 152,788 | 1.26 % |
| LOOK_DOWN | 54,066 | 0.44 % |
| LOOK_UP | 52,127 | 0.43 % |

시선 행동이 든 에피소드는 76,394개 중 21,141개(27.7 %)다.

**결정: 해당 스텝만 제거하고 전부 수평 시점으로 렌더링한다.** 근거는 Habitat에서 시선 행동이
카메라 각도만 바꾸고 에이전트의 위치와 방위각은 건드리지 않는다는 점이다. 따라서 그 스텝을 빼도
**사람이 걸어간 경로는 한 치도 달라지지 않고**, 그 순간의 카메라 상하 각도만 달라진다. 학습과
배치가 모두 수평 시점이 되어 관측 분포도 일치한다.

두 대안은 택하지 않았다. 재생 때만 시선을 움직이면 사람이 본 화면은 정확해지지만 정책이 배치
때 만들 수 없는 시점의 프레임으로 학습하게 된다. 시선 행동이 든 에피소드를 통째로 버리면 27.7 %를
잃고, 논문이 보고한 77k·7550과 맞지 않아 논문도 그 방식은 쓰지 않은 것으로 보인다.

### 0.6 두 편집을 적용한 뒤의 최종 규모

부표본 7,824 궤적에 위 두 편집을 적용한 결과다. 이 수치가 1단계 렌더링 용량과 2단계 인코딩
비용의 기준이 된다.

| | 스텝 수 | 변화 |
|---|---|---|
| 원본 | 1,236,438 | |
| 선행 STOP 제거 후 | 1,228,614 | −7,824 (에피소드당 정확히 1개) |
| 시선 행동 제거 후 | **1,219,318** | −9,296 |

평균 155.8 스텝, 중앙값 109, 최대 2,612. 길이 0인 궤적은 없다.
**RGB 용량은 1,219,318 × 900 KB = 1,047 GB.**

중앙값(109)이 평균(155.8)보다 훨씬 작은 것은 사람 시연의 길이 분포가 오른쪽으로 길게 끌리기
때문이다. 목표를 빨리 찾은 에피소드가 다수이고, 집을 오래 헤맨 소수가 평균을 끌어올린다.

---

## 1단계 — 재생 렌더링

코드: `render.py`. 검증 도구: `verify_replay.py`. slurm 스크립트는 `slurm/` 아래.

### 1.1 무엇을 만드는가

시연은 이미지를 담고 있지 않다. 시작 자세와 행동 열뿐이므로, 시뮬레이터에 그 행동을 다시
먹여 사람이 본 화면을 복원해야 한다. 부록 C.1이 "모든 관찰·행동·동역학은 결정적"이라고
밝히므로 이 복원은 원리적으로 정확하다 — **다만 시뮬레이터를 논문과 같은 설정으로 놓았을
때만 그렇다.** 이 절의 대부분은 그 설정을 맞추는 이야기다.

궤적마다 세 가지를 저장한다. RGB `(T,480,640,3)` uint8, 행동 `(T,)` int64,
자세 `(T,4)` float32 = (x, y, z, 방위각). 관찰은 **행동을 실행하기 전에** 기록한다. 정책이
마주할 상황이 "화면을 보고 무엇을 할지 고르는" 것이기 때문이다. 따라서 프레임 수와 행동
수가 같고, 마지막 프레임은 사람이 그것을 보고 정지를 택한 화면이다.

### 1.2 시뮬레이터 설정 — 논문이 인용한 설정 파일에서 직접 읽었다

**[일치]** 부록 C.1은 공간과 에이전트 규격이 "Habitat이 제공하는 기본값, 즉 HM3D ObjectNav
설정 파일과 대체로 같다"고 밝히고 본문에는 회전 각도(30°)와 이미지 크기만 적는다. 나머지는
그 설정 파일 자체에서 읽었다(habitat-lab 0.3.3,
`benchmark/nav/objectnav/objectnav_hm3d.yaml`, 지정되지 않은 항목은
`config/default_structured_configs.py`의 기본값).

| 항목 | 값 | 출처 |
|---|---|---|
| RGB 크기 | 480 × 640 | objectnav_hm3d.yaml `height`/`width` |
| 시야각 | 79° | objectnav_hm3d.yaml `hfov` |
| 카메라 높이 | 0.88 m | objectnav_hm3d.yaml `rgb_sensor.position` |
| 에이전트 높이 | 0.88 m | objectnav_hm3d.yaml `height` |
| 에이전트 반지름 | 0.18 m | objectnav_hm3d.yaml `radius` |
| 회전 각도 | 30° | objectnav_hm3d.yaml `turn_angle` (부록 C.1과도 일치) |
| 전진 거리 | 0.25 m | default_structured_configs.py `forward_step_size` |
| 벽 미끄러짐 | **끔** | objectnav_hm3d.yaml `allow_sliding: False` |
| 성공 거리 | 0.1 m (관찰 지점까지) | objectnav.yaml `success_distance`, `distance_to: VIEW_POINTS` |
| 에피소드 최대 길이 | 500 스텝 | objectnav_hm3d.yaml `max_episode_steps` (5단계에서 사용) |

### 1.3 내비메시를 다시 계산해야 한다 — 재현의 성패가 갈린 지점

**HM3D가 배포하는 `.navmesh` 파일은 반지름 0.10 m, 높이 1.5 m짜리 에이전트용이다.**
ObjectNav의 에이전트는 반지름 0.18 m, 높이 0.88 m로 더 넓고 낮다. 넓은 에이전트가 지나갈 수
없는 틈을 좁은 에이전트는 통과하므로, 배포된 파일을 그대로 쓰면 **사람이 막혔던 자리를 재생이
통과해 버리고 그 뒤로는 다른 길을 걷게 된다.**

habitat-lab은 바로 이 때문에 설정된 에이전트가 파일이 만들어진 에이전트와 다르면 통행 가능
면을 다시 계산한다(`sims/habitat_simulator/habitat_simulator.py`의 `default_agent_navmesh`
블록). 그 절차를 그대로 따랐다: `NavMeshSettings.set_defaults()` 후 반지름 0.18, 높이 0.88,
`agent_max_climb` 0.2, `agent_max_slope` 45.0, `include_static_objects` False.

효과는 자세가 온전히 기록된 시연 40개로 측정했다.

| | 배포본 그대로 | 재계산 후 |
|---|---|---|
| 통행 가능 면적 | 80.3 m² | 65.4 m² |
| 전진 한 걸음 정확도 (1 cm 이내) | 79.7 % | **99.7 %** |
| 좌회전 / 우회전 | 98.8 % / 99.7 % | **100 % / 100 %** |
| **궤적 전체 누적 오차 (중앙값)** | **0.173 m** | **0.0000 m** |
| 궤적 전체 1 cm 이내 | 34.5 % | **99.8 %** |

### 1.4 재생이 맞는지 어떻게 확인했는가

이 문제는 눈에 띄지 않는 종류다. 렌더링은 오류 없이 끝나고 파일도 정상으로 보인다. 그래서
검사를 두 겹으로 두었다.

**검사 1 — 기록된 자세와의 직접 비교.** 시연의 약 30 %는 매 스텝의 자세를 아직 담고 있다.
그 자세와 재생한 자세를 견주면 재생이 맞는지 바로 알 수 있다. 이때 두 가지를 짚어야 했다.

- 기록된 `agent_state[i]`는 행동 i를 **실행한 뒤**의 상태다. 인덱스 6이 전진인데 그 위치가
  이미 정확히 0.25 m 나가 있고 앞의 1~5가 전부 회전(움직일 수 없음)인 것이 근거다.
- 한 걸음만 재는 검사와 궤적 전체를 재는 검사를 나눠야 한다. 앞의 것은 행동 하나가 맞는지를
  보고, 뒤의 것은 오차가 쌓이는지를 본다. 원인 규명에는 앞의 것이, 실제 품질 판단에는 뒤의
  것이 필요하다.

**검사 2 — 목표 도달률.** 사람 시연은 전부 성공한 궤적이므로, 재생이 맞다면 궤적의 끝이 목표
관찰 지점 0.1 m 안에 들어야 한다. 이것이 처음 이상을 알린 신호였다(7.5 %).

**성공 판정 자체가 맞는지도 따로 확인했다.** 시뮬레이션 없이 데이터에 기록된 사람의 마지막
자세만으로 거리를 재니 200개 전부 0.1 m 이내였다(중앙 0.047 m, 최대 정확히 0.100 m). 목표
관찰 지점 조회와 판정 기준이 맞고 시연도 전부 성공했음이 확인되었으므로, 틀린 것은 재생
쪽이라고 좁힐 수 있었다.

### 1.5 시작 자세는 `start_position`을 쓴다

에피소드가 선언한 `start_position`과 재생 기록의 첫 자세가 시연의 약 20 %에서 어긋나고, 그
거리는 항상 정확히 0.250 m(전진 한 걸음)다. 어느 쪽이 맞는지 양쪽으로 재생해 측정했다.

| 출발점 | 궤적 전체 오차 중앙값 | 1 cm 이내 |
|---|---|---|
| **`start_position`** | **0.0000 m** | **99.8 %** |
| 재생 기록의 첫 자세 | 2.396 m | 14.6 % |

**`start_position`이 맞다.** 어긋나는 시연들은 기록된 자세 쪽이 밀려 있는 것이다.

### 1.6 검증 결과

두 검사를 모두 통과한 뒤에야 전체 렌더링에 들어갔다.

| | 내비메시 수정 전 | 수정 후 |
|---|---|---|
| 목표 관찰 지점 0.1 m 이내에서 종료 | 7.5 % (3/40) | **97.5 % (39/40)** |
| 궤적 전체 자세 오차 (중앙값) | 0.173 m | **0.0000 m** |
| 렌더링 속도 | 61 frames/s | **102 frames/s** |

속도가 함께 빨라진 것은 충돌 처리량이 줄었기 때문이다.

목표 관찰 지점이 재계산된 내비메시에서도 여전히 도달 가능한지는 따로 확인했다. 사람이 실제로
멈춘 것으로 기록된 지점에서 거리를 재면 **두 내비메시 모두 100 %가 0.1 m 이내이고 도달 불가는
0건**이다. 통행 가능 면적이 19 % 줄어도 관찰 지점 자체는 영향을 받지 않는다.

### 1.7 디버깅에서 얻은 교훈 — 서버에 코드가 갔는지 확인할 것

내비메시를 고친 뒤에도 도달률이 7.5 %로 똑같이 나와, 한때 "내비메시는 원인이 아니다"라고
잘못 판단했다. 실제로는 `rsync --update`가 수정된 파일을 **조용히 건너뛰어** 계산 노드가 옛
코드를 돌린 것이었다. 두 기계의 시계가 어긋나 `--update`의 "목적지가 더 최신" 판정이 잘못
걸렸고, 작업은 정상 종료했기 때문에 겉으로는 아무 이상이 없었다.

따라서 이 프로젝트에서 코드를 밀 때는 `rsync -a -c`(체크섬)를 쓰고, 중요한 변경은
`ssh <head> "grep -c <새 심볼> <원격 경로>"`로 반영을 확인한다.

### 1.8 전체 렌더링 결과

80개 장면을 slurm 배열(장면당 한 작업, 동시 실행 8개로 제한)로 돌렸고 전부 정상 종료했다.

| 항목 | 결과 |
|---|---|
| 장면 | 80 (전부) |
| 시연 | 7,824 (선정분 전부, 누락 0) |
| 프레임 | **1,219,318** (사전 계산치와 일치) |
| **목표 도달률** | **98.9 %** (7,739/7,824) |
| RGB | 1.1 TB |
| 행동 / 자세 | 33 MB / 38 MB |
| 궤적 길이 | 평균 155.8, 중앙 109, 최대 2,612 스텝 |

산출물은 `/data/topovlm/habitat/{rgb,actions,pose}/pr2l_habitat_web_hd/train/` 이고,
장면별 매니페스트를 합친 것이
`episodes/pr2l_habitat_web_hd/train/manifest.jsonl` (7,824줄)이다.

**미도달 85개(1.1 %)** 는 물체별로 고르게 퍼져 있고(bed 1.8 % ~ toilet 0.6 %), 종료 지점이
목표에서 중앙 2.70 m, 최대 14.5 m 떨어져 있으며 도달 불가는 0건이다. 재생이 정확하다는 것이
따로 검증되었으므로 이들은 **원본 시연 자체가 목표에 닿지 못한 경우**로 본다.

### 1.9 목표 관찰 지점까지의 거리 계산

`goals_by_category`는 장면·물체 조합마다 수백 개의 관찰 지점을 담는다. Habitat의
`MultiGoalShortestPath`는 그 전부에 대한 최단 경로를 한 번의 질의로 구한다. 직선거리로
후보를 추리는 방식은 쓰지 않았다 — 벽 너머로 가장 가까워 보이는 지점이 걸어서는 가장 먼
지점일 수 있기 때문이다.

---

## 2단계 — VLM 인코딩

코드: `vlm_features.py`(표현 추출), `pca.py`(차원 축소), `encode.py`(check / fit / encode).

### 2.1 토큰 열의 실제 구성

**[일치]** Prismatic은 이미지의 패치 임베딩을 여는 토큰 바로 뒤에 끼워 넣으므로, 언어 모델이
읽는 열은 다음과 같다. 실측으로 확인한 개수를 함께 적는다.

```
[BOS]  [시각 토큰 256개]  [질문 토큰 21개]  →  [생성 토큰 32~48개]
```

풀링 뒤 정책이 받는 것은 시각 16 + 질문 21 + 생성 43 = **80개**(한 표본 기준)다.

### 2.2 추출 위치 검증 — 통과

시각 토큰이 목표 물체를 모른다는 것은 이 배치에서 **반드시 성립해야 하는 성질**이다. 언어
모델은 인과 마스킹이라 각 위치가 자기보다 앞만 볼 수 있고 이미지가 질문 앞에 있으므로,
목표를 바꿔도 시각 위치의 값은 달라질 수 없다. 이를 검사로 만들어 두었다(`encode.py check`).

같은 프레임을 `tv_monitor`와 `toilet`으로 인코딩한 결과:

| 검사 | 결과 |
|---|---|
| 시각 토큰 16개가 비트 단위로 동일한가 | **OK** |
| 질문 토큰이 달라지는가 | **OK** |

생성된 답도 논문 Table 4와 같은 성격이 나왔다. 방의 종류를 판정하고 그 방과 목표 물체의
상식적 관계를 근거로 답한다.

> 목표 TV: "Yes, a tv_monitor would be found here **because it is a living room**. It is common
> for living rooms to have televisions. The presence of a **fireplace and couch** also…"
>
> 목표 변기: "No, a toilet would not be found here **because it is a living room**. Toilets are
> typically found in **bathrooms**…"

### 2.3 직접 디코딩 루프를 쓴 이유

**[이탈 아님 — 같은 계산의 다른 구현]** 라이브러리의 생성 함수를 쓰지 않았다.
`PrismaticVLM.generate`는 문자열만 돌려주고 내부 표현을 버리며, `generate_batch`는 이름과
달리 한 장씩 도는 반복문이다(소스 주석에도 "for now, only support batch size of 1"). 실측으로
배치 1은 0.86 frames/s, 배치 8은 3.47 frames/s로 **4배** 차이가 났다. 1,219,318 프레임 기준
392 GPU-시간과 98 GPU-시간의 차이다.

또한 논문이 요구하는 **프레임마다 고정된 시드**는 배치 전체에 전역 난수를 쓰는 방식으로는
지킬 수 없다. 옆에 어떤 프레임이 함께 묶였느냐에 따라 답이 달라지기 때문이다. 직접 루프에서
항목마다 별도 생성기를 두어, 배치 크기를 바꾸거나 작업을 나눠 돌려도 같은 프레임은 같은 답을
낸다. 시드는 `blake2b(episode_id:frame_index)`로 만든다 — 파이썬 기본 해시는 프로세스마다
소금이 달라 재현이 되지 않는다.

**패딩은 아예 없다.** 한 궤적의 모든 프레임은 목표가 같아 질문 문장이 완전히 동일하므로,
궤적 안에서 배치를 묶으면 길이가 저절로 맞는다.

### 2.4 메모리 — 배치 크기의 한계는 모델이 아니라 구현이 정했다

배치 16에서 CUDA 메모리 부족이 났고, 원인이 둘이었다.

**사전 채움 출력을 붙들고 있었다.** 라이브러리에 층을 골라 달라고 요청할 방법이 없어
`output_hidden_states=True`가 **33개 층 전부**를 만들고, 접두부 277개 위치 **전부**에 logits까지
만든다. 필요한 것은 층 2개와 마지막 위치 하나뿐인데 `output` 이름이 살아 있는 동안 전부
메모리에 남았다. 필요한 값을 꺼낸 직후 해제하도록 했다.

**할당자가 파편화됐다.** 실패 시점에 22.66 GiB 사용 중 **5.88 GiB가 예약됐지만 쓰이지 않는**
상태였다. 생성 길이가 32~48로 들쭉날쭉해 크기가 제각각인 블록이 쌓인 탓이며,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 완화했다.

### 2.5 확정된 설정

| 항목 | 값 | 근거 |
|---|---|---|
| 모델 | `prism-dinosiglip-224px+7b` | **[미명시 → 선택]** — 2026-09-14 정정. 논문은 "Dino+SigLIP, Llama2-7B-pure, 224px"만 적었고 맞는 체크포인트가 둘이다. 논문 Table 4 답 형식은 이 체크포인트와 맞고 controlled와는 맞지 않는다(C.9) |
| 온도 / 생성 길이 | 0.4 / 최소 32, 최대 48 | 논문 |
| 층 | 마지막 2개 | 논문 |
| 시각 풀링 | 추출 후 4×4 → 16 | 논문 |
| PCA | 4096 → 1024, **공유 기저 하나** | 논문 (§2.6) |
| 결합 | 층별 축소 후 stack → 2048 | 논문 |
| 질문 형식 | `In: {질문}\nOut: ` | Prismatic 학습 형식 [미명시] |
| 시드 | 프레임마다 고정 | [미명시 → 합의] |
| 저장 dtype | **float16** | [미명시 → 합의] |

### 2.6 PCA 기저를 하나로 둔 근거

부록 C.2 항목 2는 "모든 토큰의 주성분 벡터를 구해(compute all resulting tokens' principle
component vectors) 그 벡터로 모든 토큰을 4096에서 1024로 줄인다"고 적는다. 단수 집합을
가리키는 표현이고, **이 항목에는 "층"이라는 말이 아예 없다** — 두 층을 쓴다는 사실은 다음
항목에서야 처음 나온다. 즉 저자는 이 단계에서 추출된 것을 하나의 토큰 덩어리로 다루고 있다.
구현상으로도 `[층 2, 토큰 n, 4096]`을 `reshape(2n, 4096)` 후 한 번 적합하는 것이 가장 짧다.
항목의 첫 마디가 "To reduce the size of"인 것처럼 표현 설계가 아니라 용량 처리로 서술된다는
점도 단순한 구현 쪽을 가리킨다.

다만 두 층이 대칭이 아니라는 사실은 확인해 둘 필요가 있다. HuggingFace의 Llama는 마지막
hidden state를 **최종 RMSNorm을 통과시킨 뒤** 담고(`modeling_llama.py`의 `self.norm` 직후),
그 앞 층은 정규화 전의 잔차 흐름이다. 크기가 크게 다르면 주성분이 큰 쪽에 쏠려 작은 쪽의
정보를 버리게 되고, 그러면 두 층을 쓴 의미가 사라진다. 그래서 적합 단계에서 **층별 기저도
함께 구해 층별 보존율을 기록**한다. 추가 인코딩 없이 같은 표본에서 나오므로 비용이 없다.

**재현이 목적이므로, 층별 기저가 더 잘 보존하더라도 공유 기저를 쓴다.** 여기서 바꾸면 나중에
성공률이 논문과 달라졌을 때 표현 탓인지 PCA를 바꾼 탓인지 가릴 수 없게 된다. 격차가 심하면
보고하고 판단을 받되, 그것은 재현이 끝난 뒤의 추가 실험으로 다룬다.

### 2.7 실행 중 겪은 환경 문제 세 가지

코드와 무관하게 세 번 실패했고, 셋 다 **오류를 즉시 내지 않아** 진단이 늦었다. 같은 함정을
다시 밟지 않도록 남긴다.

1. **`bml-compute07`(titanrtx)이 `/data`에 쓰지 못했다.** 작업은 RUNNING으로 보였지만 로그
   파일이 아예 생성되지 않았다. `echo` 세 줄짜리 작업도 마찬가지였고, 같은 코드가
   compute01/02/04에서는 정상이었다. 노드는 `comp`(COMPLETING)에 물려 있었다. → 처음 쓰는
   노드에는 먼저 진단 작업을 보내고, 진단 로그는 `/data`가 아닌 곳에 쓴다.
2. **`rsync`를 상대 경로로 실행해 수정이 서버에 반영되지 않았다.** 셸의 작업 디렉터리가 이전
   `cd`의 잔재로 남아 엉뚱한 디렉터리를 동기화했다. → 절대 경로로 밀고, 반영 여부를
   `grep`으로 확인한다.
3. **`HF_TOKEN`을 빠뜨렸다.** `prismatic.load()`는 VLM 체크포인트를 얹기 전에 **gated**
   `meta-llama/Llama-2-7b-hf`에서 Llama-2 설정을 받아 빈 껍데기를 만드는데, 인증이 없으면
   시각 백본 두 개를 다 올린 **4분 뒤에** 401로 죽는다. → VLM을 올리는 스크립트는 `HF_HOME`과
   `HF_TOKEN`을 함께 내보낸다.

---

## 3단계 — 정책

### 3.1 부록 I Listing 1을 다시 읽고 고친 것 — 어텐션 헤드 수

논문 부록 I가 정책의 예시 코드를 싣는다.

```python
class Policy(torch.nn.Module):
    def __init__(self, num_actions, tf_embed_dim=4096):
        self.embed_fc = torch.nn.Linear(tf_embed_dim, 1024)
        self.action_fc = torch.nn.Linear(1024, num_actions)
        self.transformer = torch.nn.Transformer(
            1024,                     # d_model
            1,                        # ← nhead
            num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=1024, batch_first=True)
        self.cls = torch.nn.Embedding(1, 1024)
```

`torch.nn.Transformer`의 두 번째 위치 인자는 `nhead`다. 즉 **헤드는 1개**다. PLAN.md는 한때
이 항목을 "논문 미명시"로 분류하고 8을 골랐는데, 논문이 명시하고 있었다. `NUM_HEADS = 1`로
고쳤다.

헤드 수는 `nn.Transformer`의 크기를 바꾸지 않는다. 어느 쪽이든 입력 투영은 1024×3072이고,
`policy.py`를 직접 돌려 확인한 총 파라미터는 **1이든 8이든 78,717,476개**로 같다. 달라지는
것은 CLS 질의가 프레임의 토큰들을 볼 때 1024차원을 한 번에 보느냐, 128차원씩 여덟 갈래로
나눠 보느냐뿐이다. `num_heads=8`은 인자로 남겨 두었고, 재현이 끝난 뒤의 실험 대상이다 —
헤드가 여럿이면 하나는 시각 토큰에, 다른 하나는 생성된 문장에 붙을 수 있어 유리할 여지가
있다. 나침반 인코딩(`heading_encoding`)과 같은 처리다.

Listing 1의 나머지는 우리 구현과 일치한다.

| Listing 1 | 우리 | |
|---|---|---|
| d_model 1024 | 1024 | 일치 |
| num_encoder_layers 1 / num_decoder_layers 1 | 1 / 1 | 일치 |
| dim_feedforward 1024 | 1024 | 일치 |
| dropout (인자 없음 → PyTorch 기본 0.1) | 0.1 | 일치 |
| activation (인자 없음 → PyTorch 기본 ReLU) | ReLU | 일치 |
| batch_first True | True | 일치 |
| `src_key_padding_mask` + `memory_key_padding_mask` | 동일 | 일치 |
| nhead 1 | **8 → 1로 정정** | |

Listing 1은 축약본이다 — LSTM도 비시각 관찰도 없고 `tf_embed_dim` 기본값이 PCA 이전
값(4096)이다. 그래도 논문이 헤드 수에 대해 남긴 유일한 구체적 값이므로 이것을 따른다.

### 3.2 대조군의 입력 폭은 조건마다 다르다 — 그것이 Listing 1의 `tf_embed_dim`이다

Listing 1이 입력 폭을 생성자 인자로 둔 이유가 여기서 드러난다. 부록 C.2 일반 5의 "정책
구조를 동일하게 유지한다"는 표현 폭까지 같게 만들라는 뜻이 아니라, **폭을 1024로 낮추는
투영 이후가 전부 같아야 한다**는 뜻이다.

| 조건 | 표현 | 폭 | PCA | 층 stack | 4×4 풀링 |
|---|---|---|---|---|---|
| PR2L + CoT | LLM 마지막 2층, 전 위치 | 2048 | 4096→1024 | 함 | 함 |
| PR2L CoT 없음 | 같음, 생성 없음 | 2048 | 4096→1024 | 함 | 함 |
| VLM 이미지 인코더 | 시각 백본 패치만 | **2176** | **안 함** | **안 함** | 함 |

세 문장이 각각을 정한다.

1. **PCA는 PR2L 전용이다.** 부록 C.2가 "general" 항목과 "For PR2L-specific design choices"로
   나뉘는데 PCA는 후자의 항목 2이고, 문장도 `"To reduce the size of VLM representations for
   PR2L"`로 시작한다.
2. **4×4 풀링은 적용한다.** 일반 항목 5는 "for policies that receive visual observations as a
   sequence of tokens"에 걸린다. 괄호 안 예시가 `PR2L, VC-1 with patch embeddings`뿐이지만
   규칙 자체는 조건이 아니라 형태로 걸려 있고, 이미지 인코더 조건은 256개 패치 토큰 열이다.
3. **폭 2176은 모델에서 확인했다.** `prismatic/models/backbones/vision/dinosiglip_vit.py`:

   ```python
   "dinosiglip-vit-so-224px": {"dino": "vit_large_patch14_reg4_dinov2.lvd142m",
                               "siglip": "vit_so400m_patch14_siglip_224"}
   def embed_dim(self): return self.dino_featurizer.embed_dim + self.siglip_featurizer.embed_dim
   ```

   ViT-L 1024 + SO400M 1152 = 2176, 패치 256개(16×16).

층 stack을 하지 않는 이유는 층이 하나뿐이기 때문이다. 시각 백본에는 "마지막 두 층"이라는
개념이 PR2L에서와 같은 의미로 존재하지 않는다(부록 C.2 PR2L 항목 3은 LLM 층을 가리킨다).

### 3.3 논문이 통제하지 않은 교란 하나 — 결과에 함께 적는다

부록 D가 이렇게 적는다.

> We empirically note that **longer visual embedding sequences tend to perform better in
> Habitat.** To control for this, we opt to use InstructBLIP's Q-Former unprompted embeddings
> instead of the ViT embeddings directly

즉 논문은 열 길이가 성능에 영향을 준다는 것을 알고 있었고, 단순화 설정(부록 D)에서는 길이를
맞춰 통제했다. 그러나 **본문 Table 3에서는 맞추지 않았다** — 이미지 인코더 조건은 16토큰,
PR2L(CoT)은 약 70~85토큰이다. 그러므로 41.9 % 대 11.6 %에는 "프롬프팅으로 표현이 좋아진
효과"와 "정책이 볼 토큰이 5배 많은 효과"가 섞여 있다.

우리는 논문대로 재현하되, 이 점을 결과 보고에 남긴다. 참고로 **41.9 % 대 27.8 %**(CoT 유무)는
양쪽 다 질문 토큰을 받으므로 이 교란에서 비교적 자유롭고, 논문의 핵심 주장을 더 깨끗하게
뒷받침한다.

---

## 4단계 — 학습

### 4.1 overfit 관문이 잡은 것 — 전부 패딩인 프레임이 NaN을 만든다

첫 실행(작업 10232)은 **epoch 1부터 손실이 `nan`**이었고, 정책은 모든 스텝에 STOP을 냈다.

```
epoch   1 | loss nan | 정확도  28.5% | 정지  60.0% 전진 41.2% 좌회전 0.0% 우회전  5.3%
epoch   2 | loss nan | 정확도   0.8% | 정지 100.0% 전진  0.0% 좌회전 0.0% 우회전  0.0%
```

**원인.** 길이가 다른 궤적을 한 배치에 담으면 짧은 쪽은 끝을 지나 채워진다. 그렇게 채워진
자리는 85개 토큰 위치가 **전부** 패딩이라, CLS 질의가 어텐션을 걸 때 softmax의 모든 항이
금지된다. 그 결과는 0이 아니라 **NaN**이다. 실측: 배치 (3, 293, 85, 2048)에서 879칸 중
**121칸**이 전부 패딩이었고, 저장된 임베딩 자체에는 비정상 값이 하나도 없었다.

**왜 손실에서 걸러지지 않았나.** `weighted_loss`가 `losses * weights * valid`로 무효 스텝을
지우고 있었는데, **NaN에 0을 곱해도 NaN**이다. 곱셈은 NaN을 막지 못한다. 그래서 합계가
NaN이 되고, 기울기가 NaN이 되고, 가중치가 NaN이 되고, 전부 NaN인 logit에 `argmax`를 하면
인덱스 0 — 즉 STOP — 이 나온다. epoch 1의 28.5 %(거의 무작위)가 epoch 2에 0.8 %(전부
STOP)로 무너진 순서가 정확히 이것이다.

**수정 두 곳.** 하나는 원인을, 하나는 전파를 막는다.

1. `policy.FrameSummary.forward` — 전부 패딩인 행은 한 자리를 열어 둔다. 그 행이 내놓는
   요약은 무의미하지만 유한하고, 하류가 이미 `valid`로 버린다.
2. `train.weighted_loss` — 곱하는 대신 `torch.where`로 **골라낸다.** 곱셈은 NaN을 통과시키고
   선택은 통과시키지 않는다.

두 번째만 해도 손실은 살아나지만, 첫 번째가 없으면 NaN이 LSTM으로 들어간다. 둘 다 필요하다.

**수정 확인** (같은 배치, 전부 패딩 121칸 그대로): logits 전부 유한, 손실 2.8923,
**비정상 기울기 0개**. 손실값도 타당하다 — 4지선다 무작위가 ln 4 = 1.386이고 inflection
가중이 평균 2배쯤이므로 그 곱 근처다.

### 4.2 overfit 통과 — 그리고 inflection 가중이 눈에 보인다

```
epoch   1 | 정확도 61.6% | 정지   0.0% 전진 91.7% 좌회전  4.3% 우회전  3.1%
epoch  10 | 정확도 64.5% | 정지  75.0% 전진 82.2% 좌회전 27.7% 우회전 31.0%
epoch  50 | 정확도 92.9% | 정지  90.0% 전진 93.3% 좌회전 91.3% 우회전 93.1%
epoch 200 | 정확도 99.4% | 정지 100.0% 전진 99.8% 좌회전 98.5% 우회전 98.7%
```

epoch 1이 부록 C.2 일반 6이 막으려는 바로 그 정책이다 — **전진만 하고 정지도 회전도 하지
않는다.** 시연의 3분의 2가 전진이므로 그것만 맞히는 것이 초반에는 이득이기 때문이다.
epoch 10에서 회전이 27~31 %로 올라오는 것이 inflection 가중과 정지·회전 1.5배가 실제로
작동한 증거이고, 그 뒤 네 행동이 함께 수렴한다.

**이 곡선은 전체 정확도만 봐서는 읽을 수 없다.** epoch 1의 61.6 %와 epoch 10의 64.5 %는
거의 같은데, 안에서 벌어지는 일은 정반대다. 행동별로 찍는 이유가 이것이다.

### 4.3 읽기와 계산의 비율

20궤적(약 1 GB) 기준 **읽기 2~3초, 계산 1초**. 읽기가 지배하지만 실효 속도가 약 330 MB/s로
나온다. 782궤적 37 GB로 환산하면 epoch당 약 110초, 40 epoch에 **약 1.2시간**이다. 착수 전에
최악 32시간까지 열어 두었던 추정이 크게 좁혀졌고, 임베딩을 `/scratch`로 옮길 필요는 없다.

(이 추정의 근거였던 `/data` 115 MB/s는 `frames_of`의 mmap이 만든 수치였다. 같은 파일을 통짜로
읽으면 1 GB/s가 넘는다 — 145 MB 파일에서 mmap 배치읽기 1.12초 대 통짜 읽기 0.13초로 8.3배.
NFS에서 mmap은 4 KB 페이지 단위 요청으로 쪼개지기 때문이다.)

---

## 5단계 — 평가

### 5.1 행동 선택은 argmax — Listing 3이 명시한다

부록 I Listing 3:

```python
act_logits = policy.forward((seq, mask)).reshape(env.num_actions)
action = torch.argmax(act_logits)
obs, _, _, _ = env.step(action)
```

표집이 아니라 결정적 선택이다. `evaluate.py`도 `argmax`를 쓴다. ObjectNav에서 결정적
정책은 좌회전만 반복하는 고리에 갇히기 쉬운데, 평가 로그의 `steps`가 상한 500에 몰리면
그 증상이다. 논문이 그렇게 했으므로 그대로 둔다.

### 5.2 디코딩 시드는 에피소드마다 다르다

학습 때 시드는 `blake2b("{episode_id}:{frame_index}")`였다. 평가는 여러 에피소드를 동시에
굴리고 목표별로 묶어 인코딩하므로 한 배치가 여러 에피소드에 걸친다. 처음에는 배치 전체에
`f"eval:{goal}"` 하나를 넘겨서, **같은 목표를 가진 두 에피소드가 같은 스텝에서 같은 시드**를
쓰고 있었다. 이미지가 다르니 생성 결과는 어차피 갈라져 정확성 문제는 아니었지만 학습과
일관되지 않았다.

`encode_batch`가 이름 하나 또는 프레임당 하나를 받도록 고쳤다.

```python
owners = [episode_id] * batch if isinstance(episode_id, str) else list(episode_id)
if len(owners) != batch:
    raise ValueError(f"{len(owners)} episode ids for {batch} frames")
```

`encode.py`는 문자열을 넘기므로 시드가 한 비트도 바뀌지 않는다(`ep_A:7`이 이전과 같은 값).
이미 만들어진 임베딩과 앞으로 만들 것이 섞여도 안전하고, 재인코딩이 필요 없다.

---

## 4단계 이후

(진행하면서 채운다.)

---

## 6단계 — 이미지 인코더 대조군

### 6.1 논문이 정하는 것 — 침묵하는 곳이 거의 없다

| 인용 | 출처 |
|---|---|
| "a policy on **Prismatic VLM image encoder embeddings** (equivalent to Minecraft approach (a), but with Dino+SigLIP)" | 본문 4.2 |
| "instead using **task-agnostic embeddings from the VLM's image encoder** … For a fair comparison, we use **the exact same policy architecture and hyperparameters**" | 본문 4.1 (a) |
| "replace the image embeddings with **a learned Transformer layer that condenses our input token embeddings (from the VLM, VLM image encoder, or VC-1)** into a single summary embedding" | 본문 4.2 |
| "PR2L outperforms (a) … **even though both approaches receive the same visual features**, with PR2L simply transforming those features via prompting an LLM (**with no additional information from the environment**)" | 본문 5 |

마지막 문장이 이 조건의 존재 이유다. 두 조건이 받는 픽셀이 같아야만 41.9 % 대 11.6 %를
**프롬프팅의 효과**로 읽을 수 있다. 그러므로 구현에서 지켜야 할 것은 "비슷하게"가 아니라
**같은 전처리·같은 백본·같은 산술**이다.

### 6.2 세 가지가 PR2L에만 걸린다

| | 적용 | 근거 |
|---|---|---|
| 4×4 평균 풀링 | **함** | 부록 C.2 **일반** 5 — "policies that receive visual observations as a **sequence of tokens**"에 걸린다. 256 패치가 정확히 그것이다 |
| PCA 4096→1024 | **안 함** | 부록 C.2 **PR2L 전용** 2. 문장도 "To reduce the size of VLM representations **for PR2L**" |
| 두 층 stack | **안 함** | 같은 PR2L 전용 3. 시각 백본에는 그런 의미의 "마지막 두 층"이 없다 |

#### 6.2.1 풀링 항목의 괄호에 이미지 인코더가 없다 — 그래도 적용이 맞다

부록 C.2 일반 5의 열거는 이렇게 되어 있다.

> "For policies that receive visual observations as a sequence of tokens (**PR2L, VC-1 with
> patch embeddings**), we apply 2D average pooling with kernel sizes of 4 × 4 … **We do this to
> ensure that policy performance differences are due to representation quality, not
> architecture.**"

괄호에 이미지 인코더가 없다. 한때 이것을 "논문은 대조군에 풀링을 안 했을 수도 있다"는
가설로 세웠는데, **문장 자신의 근거와 충돌하므로 버린다.**

- 풀링의 목적이 "차이가 표현의 질에서 오게 하고 구조에서 오지 않게" 하는 것이다. 대조군만
  256 토큰으로 두면 PR2L(16 시각 + 21 질문 + 32~48 생성 ≒ 78)보다 **세 배 넘게 긴 열**을
  받게 되어, 목적이 정확히 뒤집힌다.
- 본문 4.2는 요약 Transformer의 입력을 "our input token embeddings (from the VLM, **VLM
  image encoder**, or VC-1)"로 적으며 대조군을 명시적으로 포함한다. 즉 대조군도 토큰 열을
  Transformer에 넣는다.
- 본문 4.1 (a)의 "the exact same policy architecture and hyperparameters"도 같은 방향이다.

따라서 괄호의 누락은 열거가 불완전한 것이고, **풀링 적용이 논문의 의도와 일치하는 유일한
읽기**다. 우리 구현은 이대로 둔다.

### 6.3 폭 2176은 논문에 없지만 모델에 있다

```python
# prismatic/models/backbones/vision/dinosiglip_vit.py
"dinosiglip-vit-so-224px": {"dino": "vit_large_patch14_reg4_dinov2.lvd142m",
                            "siglip": "vit_so400m_patch14_siglip_224"}
def forward(self, pixel_values):
    return torch.cat([self.dino_featurizer(...), self.siglip_featurizer(...)], dim=2)
def embed_dim(self): return dino.embed_dim + siglip.embed_dim      # 1024 + 1152 = 2176
```

Listing 1이 `def __init__(self, num_actions, tf_embed_dim=4096)`으로 폭을 **생성자 인자**로
둔 이유가 이것이다. `policy.NavigationPolicy(token_dim=...)`으로 옮겼고, **1024로 낮추는
투영 이후는 세 조건이 한 줄도 다르지 않다** — 부록 C.2 일반 5가 요구하는 통제다.

#### 6.3.1 투영기 앞에서 뽑는가 뒤에서 뽑는가 — 부록 G가 답한다

시각 백본의 출력(2176)과 projector를 통과한 뒤의 값(4096) 중 어느 쪽이 "image encoder
embeddings"인지는 Habitat 절에 없다. 나중에 부록 G(Minecraft 세부)에서 답을 찾았다.

> "InstructBLIP's token embeddings are larger than **ViT-g/14's (used in the VLM image encoder
> baseline)**, and so may carry more information. … to ensure consistent policy expressivity, we
> include **a learned linear layer projecting all representations for this baseline and our
> approach to the same size (512 dimensions)**"

ViT-g/14는 InstructBLIP의 **시각 타워 원본**이고 Q-Former 이전이다. 즉 대조군은 언어 쪽으로
넘어가기 전의 백본 출력을 쓴다. 두 조건의 폭이 서로 다르다는 사실을 논문이 명시적으로
언급하고, 그 차이를 **학습된 선형층 하나로 흡수**한다는 점까지 우리 구현과 같다 —
Minecraft에서는 512로, Habitat Listing 1에서는 1024로 내린다. Habitat 대조군은 "equivalent
to Minecraft approach (a), but with Dino+SigLIP"이므로 같은 규칙이 걸린다.

따라서 **projector 이전 2176**이 맞고, 이것은 우리가 고른 값이 아니라 논문이 정한 값이다.

#### 6.3.2 백본의 어느 층인가 — 우리가 고르지 않았다

DINOv2도 SigLIP도 마지막 층이 아니라 **끝에서 두 번째 층**을 낸다. Prismatic이 그렇게
바꿔 놓았기 때문이다.

```python
# dinosiglip_vit.py:61  "By default set `get_intermediate_layers` to return the
#                        *SECOND-TO-LAST* layer patches!"
self.dino_featurizer.forward = unpack_tuple(
    partial(self.dino_featurizer.get_intermediate_layers, n={len(...blocks) - 2}))
```

우리는 `vlm.vision_backbone(pixel_values)`를 그대로 호출하므로 PR2L 경로가 밟는 층을 그대로
밟는다. 층 선택은 구현 결정이 아니라 상속이다. 256개(16×16)가 나오는 것도 여기서 확인된다 —
`get_intermediate_layers`가 접두 토큰(CLS·register)을 빼고 패치만 주기 때문이고,
`encode_vision_batch`가 개수를 검사해 다르면 즉시 멈춘다.

### 6.4 정밀도를 한 번 틀렸다가 고쳤다

처음에는 시각 백본을 **float32**로 돌렸다. 이유는 "rtx2080에는 bfloat16이 없는데 그 카드가
13장 놀고 있으니, 카드에 맞춰 정밀도를 정하자"였다. **순서가 거꾸로였다.**

`load_vlm()`은 모델 전체를 `llm_backbone.half_precision_dtype`으로 캐스팅하고, 그 값은
**bfloat16**이다(`llama2.py`: "LLaMa-2 was trained in BF16"). 즉 PR2L 경로에서 시각 백본은
bfloat16으로 돈다. 대조군을 float32로 돌리면 더 정확하기는 해도 **"the same visual
features"가 아니게 된다.** 이 대조군의 가치는 정확도가 아니라 일치에서 나온다.

고친 내용은 셋이다.

1. `load_vision_backbone`이 `torch.bfloat16`으로 캐스팅한다.
2. `encode_vision_batch`가 **CoT 경로와 같은 형태**로 돈다 — float32 픽셀을 넣고 autocast가
   캐스팅한다. dtype만 맞추고 형태가 다르면 누적 방식이 갈릴 수 있다.
3. bfloat16을 지원하지 않는 카드에 떨어지면 `RuntimeError`로 **즉시 중단**한다. 샤드마다
   정밀도가 다른 것이 가장 나쁜 결과다.

대가는 Turing 카드를 쓸 수 없다는 것이고, 작업은 `--partition=rtx5090,rtx4090,rtx3090`으로
bfloat16 카드만 요청한다.

**저장 정밀도는 원래부터 양쪽이 같다** — `STORE_DTYPE = np.float16`. 헷갈리기 쉬운 지점이라
적어 둔다: 저장이 float16이고 연산이 bfloat16이다.

### 6.5 가중치가 같다는 것은 우연이 아니다

Prismatic 체크포인트는 projector와 언어 모델만 담는다.

```python
assert "projector" in model_state_dict and "llm_backbone" in model_state_dict, \
    "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"
```

시각 백본은 두 경로 모두 timm에서 받는다. 따라서 백본만 따로 세워도 **VLM 안의 것과 같은
가중치**이고, 논문의 "same visual features"가 이 구현에서도 참이 된다.

### 6.6 실측

| | |
|---|---|
| 속도 | **22.5 프레임/s** (rtx5090, 배치 32) — CoT 3.95의 5.7배 |
| 프레임당 | 16토큰 × 2176 × 2바이트 = **68 KB** (CoT 332 KB의 5분의 1) |
| 전체 | 1,219,318 프레임 → **15 GPU-시간**, 약 83 GB |
| 값 범위 | −71.9 ~ 55.4, 비정상값 0 |

**7B를 올리지 않는 것이 실무적으로 중요하다.** 백본만이면 VRAM이 20 GB에서 3 GB로 줄어,
PR2L이 3090 넉 장을 19시간 쓰는 동안 **다른 카드에서 병렬로** 돌 수 있다. 처음에는
`--dependency=afterany`로 묶어 19시간을 기다리게 해 두었는데, 그럴 이유가 없었다.

---

## 7단계 — 표현의 PCA 그림 (보류, 설계 확정)

계획과 남은 작업은 `PLAN.md` §6.3에 있다. 여기에는 이미 만들어진 것만 적는다.

**전문가 롤아웃 28궤적 / 2,304프레임** (`rollout.py`, 작업 10341, 2분 42초).

| | 값 | 대조 |
|---|---|---|
| 목표 도달 | **28/28** | 최단경로 추종기는 거의 최적이므로 100 %가 정상. 80 % 미만이면 실패 처리하게 해 두었다 |
| 평균 스텝 | **82** | 본문: "taking **80 steps** for a privileged shortest path follower to succeed and 150+ for humans" |
| 사람 시연 평균 | 155 | 같은 문장의 "150+"와 일치 |
| 출발 거리 | 4.0 ~ 21.9 m | 색(가치)이 실제로 변화를 보이기에 충분한 폭 |

30이 아니라 28인 것은 고른 4개 장면에서 특정 물체의 에피소드가 10개에 못 미쳤기 때문이다.
장면 수를 늘리면 30을 채울 수 있고, 그림의 성격은 바뀌지 않는다.

매 스텝의 **목표까지 측지거리**를 함께 기록했다. 이것이 논문의 색(오라클의 가치)에 대응하는
값인 이유는 `PLAN.md` §6.3에 적었다.

### 6.7 rtx5090 경고 — 확인 후 무해로 판정

이미지 인코더 인코딩 전량(7,824궤적, 80 GB)이 rtx5090에서 돌았는데, 학습을 시작할 때 이
경고가 stderr에 있는 것을 발견했다.

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current
PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 ... sm_90.
```

**"돌아갔으니 괜찮다"로 넘길 수 없는 종류다.** 커널이 조용히 틀린 값을 내는 경우는 4단계의
NaN 버그와 같은 부류 — 학습은 진행되는데 값이 틀린 — 이고, 걸린 것이 11 GPU-시간과 조건
하나 전체였다.

**정황 증거는 정상 쪽이었다**: GPU가 실제로 점유돼 있었고(6 GB, 97 %), CPU 대체가 아니었으며,
임베딩에 NaN이나 발산이 없었고, 학습 첫 epoch도 그럴듯했다. 그러나 정황은 검증이 아니다.

**대조 방법**: 같은 프레임 6장을 **CPU에서 float32로** 다시 통과시켜 저장된 값과 비교했다.
정밀도가 달라 완전히 같을 수는 없지만(bfloat16 연산 후 float16 저장), 커널이 망가졌다면
차이가 반올림 수준을 한참 넘는다.

```
최대 절대차 0.0151   (값 범위 −71.9 ~ 55.4)
최대 상대차 0.0220   (bfloat16 가수 8비트가 주는 약 0.008에 float16 저장이 얹힌 정도)
상관        1.000000
```

**정상**이다. 경고는 이 빌드에 sm_120 네이티브 커널이 없다는 뜻이고, 드라이버가 하위
아키텍처의 PTX를 JIT 컴파일해 실행한다. 재인코딩하지 않았다.

**남길 것 하나**: `load_vision_backbone(device="cpu")`는 그대로 쓸 수 없다. 가중치를
bfloat16으로 캐스팅하는데 `encode_vision_batch`의 autocast는 `device.type == "cuda"`일 때만
켜지므로, CPU에서는 float32 입력이 bfloat16 가중치를 만나 죽는다. 검증에서 이 경로를 처음
밟아 드러났다. 실행 경로가 항상 cuda라 실무에는 영향이 없어 고치지 않았다.

### 5.3 성공 판정이 habitat-lab의 것과 같은지 확인

논문은 성공을 스스로 정의하지 않는다. 부록 C.1이 "the defaults provided by Habitat, as
specified in the HM3D ObjectNav configuration file"이라고 위임하므로, 이 항목은 읽어서 판단할
문제가 아니라 **대조해서 확인할 문제**다. 자세 변환을 검증했던 방식과 같다.

**설정 파일이 정하는 것** (`objectnav_hm3d.yaml` + 상속받는 `task/objectnav.yaml`):

```yaml
distance_to: VIEW_POINTS      success_distance: 0.1      max_episode_steps: 500
turn_angle: 30   hfov: 79   height: 0.88   radius: 0.18   allow_sliding: False
```

**측정 코드** (`habitat/tasks/nav/nav.py`):

```python
# DistanceToGoal
view_points = [vp.agent_state.position for goal in episode.goals for vp in goal.view_points]
distance    = sim.geodesic_distance(current_position, view_points, episode)
# Success
is_stop_called and distance_to_target < success_distance
```

세 가지가 맞아야 같은 판정이고, 셋 다 소스에서 확인했다.

| | habitat-lab | 이 구현 |
|---|---|---|
| 관찰 지점 | `episode.goals = goals_by_category[goals_key]`, `goals_key = f"{basename(scene_id)}_{object_category}"` — **그 물체의 모든 인스턴스** | 같은 키로 같은 사전을 조회 |
| 거리 | `HabitatSim.geodesic_distance`가 `habitat_sim.MultiGoalShortestPath`에 전 지점을 `requested_ends`로 넣음 | `geodesic_to_viewpoints`가 같은 호출 |
| 임계 | `< 0.1` | `<= 0.1` (경계값에서만 다름) |

`episode.goals`가 **카테고리의 모든 인스턴스**라는 점이 중요하다. "의자를 찾아라"는 집 안의
모든 의자의 모든 관찰 지점이 목표라는 뜻이고, 에피소드당 관찰 지점이 중앙 741개인 이유다.

**대조 결과** (`evaluate.py --check-success`, 작업 10489):

```
최대 거리 오차   0.000e+00 m
성공 판정 불일치 0
→ habitat-lab의 판정과 동일
```

항행 가능 영역의 무작위 위치를 두 방식으로 채점해 비교한 것이고, 오차가 정확히 0이다.
`--check-success`로 언제든 다시 돌릴 수 있다.

**한 번 실패했다.** 검사 블록을 모델 적재 뒤에 두어, 판정을 비교하기도 전에 시각 백본을
Turing 카드에 올리려다 bfloat16 가드에 걸려 죽었다(작업 10488). 이 검사는 pathfinder만
필요하므로 블록을 체크포인트·정책·모델보다 앞으로 옮겼다. 가드 자체는 의도대로 동작했다 —
조용히 다른 정밀도로 도는 대신 멈췄다.

---

# 부록 Z. 고쳐야 할 것 (2026-08-26 코드 감사)

CoT 평가가 도는 동안 파이프라인 전체를 다시 읽으며 정리했다. 감사의 출발점은 "우리 성공률이
논문보다 두 배 높은 것이 버그 때문인가"였고, **성공률을 부풀리는 버그는 찾지 못했다.** 아래는
그 과정에서 나온 실제 결함들이며, 셋 다 결과를 뒤집지 않지만 셋 다 논문의 명세와 다르다.

## Z.1 시각 토큰이 한 칸 밀린다 — 가장 큰 것

Prismatic이 언어 모델에 넣는 순서는 `prismatic/models/vlms/prismatic.py:329`가 정한다.

```python
multimodal_embeddings = torch.cat([
    input_embeddings[multimodal_indices, :1, :],   # BOS
    projected_patch_embeddings,                    # 패치 256개
    input_embeddings[multimodal_indices, 1:, :],   # 나머지 질문
], dim=1)
```

`vlm_features._assemble`은 이렇게 자른다.

```python
visual = prefix[:, :VISUAL_GRID**2]     # BOS + 패치 0~254
text   = prefix[:, VISUAL_GRID**2:]     # 패치 255 + 질문
```

**한 칸 밀렸다.** `prefix[:, 1:257]`이 시각이고, BOS는 질문 쪽에 붙어야 한다.

`visual_count = prefix.shape[1] - prompt_length`가 256으로 나와 검사를 통과하는 이유는, 총
길이가 `1 + 256 + (prompt_length-1)`이라 **어디서 자르든 차이가 256이기 때문**이다. 이 검사는
길이만 보고 위치는 보지 않는다.

**영향**

- 4×4 풀링이 공간적으로 이웃하지 않는 패치를 함께 평균한다. BOS가 첫 풀링 토큰에 섞인다
- 패치 255는 풀링을 피해 "질문 토큰"으로 남는다
- **정보 손실은 없다** — BOS와 256패치가 모두 표현에 있고 토큰 총 개수도 의도대로다
- **PR2L 경로에만 있다.** `encode_vision_batch`는 백본에서 패치를 직접 받아 BOS가 없다

즉 **CoT만 시각 정보가 흐트러진 채로 학습됐고, 그 상태로 이미지 인코더를 이겼다.** 성능을
낮추는 방향의 결함이므로 우리 수치가 높은 이유가 아니다. 고치면 CoT가 더 오를 여지가 있다.

**수정 완료 (2026-08-26).** `_assemble`이 이제 이렇게 자른다.

```python
visual = prefix[:, 1:1 + VISUAL_GRID**2]
text   = torch.cat([prefix[:, :1], prefix[:, 1 + VISUAL_GRID**2:]], dim=1)
```

**단, 지금까지의 모든 산출물은 수정 전 코드로 만들어졌다.** 저장된 임베딩(1.2M 프레임), 학습된
체크포인트, 그리고 이 문서에 실린 모든 성공률과 PCA 수치가 그렇다. 수정된 코드로 같은 결과를
얻으려면 **재인코딩 13시간 + 재학습 7시간 + 재평가 1시간**이 필요하고, 그 판단은 따로 한다.

**실측으로도 확인했다 (작업 10749).** 인과 마스킹 때문에 위치 0은 자기 자신만 볼 수 있으므로,
그 자리가 BOS라면 이미지가 바뀌어도 은닉 상태가 변하지 않아야 한다. 서로 다른 두 이미지를 같은
프롬프트로 통과시킨 결과다.

```
첫 토큰 id 1 | BOS id 1 | 질문 길이 18
prefill 길이 274 = 256 + 18

위치   0:  두 이미지 간 최대 차이  0.000e+00   ← 이미지와 무관 = BOS
위치   1:  1.175e+01
위치   2:  7.656e+00
위치 255:  7.547e+00
위치 256:  6.203e+00
```

위치 0의 차이가 **정확히 0**이고 위치 1부터 커진다. 패치는 1번에서 시작한다. 소스 확인과 실측이
같은 답을 냈다.

길이 검사가 왜 무력했는지도 이 출력에 드러난다 — `274 − 18 = 256`이라, 어디서 자르든 256이 나온다.

**중간 상태를 만들지 않는 것이 중요하다.** 인코딩만 고치고 옛 체크포인트로 평가하면, 정책이 본 적
없는 표현을 받게 되어 성능이 떨어지는데 그 원인이 버그 수정인지 다른 무엇인지 구분할 수 없다.
인코딩·학습·평가는 한 묶음으로 다시 돌려야 한다.

## Z.2 학습은 float16, 평가는 float32

```python
# encode.py:218        학습 데이터로 저장할 때
pieces.append(reduced.astype(np.float16))
# evaluate.py:434      평가에서 만들 때
tokens[slot, 0, :count] = torch.from_numpy(reduced)   # float32
```

정책이 float16으로 반올림된 토큰으로 학습하고 평가에서는 반올림되지 않은 값을 본다. 상대
오차가 10^-3 수준이고 **두 조건에 똑같이 적용**되므로 비교는 왜곡되지 않지만, "학습과 평가가
같은 표현을 쓴다"는 전제와 어긋난다.

**수정 (2026-09-10 구현)**: `STORE_DTYPE`을 `vlm_features.py` 한 곳으로 옮기고 — 이전에는
`encode.py`와 `figure_encode.py`에 같은 값이 두 번 적혀 있었다 — 평가의 롤아웃도 같은 상수로
반올림하게 했다. 재인코딩은 필요 없고 평가만 다시 돌리면 된다.

**다만 기본값은 꺼짐(`--match-train-precision`)이다.** 지금까지 보고한 모든 성공률(이미지
25.2 %, C3 13.6 %, C3′ 15.2 %, CoT 60.2 %)이 반올림 없이 나온 값이라, 켠 채로 새 평가를
돌리면 레시피 효과에 이 수정이 섞인다. 부록 X가 폭과 깊이를 한 번에 바꿔 두 실행을 더 써야
했던 것과 같은 종류의 실수다.

**언제 켜는가**: 서로 비교할 실행 묶음 전체를 켠 채로 돌릴 때. 그 전에 대조 하나가 필요하다 —
기존 체크포인트(예: `stageB_image`, 25.2 %)를 켠 채 다시 평가해 총계가 잡음 안에서 같은지
본다. 부록 V.10에서 입력이 2.4e-7까지 일치해도 궤적은 갈렸지만 총계는 1.0σ 안에서 맞았으므로,
1e-3 섭동에서도 총계는 견딜 것으로 예상하나 측정된 바는 없다.

**대조 결과 (2026-09-13, 작업 15850)**: `stageB_image`를 반올림을 켠 채 같은 500 에피소드로
다시 평가했다. **27.4 % 대 25.2 %, +2.2 %p, 0.8σ** — 구분되지 않는다. float16 반올림은 총계를
움직이지 않으므로 지금까지의 모든 수치는 그대로 유효하고, 이후 실행은 켠 채로 돌려도 과거와
비교할 수 있다.

## Z.3 5090에서 PyTorch가 JIT 폴백으로 돈다

```
torch 2.2.0, CUDA 빌드 11.8  →  sm_120 커널 없음
```

에피소드당 소요가 3090 137초, 4090 177초인데 **5090은 423~582초**다. 그림 인코딩에서도 5090이
0.78 프레임/s, 3090이 3.05 프레임/s였다 — 당시 프롬프트 길이 차이로 설명했으나 **틀렸다.**

**수정**: 별도 환경에 torch 2.7 + cu128을 설치한다. habitat-sim 0.3.3이 `torchvision 0.17.0
py39_cu118`과 묶여 있어 재빌드나 버전 맞추기가 필요하다. **기존 환경은 보존**하고, 새 환경에서
같은 체크포인트로 한 장면을 평가해 성공률이 일치하는지 먼저 대조한다. 일치하지 않으면 버전
탓인지 실수 탓인지부터 가려야 한다.

세 조건(image / cot / no-cot)을 **모두 새 환경에서 다시** 돌릴 때 적용하는 것이 맞다. 지금
바꾸면 이미지 인코더의 25.2 %와 산술 기반이 달라진다.

## Z.4 고친 것

`evaluate.check_success_against_habitat`의 부등호가 한쪽은 `<`, 다른 쪽은 `<=`였다. 채점이
아니라 대조 코드이고 경계값에 걸릴 확률이 0이라 영향은 없었지만, 대조의 의미가 약해지므로
양변을 `<=`로 맞췄다.

## Z.5 검사해서 이상 없었던 것

| 항목 | 확인 |
|---|---|
| 평가 시 정책 입력 | 토큰·자세·나침반·이전행동·목표원핫뿐. 특권 정보 없음 |
| 성공 판정 | `stopped and distance <= 0.1m`. STOP을 눌러야 성공, 500스텝 소진은 실패 |
| 목표까지 거리 | `MultiGoalShortestPath` = 모든 관찰 지점 중 최소. habitat-lab과 동일 |
| 내비메시 | radius 0.18 / height 0.88 / climb 0.2 / slope 45 / static 제외 — **habitat-lab 기본값과 완전히 일치**하며 habitat-lab도 기본적으로 재계산한다 |
| 학습/평가 장면 | 80 대 20, 겹침 0 |
| 표현 구성 | `encode.py`와 `evaluate.py`가 같은 식, 같은 basis 파일 |
| 관찰-행동 정렬 | 재생·평가 모두 행동 전에 관찰 기록 |
| 손실 마스킹 | 패딩 스텝을 선택 제외(0 곱셈 아님), 유효 스텝 수로 나눔 |
| 물체 인덱스 | `dataset.py` 한 곳에서 정의하고 평가가 import |
| 부표본 규칙 | (장면×물체) 그룹별 stride 10 → 7,824궤적 / 1.22M스텝 (논문 7,550 / 1.1M) |
| 평가 부표본 | 중앙거리 6.09m vs 전체 6.02m, 물체 비율 동일 |
| `pca.py` | float64 누적, 대칭 고유분해, 내림차순. encode와 evaluate가 같은 basis를 로드 |
| `attention.py` | 학습·평가에서 import되지 않음 |

---

# 부록 Y. 왜 우리 수치가 논문보다 두 배 높은가 (미해결)

```
논문   PR2L(CoT) 41.9 %   이미지 인코더 11.6 %   격차 3.6배
우리   CoT       64.3 %*  이미지 인코더 25.2 %   격차 2.6배
                 * 227/500 잠정치
```

**두 조건이 함께 올랐다는 점이 핵심이다.** 한쪽만 올랐다면 우리 PR2L 구현이 유리하게 틀어진
것이겠지만, 기준선까지 2.2배 오른 것은 양쪽에 공통으로 작용한 요인을 가리킨다.

## Y.1 배제된 것

| 후보 | 어떻게 배제했는가 |
|---|---|
| 평가 표본이 쉬웠다 | 500개의 중앙 거리 6.09 m, 전체 2,000개는 6.02 m. 물체 비율은 퍼센트 단위까지 동일 |
| 평가 환경이 관대했다 | 내비메시 설정이 habitat-lab 기본값과 완전히 일치하고, habitat-lab도 기본적으로 재계산한다 |
| 성공 판정이 느슨했다 | habitat-lab과 362개 위치 대조, 최대 오차 0.000e+00 m, 불일치 0건. STOP을 눌러야 성공 |
| 학습 장면이 새어 들어갔다 | 학습 80장면과 평가 20장면이 겹치지 않는다 |
| 정책이 특권 정보를 봤다 | 평가 입력은 토큰·자세·나침반·이전행동·목표원핫뿐 |
| 논문이 밝힌 제약 때문 | 데이터 10분의 1, 40 epoch, 증강 없음, 궤적 단위 배치 — **우리도 전부 동일하게 따랐다** |

## Y.2 재생 품질 가설 — 처음 생각보다 약하다

내비메시를 재계산하지 않으면 목표 0.1 m 이내 종료율이 7.5 %로 떨어진다. 이것을 근거로 "시연의
92.5 %가 망가진다"고 서술했는데, **그 읽기는 틀렸다.**

| | 배포본 그대로 |
|---|---|
| 전진 한 걸음 정확도 | 79.7 % |
| 좌회전 / 우회전 | 98.8 % / 99.7 % |
| 궤적 누적 오차 중앙값 | **0.173 m** |

성공 반경이 0.1 m이므로 **17 cm만 밀려도 무조건 실패로 찍힌다.** 7.5 %는 좁은 관문의 통과율을
잰 것이지 관찰-행동 대응이 무너졌는지를 잰 것이 아니다. 전진의 80 %와 회전의 99 %가 정확한
데이터라면 "복도가 보이면 전진"은 대체로 학습된다.

여전히 한 방향으로 작용하는 요인이지만, **2배 차이를 혼자 설명하기에는 부족하다.**

## Y.3 남은 후보

**① 정책 세부 구현.** 논문이 밝히지 않은 값을 PIRLNav에서 가져왔다 — LSTM 2048×2층, 부가 입력
임베딩 32, 어텐션 헤드 1. 논문은 "[43]과 같은 LSTM"이라고만 쓴다. 이 값들이 다르면 성능이
갈리고, **양쪽 조건에 똑같이 작용**하므로 관측된 모양과 맞는다.

**② 논문 정책이 덜 학습됐을 가능성.** 논문은 학습 정확도를 보고하지 않는다. 우리는 40 epoch에서
CoT 91.9 %, 이미지 인코더 92.9 %다. 논문이 같은 수준에 닿았는지 알 방법이 없다.

**③ 재생 품질** (Y.2에서 약해진 것).

## Y.4 확정하는 실험

**내비메시 재계산 없이 렌더링한 데이터로 이미지 인코더를 다시 학습한다.**

- 11.6 % 부근이 나오면 → 재생 품질이 원인, 가설 확정
- 여전히 20 %대면 → 재생은 원인이 아니고 ①·②로 넘어간다

어느 쪽이 나오든 알아낼 것이 있다. 비용은 렌더링 12시간 + 학습 7시간 + 평가 1시간.

---

# 부록 X. 용량 가설과 그 반증 실험 (설계만, 미실행)

## X.1 가설

우리 정책이 논문 것보다 커서 성공률이 두 배다. 정보 누출도 구현 오류도 아니고, 같은 데이터·같은
정보에 **더 큰 망**을 얹었을 뿐이라는 설명이다. 양쪽 조건(CoT와 이미지 인코더)에 똑같이 작용하므로
관측된 모양 — 둘 다 2.2배 — 과 맞는다.

## X.1a "정책 용량"은 사실상 LSTM 폭 하나다 (실측, 2026-08-28)

파라미터를 모듈별로 세어 보면 흔들 곳이 어디인지가 분명해진다.

```
memory  (LSTM 2048 x 2층)      59,801,600    75.8%
summary (Transformer + CLS)     19,038,208    24.1%
action  (4-way 헤드)                 8,196     0.0%
side    (자세·나침반·직전행동·목표)      544     0.0%
────────────────────────────────────────────────
합계                            78,848,548
```

**정책의 4분의 3이 LSTM 하나다.** 부가 입력 임베딩은 544개로 반올림하면 0이므로, X.2가 "미확정
값이 셋"이라 적은 것 중 임베딩 폭(32)은 성공률을 움직일 수 있는 규모가 아니다. S 실행에서 16으로
줄이는 것은 실험을 단순하게 하려는 것이지 그 자체를 시험하는 것이 아니다.

따라서 이 실험이 실제로 흔드는 변수는 **LSTM 폭과 층수 하나**이고, 판정 기준 ①·②는 그것에 대한
진술로 읽어야 한다.

## X.2 논문이 밝힌 것과 밝히지 않은 것

**Habitat 정책의 구조를 적은 표도 코드도 없다.** 표 8개 중 Habitat 관련은 Table 3(결과)와
Table 4(CoT 예시)뿐이고, Table 6·7과 Listing 1은 전부 Minecraft용이다(Table 7의 제목이
"All policy hyperparameters for all **Minecraft** tasks").

LSTM 언급은 본문 4.2의 한 문장이 전부다.

> "We adopt **the same LSTM-based recurrent architecture used by that work** [Majumdar et al.],
> but replace the image embeddings with a learned Transformer layer..."

폭도 층수도 없다. 부록 C.2 항목 3이 "hidden states for the RNN portion of our policy"로 순환의
존재만 재확인한다.

따라서 우리 값은 두 단계를 거슬러 얻은 것이다.

```
PR2L 논문  "[43]과 같은 LSTM"  →  VC-1 (Majumdar)  →  PIRLNav  →  LSTM 2048 × 2층
```

**정정 (2026-09-03, VC-1 원문 확인 후).** 이 사슬은 절반만 사실이다. VC-1을 받아 읽은 결과:

* **층수 2는 추적된 값이다.** 부록 A.2가 직접 적는다 — "for use by the policy layers, which is
  **a 2-layer LSTM for navigation** and a 2-layer GRU for manipulation".
* **폭 2048은 유추다.** VC-1 전체에 폭이 없다 — `2048`도, hidden size라는 말도 나오지 않는다.
  그리고 **VC-1은 PIRLNav을 구조의 출처로 인용하지 않는다.** PIRLNav[47]은 ObjectNav의
  "best prior result"(70.4 %) 수치 출처로 한 번 나올 뿐이고, 데모 데이터는 Habitat-Web[62]을
  인용한다. 같은 저자군·같은 데이터·같은 과제라 찾아볼 곳으로는 타당했지만, **인용을 따라간
  것이 아니라 유사 연구에서 가져온 것**이다.

**이것이 용량 실험의 해석을 바꾼다.**

```
L  2048x2   깊이는 논문 명시값, 폭은 유추
W  2048x1   VC-1이 명시한 깊이 2를 벗어남
N   512x2   깊이를 지키고 폭만 줄임        <- 논문 근거에 가장 가까운 축소
S   512x1   폭·깊이 둘 다 벗어남           <- 판정을 가른 실행
```

**판정을 가른 S가 VC-1이 명시한 2층을 어긴 실행이다.** "논문이 이 크기를 썼을 수 있다"는
주장에서 S보다 **N(512x2)이 더 옹호 가능한 후보**이며, 그래서 나중에 추가한 W·N 두 실행 중
N의 값어치가 처음 생각보다 크다.

**정황 하나**: 같은 논문의 Minecraft 정책(Listing 1)에는 순환 층이 아예 없다. 요약 Transformer
뒤에 `action_fc`가 바로 붙고, Table 7의 MLP는 은닉층 1개 × 128이다. 저자들의 관심이 "표현이
좋으면 작은 정책으로도 된다"는 데 있으므로, 정책 쪽에 큰 망을 쓰는 경향은 아니라고 읽힌다.
Habitat은 VC-1 계열을 따른다고 했으니 그대로 옮길 수는 없지만, 우리가 상한을 잡았을 가능성은
남는다.

미확정 값이 셋이고 **셋 다 양쪽 조건에 똑같이 작용한다**: LSTM 폭·층수, 부가 입력 임베딩 폭(32),
어텐션 헤드 수(이것만 Listing 1의 둘째 인자에서 확인됨).

## X.3 설계

**조건은 이미지 인코더로 한다.** 세 가지 이유다. (1) 그 격차(11.6 → 25.2 %)는 VLM 체크포인트
변종으로 설명되지 않는 유일한 것이다 — 그 경로는 projector도 언어 모델도 지나지 않는다.
(2) 토큰이 이미 저장돼 있어 **재인코딩이 필요 없고 정책만 바뀐다.** (3) 데이터가 80 GB라 노드
페이지 캐시에 들어가 epoch이 빠르다(CoT의 365 GB와 다르다).

| 실행 | LSTM | 부가 임베딩 | 비고 |
|---|---|---|---|
| **L** | 2048 × 2층 | 32 | 현재. **25.2 % 이미 있음** |
| **M** | 1024 × 2층 | 32 | 중간 |
| **S** | 512 × 1층 | 16 | 어떤 합리적 해석보다도 작음 |
| **XS** | 순환 없음 | 16 | Listing 1 형태 — 요약에서 행동 직결 |
| **L′** | 2048 × 2층 | 32 | **L의 재실행, 다른 초기화 시드** |

**L′이 없으면 실험이 성립하지 않는다.** L과 S의 차이가 용량 때문인지 초기화 운 때문인지 가릴
수 없기 때문이다.

나머지는 전부 동일하게 둔다 — 같은 데이터, 40 epoch, lr 1e-4, 같은 스케줄, 같은 손실 가중치.

## X.4 판정 기준 (실행 전에 못박는다)

```
① L · M · S 가 서로 ±3 %p 안에 모임
   → 이 범위에서 용량은 성공률을 좌우하지 않는다
   → 논문이 어떤 크기를 썼든 11.6 % 를 만들지 못한다. 가설 기각

② S 가 12~16 % 로 떨어짐
   → 용량이 살아 있는 설명. 격차의 몇 %p 를 덮는지 수치로 나온다

③ |L − L′| 이 |L − S| 만큼 큼
   → 검정력 부족. 시드를 더 돌리기 전에는 ①·②를 읽을 수 없다
```

**③을 먼저 확인해야 ①·②가 의미를 갖는다.**

XS는 크게 떨어질 것으로 예상된다 — 탐색 과제에서 과거를 기억하지 못하기 때문이다. 그것 자체가
유용하다. 순환이 성공률에 얼마나 기여하는지의 하한이 된다.

## X.5 검정력과 비용

500 에피소드의 표준오차는 ±1.9 %p이고, 가르려는 차이(13.6 %p)는 그 7배다. 250 에피소드
(±2.7 %p)로 줄여도 5배라 충분하다.

```
학습  4회 (M, S, XS, L′) × 약 6시간  →  GPU 4장 병렬이면 6시간
평가  4회 × 500 에피소드 (샤드 분할)  →  약 1시간
합계  약 7시간
```

**싼 사전 점검**: S를 10 epoch만 돌려 L의 첫 10 epoch과 학습 정확도 궤적을 비교한다(1.5시간).
바짝 따라가면 이 범위에서 용량이 적합을 제한하지 않는다는 약한 증거다.

## X.6 필요한 코드 변경 (약 20줄, 아직 안 함)

`policy.py`의 `LSTM_HIDDEN`, `LSTM_LAYERS`, `SIDE_EMBED_DIM`이 모듈 상수다. 생성자 인자와 CLI
플래그로 빼고 **체크포인트에 저장해야 한다** — `evaluate.py`가 체크포인트에서 정책을 재구성하므로,
저장하지 않으면 평가가 다른 크기로 만들어 가중치 적재에서 실패한다.

```python
payload = {..., "lstm_hidden": ..., "lstm_layers": ..., "side_dim": ...}
```

## X.7 인정해야 할 교란

작은 망에는 다른 학습률이 최적일 수 있고, 학습률을 고정하면 S가 불리해진다. 다만 시험하는 명제가
**"논문이 같은 레시피로 작은 망을 썼다면"** 이므로 레시피 고정이 맞다. ②가 나왔을 때만 "용량인가
학습률인가"를 다시 갈라내면 된다.

---

# 부록 W. 우리 결과의 타당성 검증 (설계만, 미실행)

부록 X·Y가 "왜 논문과 다른가"를 묻는다면, 여기는 다른 질문이다 — **"우리 구현에 잘못된 것이
없고, 정책이 같은 조건에서 길찾기를 잘할 뿐"임을 어떻게 보이는가.**

지금까지의 검증은 대부분 **우리 코드가 우리 코드를 검사**하는 형태였다. 성공 판정을 habitat-lab과
대조했지만 그것은 거리 계산 하나였고, 롤아웃 루프·자세 복원·에피소드 종료·배치 처리는 전부 우리
것이다. 아래는 그 사슬을 바깥에서 끊어 보는 시험들이며, **통과를 확인하는 것이 아니라 반증을
시도하는 형태**로 설계했다. 판정 기준은 실행 전에 못박는다.

## W.1 실험 1-A — habitat-lab 환경 안에서 정책을 돌린다 ★ 최우선

`evaluate.py`를 쓰지 않고 habitat-lab의 `Env`로 같은 정책을 굴린다.

```
habitat-lab 이 제공                  우리가 제공
관찰 (rgb, gps, compass, objectgoal)  VLM 인코딩 + 정책 forward
에피소드 진행·종료                     행동 하나
성공 판정 (Success measure)
```

우리 코드에서 남는 것은 **표현을 만들고 행동을 고르는 부분뿐**이고 나머지는 표준 구현이 된다.

**판정 (2026-08-28 고침)**: 검증 **2,000 에피소드 전부**에서 두 루프의 성공률이 **±3 %p 안**.

원래 기준은 "같은 100 에피소드에서 ±3 %p"였고 **틀렸다.** 에피소드를 짝지어 비교할 수 있다고
전제했는데, 폐루프에서는 성립하지 않는다. 진단(V.9)이 보인 대로 두 루프는 같은 입력을 주고도
어느 스텝에서 로짓이 1e-4 어긋나면 argmax가 뒤집히고, 그 순간부터 에이전트가 다른 곳으로 가서
관측이 전부 달라진다. 6 에피소드 예비 실행에서 380 대 202 스텝, 500 대 117 스텝이 나온 것이
그 결과다. **개별 에피소드의 성공/실패가 일치하기를 요구하는 것은 원리적으로 불가능한 요구였다.**

성립하는 진술은 이것이다 — 정책이 같고 환경이 같다면, 경로가 각기 달라도 **성공률의 기댓값은
같아야 한다.** 다르면 그 차이가 루프의 편향이다.

짝지음의 이득이 사라지므로 표본 수가 그만큼 더 필요하다.

```
n=100    ±4.3 %p   기준 ±3 %p보다 커서 통과·실패를 가르지 못한다
n=500    ±1.9 %p
n=2000   ±1.0 %p
```

또한 조건은 **이미지 인코더**로 한다. habitat-lab은 에피소드를 하나씩 넘기므로 CoT로 하면
배치 구성이 우리 루프와 다를 수밖에 없고, 3-A가 보인 표집 갈림이 비교 대상 수치 안으로 들어온다.
이미지 인코더는 생성이 없어 그 교란이 없으며, 검사 대상(롤아웃·종료·채점)은 조건과 무관하게
공유하는 코드다.

**비용**: 래퍼 약 200줄(작성 완료) + 2,000 에피소드 실행. **통과하면 "평가가 관대해서 높다"는
반론이 이 프로젝트의 모든 수치에 대해 한 번에 닫힌다.**

## W.2 실험 1-B — 무작위 행동 정책

학습 없이 네 행동을 균등 추출하는 정책으로 500 에피소드를 굴린다.

**판정**: **1 % 미만.** 3 %를 넘으면 과제나 채점이 느슨하다는 뜻이다.

**비용**: 30분, GPU 거의 불필요.

## W.3 실험 1-C — 결정론과 배치 등가성

(가) 같은 평가를 두 번 돌려 **에피소드별 성공 여부가 완전히 동일**한지, (나) `--parallel 1`과
`--parallel 8`이 **같은 성공 집합**을 내는지 본다.

**판정**: 완전 일치. 한 건이라도 다르면 시뮬레이터 공유(에피소드가 번갈아 자세를 넣었다 빼는 것)에
문제가 있다. habitat 이 결정론적이라는 전제(부록 C.1) 위에 세운 설계이므로 그 전제를 직접 시험하는
것이다.

**비용**: 1.5시간.

## W.4 실험 2-A — 시각 정보를 뒤섞는다 ★

평가 중 **에피소드 A의 정책에 에피소드 B의 프레임**을 넣는다. 자세·나침반·이전행동·목표는 그대로
둔다.

**판정**: 시각을 뒤섞은 성공률이 **토큰 0 바닥과 같아야 한다.** 그보다 뚜렷하게 높으면 정책이
시각을 내용이 아니라 부수 신호(토큰 개수, 길이 따위)로 쓰고 있다는 뜻이다.

**비용**: 코드 5줄 + 1시간. **싸고 결정적이다.**

## W.5 실험 2-B — 부가 입력을 하나씩 끊는다

평가에서 gps / compass / 이전행동 / 목표원핫을 각각 0으로 만들고 성공률을 본다.

**판정**: **목표원핫을 끊었을 때 반드시 크게 떨어져야 한다.** 안 떨어지면 정책이 목표를 보지 않는
것이고, "지정된 물체를 찾는다"는 주장 자체가 성립하지 않는다.

**비용**: 4회 × 1시간, 병렬 가능.

## W.6 실험 3-A — 학습 표현과 평가 표현의 동일성 ★

학습 집합의 궤적 하나를 **`evaluate.py`의 경로로 다시 인코딩**해 저장된 토큰과 비교한다.

**판정**: float16 반올림 범위 안에서 일치. 어긋나는 지점이 두 경로가 갈라진 곳이다.

**이 하나가 인코딩·PCA·저장 형식을 한꺼번에 검사한다.** 앞서 찾은 float16(학습)/float32(평가)
불일치(부록 Z.2)의 크기도 여기서 정량화된다.

**비용**: 30분.

## W.7 실험 5-A — 두 조건의 학습 집합이 문자 그대로 같은가

두 조건의 **에피소드 id 집합이 완전히 일치**하는지 대조한다. 중복 261건 사고(부록 Z 이전) 이후
스텝 수는 맞췄지만 **id 집합 자체를 대조한 적은 없다.**

**비용**: 5분.

## W.8 실험 5-B — 정책 파라미터 수가 입력 투영만 빼고 같은가

```
CoT            Linear(2048, 1024) + 나머지
이미지 인코더    Linear(2176, 1024) + 나머지
```

**판정**: 차이가 정확히 `(2176 − 2048) × 1024 = 131,072`개.

**비용**: 2분.

## W.9 우선순위

| | 실험 | 무엇을 무너뜨리려 하나 | 비용 |
|---|---|---|---|
| ★1 | 1-A habitat-lab 환경 | "평가 루프가 관대하다" | 래퍼 100줄 + 2h |
| ★2 | 2-A 시각 뒤섞기 | "시각을 안 쓰고 성공한다" | 5줄 + 1h |
| ★3 | 3-A 표현 동일성 | "학습과 평가가 다른 걸 본다" | 30분 |
| 4 | 1-B 무작위 정책 | "과제가 느슨하다" | 30분 |
| 5 | 2-B 부가 입력 절단 | "목표를 안 본다" | 4h (병렬) |
| 6 | 5-A·5-B 조건 동일성 | "비교가 불공정하다" | 10분 |
| 7 | 1-C 결정론·배치 | "평가가 불안정하다" | 1.5h |

**상위 3개(약 4시간)가 주장의 대부분을 지탱한다.** 5-A·5-B는 10분이라 언제든 끼워 넣을 수 있다.

**실행 순서는 토큰 0 대조군 결과를 본 뒤에 정한다** — 2-A의 판정 기준이 그 바닥값이기 때문이다.

---

# 부록 V. 부록 W의 실행 결과 (2026-08-26)

설계는 부록 W에 있다. 여기에는 **실제로 돌린 것과 나온 수치**만 적는다.

## V.0 실행 순서를 W.9에서 바꿨다 — 이유

W.9는 1-A를 ★1로 뒀으나, 착수 시점에 **BOS 수정(Z.1)이 이미 코드에 들어가 있고 저장된 임베딩과
체크포인트는 전부 수정 전 것**이라는 제약이 드러났다. VLM 인코딩을 하는 실험(1-A·2-A·2-B·3-A)을
그대로 돌리면 수정된 표현을 수정 전 정책에 먹이게 되어, 결과가 나빠져도 원인이 갈리지 않는다.
Z.1이 "중간 상태를 만들지 말라"고 적어둔 상황 그대로다.

되돌리지도 재인코딩(21시간)하지도 않고 **스위치를 넣었다.**

```python
# vlm_features.py
LEGACY_BOS = os.environ.get("PR2L_LEGACY_BOS", "") == "1"
```

기본값은 수정본이고, 켰을 때만 옛 자르기가 나온다. 이로써 (1) 검증 실험은 체크포인트가 실제로
학습한 분포 위에서 돌고, (2) 두 버전의 차이를 따로 잴 수 있다.

바꾼 순서: **5-A·5-B → 3-A → 1-B → (토큰 0) → 2-A → 1-A → 2-B·1-C.**
3-A를 올린 것은 그것이 스위치의 정확성까지 함께 검사하기 때문이고, 1-A를 내린 것은 유일하게
새 코드 100줄이 필요해 앞의 결과에 따라 설계가 달라지기 때문이다. 중요도 순서는 그대로다.

## V.1 5-A·5-B 통과 (`validate_conditions.py`)

```
5-A  CoT   궤적 7824 | 고유 7824 | 스텝 1,219,318
     이미지 궤적 7824 | 고유 7824 | 스텝 1,219,318
     차집합 0 / 0 | 공통 7824건 중 길이 불일치 0                       통과

5-B  폭 2048 → 78,717,476 | 폭 2176 → 78,848,548
     모양이 다른 텐서 1개: summary.project.weight (1024,2048) vs (1024,2176)
     차이 131,072 = (2176-2048) x 1024                                통과
```

중복 261건 사고 이후 스텝 총합은 맞췄지만 **id 집합을 대조한 적은 없었다.** 실제로 한 건도
어긋나지 않았고 길이까지 같으므로 두 조건은 같은 재생을 봤다. 5-B는 파라미터 수만이 아니라
이름 집합과 텐서별 모양을 전부 대조했다 — 수가 맞아도 두 군데가 상쇄됐을 수 있기 때문이다.

## V.2 1-B 통과 — 무작위 정책 0.0 %

```
500 에피소드  성공률 0.0%  SPL 0.000  자발적 정지 100.0%  (134초)
물체 6종 전부 0.0% | 거리 구간 3개 전부 0.0%
```

판정 기준 1 % 미만을 크게 통과했다. 균등 추출이면 매 스텝 1/4로 STOP이라 평균 4스텝에 멈추므로,
이 대조군이 실제로 묻는 것은 **"출발 근처에서 멈추면 성공하는가"** 이고 500번 모두 아니오였다.
검증 집합의 최소 출발 거리 구간이 2–4 m이니 당연하지만, 확인해야 확인된 것이다.

체크포인트는 읽지 않는다(`evaluate.py --random-actions`). 학습된 산출물이 있어야만 성립하는
바닥값은 바닥값이 아니다.

## V.3 3-A — 사슬은 결백, 그러나 배치 독립성 주장은 반증됐다

처음 돌린 판(작업 10754)은 저장분과 재인코딩을 통째로 비교해 12개 중 4개가 어긋났고, 그것이
사슬의 결함인지 표집의 흔들림인지 갈리지 않았다. 토큰 열이 `[풀링 시각 16][질문 18][생성 32~48]`
이고 **앞의 34개는 표집이 닿지 않으므로**, 비교를 넷으로 쪼개 다시 돌렸다(작업 10756).

| | 무엇을 비교했나 | 결과 |
|---|---|---|
| (가) | 저장분 vs 재인코딩, **결정적 블록만** | 모양 12/12 일치, 코사인 ≥ 0.99973, 최대 상대 2.57e-02 |
| (나) | 저장분 vs 재인코딩, 전체 열 | 모양 9/12, 생성 길이가 달라진 것 3건 |
| (다) | **같은 배치로 두 번** | **상대 0.00e+00, 12/12 완전 일치** |
| (라) | 배치 4개 vs 한 장씩 | 모양 8/12, 생성이 갈린 것 4건, 나머지도 ~1e-3 이동 |

**(다)가 핵심이다.** 배치가 같으면 파이프라인은 비트 단위로 재현된다. 우리 코드에 순서 의존이나
초기화되지 않은 상태 같은 흔들림은 없다.

**(라)가 원인을 확정한다.** 배치 구성을 바꾸면 갈린다. 난수열 자체는 프레임별 생성기라 배치와
무관하지만, **그 난수가 겨루는 로짓이 무관하지 않다** — 배치 모양이 다른 커널을 고르고, 반정밀도
결과가 끝자리에서 움직이고, 확률 경계 근처에 있던 토큰이 반대쪽으로 떨어지면 그 뒤가 전부 갈린다.

따라서 (나)의 불일치는 사슬의 결함이 아니라 표집의 흔들림이다. (가)에서 결정적 블록이 12/12
모양이 맞고 코사인이 0.9997 아래로 내려가지 않는 것이 그것을 뒷받침한다 — 이미지 변환·층
슬라이스·풀링·PCA 기저·저장 dtype이 두 경로에서 같은 것을 만든다.

**단, (가)는 사전에 못박은 5e-3 기준을 2건에서 넘겼고 그건 통과가 아니다.** 기준을 세울 때
float16 저장의 반올림만 셈에 넣고 **반정밀도 순전파가 배치 모양에 민감하다는 것을 빼먹었다.**
(라)가 그 크기를 따로 재 주는데 1e-3~3.6e-3이고, 여기에 저장 반올림이 얹히면 2.57e-02가 나올 수
있다. 기준이 잘못 세워진 것이지 결과가 기준을 통과한 것이 아니므로, 재기준을 세워 다시 돌리기
전까지 3-A는 **"구조는 검증됐고 수치 기준은 미달"** 로 남긴다.

**성공률에 주는 영향은 없다.** 논문이 temperature 0.4로 표집하라 했으므로 한 프레임의 표현은
애초에 확률변수다. 학습은 한 추출을, 평가는 같은 분포의 다른 추출을 받으며 어느 쪽으로도 유리하지
않다. 오히려 불일치는 성능을 낮추는 방향이다.

**`vlm_features.py`의 문서를 고쳤다.** "프레임마다 자기 생성기에서 뽑으므로 작업을 어떻게 나누든
답이 같다"는 문장은 반증됐다. 난수열이 배치와 무관하다는 것까지만 참이다.

## V.4 Z.1 정량화 — BOS 수정은 표현을 크게 바꾼다

같은 프레임·같은 시드로 legacy와 수정본을 인코딩해 **결정적 블록만** 비교했다.

```
코사인  0.718 ~ 0.795  (12 프레임, 최소 0.718290)
```

구조로 설명된다. 결정적 블록 34~37행 중 질문 17행은 두 버전이 공유하고, **풀링 시각 16행 전부와
경계 1행이 달라진다.** 즉 절반이 바뀐다. 전체 열(생성 포함) 기준으로는 약 20 %가 바뀌어 코사인
0.82가 나온다.

**재인코딩(13h) + 재학습(7h) + 재평가(1h)의 근거로 삼을 수치는 이것이다.** 코사인 0.72는 무시할
크기가 아니다. 다만 방향은 여전히 성능을 **낮추는** 쪽이므로(공간적으로 이웃하지 않는 패치를
평균한다), 우리 성공률이 논문보다 높은 이유의 후보는 아니다.

## V.5 아직 안 한 것

2-A(시각 뒤섞기)는 토큰 0 바닥값을 기다린다. **판정 기준은 손봐야 한다** — 토큰 0 학습이 17
epoch까지 정지를 0.1~0.3 %로만 예측하고 있어 바닥이 구조적으로 0 %가 될 공산이 크고, 그러면
"뒤섞기 ≈ 바닥"이 거의 자동으로 참이 되어 시험이 무력해진다. 성공률 대신 **최종 측지거리 분포**를
주 지표로 쓴다.

1-A·2-B·1-C는 그 뒤다. 1-A는 legacy 스위치를 켜고 돌려야 한다.

## V.6 토큰 0 대조군 — 바닥은 3.4 %다 (2026-08-26)

```
500 에피소드  성공률 3.4%  SPL 0.015  자발적 정지 23.2%

물체별                      출발 거리별
  chair       9.3%           2-4 m   111개   12.6%
  plant       4.8%           4-6 m   133개    2.3%
  sofa        4.3%           6+  m   256개    0.0%
  bed         1.9%
  toilet      0.0%
  tv_monitor  0.0%
```

**학습 중의 예측이 틀렸다.** 19 epoch 시점에 정지 예측이 0.8 %인 것을 보고 "40 epoch까지 사실상
0이라 성공률이 구조적으로 0 %가 된다"고 적었으나, 남은 20 epoch에서 정지가 더 올라와 자발적
정지 23.2 %가 됐다. 정지 비율의 초기 기울기로 최종값을 추정한 것이 무리였다.

**3.4 %의 출처는 거리다.** 2–4 m에서 12.6 %, 6 m 이상에서 정확히 0.0 %. 시각이 없으면 방향을
정할 근거가 없어 헤매다 가끔 멈추는데, 목표가 가까우면 우연히 반경 안에서 멈출 수 있고 멀면
한 번도 되지 않는다. 물체별도 같다 — 의자 9.3 %(집 안 어디에나 있음) 대 변기·TV 0.0 %(특정
방에만 있음).

**네 지점이 정렬됐다.**

```
무작위 행동     0.0 %   과제 자체의 바닥
토큰 0          3.4 %   자세·나침반·직전행동·목표만 준 경우
이미지 인코더   25.2 %
PR2L (CoT)     60.2 %
```

CoT와 토큰 0의 차이 **56.8 %p가 VLM 표현이 벌어들이는 몫**이다. 부가 입력만으로는 3.4 %에
그치므로 "성공률이 표현이 아니라 자세·나침반에서 나온다"는 반론은 닫힌다.

**2-A의 판정 기준을 확정한다.** 앞서 바닥이 0 %로 눌릴 것을 우려해 측지거리를 주 지표로 쓰자고
했으나 그럴 필요가 없어졌다. 시각을 뒤섞은 정책이

- **3.4 % 근처이고 거리 프로파일(12.6 / 2.3 / 0.0)도 닮으면** → 통과
- **그보다 뚜렷이 높으면** → 정책이 시각을 내용이 아니라 부수 신호(토큰 개수 따위)로 쓰는 것

성공률 하나가 아니라 거리별 세 값을 함께 보는 것은 유지한다. 전체 값만 맞고 프로파일이 다르면
다른 방식으로 성공하고 있다는 뜻이기 때문이다.

## V.7 학습 곡선 — 수렴 정도는 원인이 아니다 (2026-08-28)

지도교수가 짚은 네 가지 중 **"우리 쪽이 더 잘 수렴한 것인지"** 를 재기 위해, 10 epoch마다 저장해
두고 한 번도 쓰지 않았던 중간 체크포인트를 굴렸다. 학습은 하지 않았다.

```
CoT (200 에피소드)                이미지 인코더 (500 에피소드)
      성공률  자발적정지  논문대비        성공률  자발적정지  논문대비
ep10   46.5%    53.0%    1.11배    ep10   14.0%    53.6%    1.21배
ep20   60.5%    72.5%    1.44배    ep20   20.2%    63.4%    1.74배
ep30   62.0%    74.0%    1.48배    ep30   25.2%    69.6%    2.17배
ep40   62.0%    71.0%    1.48배    ep40   25.2%    76.4%    2.17배
논문   41.9%                        논문   11.6%
```

**결정적인 것은 epoch 10이다.**

```
              우리 epoch 10   논문 epoch 40   차이     결합 표준오차
CoT             46.5%           41.9%       +4.6%p      3.7%p  (1.2σ)
이미지           14.0%           11.6%       +2.4%p      1.7%p  (1.4σ)
```

**우리 학습의 4분의 1 지점이 논문의 최종값과 통계적으로 구분되지 않는다.** 그런데 논문도 40 epoch을
돌렸다고 적혀 있다. 학습 길이가 같으므로 "우리가 더 오래 돌려서 높다"가 성립하지 않는다. 게다가
우리 곡선은 CoT가 epoch 20, 이미지가 epoch 30에서 포화하므로 **더 돌려서 얻은 것도 아니다** —
그 구간에서 학습 정확도는 89.3 → 95.2 %로 오르는데 성공률은 움직이지 않는다.

**정직하게 남길 한계**: 이 실험은 "논문 곡선이 우리와 같은 모양이되 더 낮다"와 "모양 자체가
다르다"를 구분하지 못한다. 학습률 일정이나 배치 크기 같은 최적화 세부가 곡선 모양을 바꿀 수 있고,
그 가능성은 열려 있다. 배제된 것은 **학습 길이**이지 최적화 전반이 아니다.

**곁가지**: 두 조건 모두 후반에 자발적 정지만 계속 오르고 성공률은 평평하다(이미지 69.6 → 76.4 %에
성공률 변화 0). 사람 시연의 정지 시점을 흉내 내는 능력과 목표를 찾아가는 능력이 후반에 갈라진다.

**부분집합 처리.** `stratified_subset`은 중첩되지 않는다 — 같은 난수 생성기를 다른 크기로 부르므로
200개가 500개의 부분집합이 아니다. 그대로 뒀으면 곡선의 마지막 점만 다른 에피소드에서 잰 값이
되어 점들 사이의 차이가 수렴 효과가 아니게 된다. `evaluate.py --episode-ids`와 `pick_subset.py`를
추가해 **끝난 500개 결과에서 물체 비율을 지켜 200개를 뽑아** 이름으로 지정했고, epoch 40 값은 그
200개로 다시 계산했다(500개 60.2 % → 200개 62.0 %).

## V.8 이미지 인코더 전체 2,000 에피소드 — 부분집합은 대표적이었다 (2026-08-28)

지도교수가 첫 순서로 지목한 **"논문과 같은 전체 validation set"** 중 에피소드 수 부분을 닫는다.
이미지 인코더는 생성이 없어 에피소드당 8.4초라 2,000개가 4샤드로 한 시간이면 끝난다.

```
              500개    2000개    차이
전체          25.2%    24.2%    -1.0%p
SPL           0.110    0.108
자발적 정지    76.4%    78.0%

물체별        500개    2000개          거리별      2000개  에피소드
  chair       29.0%    30.4%           2-4 m      36.1%     443
  sofa        25.5%    28.2%           4-6 m      29.8%     553
  bed         25.9%    26.1%           6+  m      15.8%    1004
  toilet      24.0%    21.6%
  plant       23.8%    19.0%
  tv_monitor  20.0%    11.7%   <- 8.3%p 하락
논문           11.6%
```

**차이 1.0 %p는 500개의 표준오차(±1.9 %p) 안이다.** 논문과의 배율도 2.17배에서 2.09배로 거의
그대로다. **평가 표본 크기는 격차의 원인이 아니다.**

**단, 물체별은 다르다.** `tv_monitor`가 20.0 → 11.7 %로 떨어졌다. 500개에서 70개뿐이라 과대평가돼
있었다(2,000개에서는 281개). `plant`는 2,000개에서도 84개라 여전히 불안정하다. **전체 성공률은
500개로 충분했지만 물체별 수치는 표본이 적은 항목에서 흔들린다** — 앞으로 물체별을 인용할 때
붙여야 할 단서다.

**아직 닫히지 않은 것**: 지도교수의 요구는 "habitat-lab 표준 평가 경로 + 전체 validation set"
이었고, 여기서 닫힌 것은 뒤쪽뿐이다. 평가 루프는 여전히 우리 것이며 그것은 실험 1-A의 몫이다.
CoT 전체 2,000개는 에피소드당 약 150초라 83 GPU시간이므로 1-A와 묶어 한 번에 도는 것이 맞다.

## V.9 1-A 예비 실행과 진단 — 우리 루프에 계통적 편향은 없다 (2026-08-28)

`validate_env.py`로 같은 이미지 인코더 체크포인트를 habitat-lab의 `Env` 안에서 굴렸다.

**6 에피소드 예비 실행에서 성공률이 16.7 % 대 33.3 %로 갈렸다.** 6개는 표본이 아니라 한 건이
16.7 %p이므로 그 수치 자체는 의미가 없다. 의미가 있는 것은 **스텝 수**였다 — 같은 정책이 같은
출발 자세에서 380 대 202, 500 대 117 스텝을 갔다. 채점이 아니라 주행이 갈린 것이다.

두 설명이 가능했고 대응이 정반대였다. (1) 두 루프가 정책에 다른 것을 먹인다 — 버그. (2) 같은
것을 먹이는데 argmax가 뒤집혀 폐루프가 증폭한다 — 버그 아님.

**행동열을 하나로 묶어 갈랐다** (`diagnose_env.py`). habitat-lab이 자유롭게 돌린 행동열을 기록해
우리 루프가 그대로 재생하면, 행동 이력이 강제로 같으므로 관측 차이는 루프의 차이가 된다.

```
스텝 0~29   rgb 지문 일치 30/30
            gps 차이     최대 2.38e-07
            나침반 차이   최대 1.19e-07
            로짓 최대차   30번 중 29번이 정확히 0.00e+00 (한 번 1e-4)
            행동         30/30 일치
```

**계통적 차이가 없다.** 두 시뮬레이터가 픽셀 단위로 같은 그림을 렌더하고, 우리 `episodic_pose`
재구현이 habitat의 센서와 float32 정밀도 안에서 같으며, 같은 입력에 같은 로짓이 나온다.
시뮬레이터 하나를 여러 에피소드가 공유하며 자세를 넣었다 빼는 비표준 구조가, 자세 복원과 렌더링
면에서 표준과 구분되지 않는다.

따라서 예비 실행의 차이는 **폐루프 증폭**이다. 스텝 2에서 로짓이 1e-4 갈렸는데 그 자리에서 상위
두 행동이 가까웠다면 argmax가 뒤집히고, 그 뒤 관측이 전부 달라진다. 3-A와 같은 기제이며 그때는
표집이, 여기서는 폐루프가 증폭한다.

**곁가지**: habitat-lab 쪽에서만 `SemanticScene.cpp: SSD Load Failure`가 뜬다(`.basis.scn`).
우리 `make_sim`은 의미 주석을 요청하지 않아 나오지 않는다. rgb가 30/30 일치했으므로 렌더에
영향이 없고, 목표는 에피소드 파일의 관찰 지점으로 주어지므로 채점과도 무관하다.

**smoke test가 값을 했다.** 6 에피소드를 세 번 돌리는 동안 2,000개였으면 시간을 버렸을 오류를
둘 잡았다 — hydra가 `data_path`의 `{split}` 중괄호를 문법으로 해석한 것, 그리고 habitat-lab이
`episode_id`를 0,1,2,…로 다시 매겨 `episode_uid`가 두 쪽에서 달라진 것(495개가 하나도 안 맞았다).
뒤엣것은 설계 실수였다 — `episode_uid`를 만든 이유가 원본 `episode_id`가 식별자 노릇을 못 해서인데
그 쓸모없는 절반을 이름 안에 남겨두었다. 출발 자세 해시만으로 잇도록 고쳤다(`pose_key`).

## V.10 1-A 통과 — 평가 루프에 편향이 없다 (2026-09-02)

같은 이미지 인코더 체크포인트를 habitat-lab의 `Env` 안에서 검증 2,000 에피소드 전부에 대해
굴렸다. 4샤드, 중복 0.

```
habitat-lab 표준 환경   25.6 %   SPL 0.112
우리 평가 루프          24.2 %   SPL 0.108
차이                   +1.4 %p   결합 표준오차 ±1.4 %p  (1.0σ)      통과 (허용 ±3 %p)
```

**전체 수치만 맞은 것이 아니라 구조가 같다.**

```
              lab    우리    차이              lab    우리    차이
  chair      35.0   30.4   +4.6      2-4 m   37.7   36.1   +1.6
  sofa       29.5   28.2   +1.3      4-6 m   32.5   29.8   +2.7
  bed        28.4   26.1   +2.3      6+  m   16.4   15.8   +0.6
  toilet     20.9   21.6   -0.7
  plant      17.9   19.0   -1.1
  tv_monitor 10.7   11.7   -1.0
```

물체 순위가 동일하고 거리에 따라 무너지는 모양도 같다. 차이가 +4.6에서 -1.1까지 양쪽으로
흩어져 계통적 편향이 아니라 표본 요동이다.

**"평가 루프가 관대해서 높다"는 설명이 이 프로젝트의 모든 수치에 대해 닫힌다.** 우리 루프에만
있던 것 — 시뮬레이터 하나를 여러 에피소드가 공유하며 자세를 넣었다 빼는 구조, 500스텝 종료,
`episodic_pose` 재구현, SPL 계산 — 을 전부 표준으로 갈아끼웠는데 1.0σ 안에서 같은 값이 나왔다.

### 실행 중 고친 것 두 가지

**깊이 센서를 껐다.** 처음에는 "stock 설정을 그대로 받아들이는 것이 이 실험의 요점"이라 적고
`rgbd_agent`를 그대로 뒀는데 **틀린 판단이었다.** 논문 부록 C.1은 깊이를 명시적으로 제외하고
우리 `make_sim`에도 깊이 센서가 없으므로, 켜 두면 **비교의 양쪽 모두와 다른** 상태가 된다.
루프의 차이를 재는 실험에 렌더링 차이를 하나 더 얹고 있었다. Success·SPL·DistanceToGoal은
자세와 pathfinder로 계산되므로 측정값은 움직이지 않는다.

**스레드 수를 요청량에 맞췄다.** `--cpus-per-task=8`을 아무것도 강제하지 않아 torch가 노드
크기에서 16/32를 잡았고, 네 샤드가 한 노드에서 248 스레드로 48코어를 서로 뺏었다.

```
                에피소드당      실행대기 r    idle    GPU
수정 전         20.8~31.2초    97 / 48코어    0%     5~19%
수정 후         10.7~11.9초    32 / 48코어   33%    19~34%
```

**2.2배 빨라졌다** (3~4.5시간 -> 1시간 34분). 두 수정을 함께 넣어 각각의 몫은 가르지 못했다.
`slurm/validate_env.slurm`에 `OMP_NUM_THREADS=8`, `validate_env.py`에 `torch.set_num_threads`가
들어갔다 — 환경변수만으로는 torch가 이미 초기화된 뒤 무시할 수 있어 코드에서도 못박았다.

### 지도교수 조언 대비 현황

```
① 표준 평가 경로 + 전체 validation set     닫힘 (V.8 + V.10)
② 중간 checkpoint 수렴 비교                닫힘 (V.7, epoch 10에서 이미 논문 초과)
③ policy memory 크기                       부록 X, 설계만
③ demonstration 재생 방식                  부록 Y.4, 설계만
```

**먼저 하라고 지목된 두 가지가 모두 끝났고 둘 다 "원인이 아니다"로 나왔다.**

# 부록 X-R. 용량 실험 결과 (2026-09-02)

설계는 부록 X. 이미지 인코더 조건에서 순환 크기만 바꾸고 나머지를 전부 고정해 40 epoch씩
학습한 뒤 같은 500 에피소드로 평가했다.

```
              LSTM      파라미터    성공률   표준오차   SPL     자발적 정지   논문 대비
L  (시드 0)   2048x2   78,848,548   25.2%   ±1.94p   0.110    76.4%      2.17배
L' (시드 1)   2048x2   78,848,548   22.6%   ±1.87p   0.094    74.0%      1.95배
M            1024x2   36,360,740   19.2%   ±1.76p   0.088    57.6%      1.66배
S             512x1   22,321,428   13.0%   ±1.50p   0.056    57.8%      1.12배
XS         순환 없음   19,042,836    1.4%   ±0.53p   0.005    12.6%      0.12배
논문           미명시                11.6%
```

## X-R.1 판정 — 기준 ② 충족

```
시드 |L - L'|          +2.6%p   ±2.70p   1.0σ    노이즈와 구분되지 않음
크기 |L평균 - M|        +4.7%p   ±2.22p   2.1σ
크기 |L평균 - S|       +10.9%p   ±2.02p   5.4σ
S - 논문               +1.4%p   ±1.67p   0.8σ    구분되지 않음
```

**③ 검정력** — 크기 효과(10.9 %p)가 시드 효과(2.6 %p)의 **4.2배**. 통과.

**② 용량이 살아 있는 설명** — S가 12~16 % 구간에 들어왔고 논문과의 차이는 0.8σ로 구분되지
않는다. **격차 12.3 %p 중 10.9 %p, 89 %를 순환 크기가 덮는다.**

**말할 수 있는 것**: 논문이 침묵한 순환 크기를 그럴듯한 범위에서 줄이면 논문 수치가 재현된다.
다른 것은 아무것도 바꾸지 않았다 — 같은 데이터, 같은 40 epoch, 같은 학습률과 스케줄, 같은 손실
가중치. 우리 2048x2는 논문의 "[43]과 같은 LSTM"을 VC-1을 거쳐 PIRLNav까지 두 단계 거슬러 얻은
값이고, 저자가 더 작은 것을 썼다면 11.6 %가 자연스럽게 나온다.

**말할 수 없는 것**: 논문이 512x1을 썼다는 증명은 아니다. 그런 크기가 존재한다는 것뿐이며,
논문에 하이퍼파라미터 표도 코드도 없어 확인할 길이 없다. 부록 X.7의 교란도 그대로다 — 작은
망에는 다른 학습률이 맞을 수 있는데 레시피를 고정했다. 시험한 명제가 "논문이 **같은 레시피로**
작은 망을 썼다면"이므로 고정이 맞지만, S의 학습 정확도가 76.1 %로 L의 95.2 %보다 한참 낮다는
사실은 남는다. 논문 정책도 그렇게 덜 맞춰졌는지는 알 수 없다(학습 정확도를 보고하지 않는다).

**이것은 이미지 인코더 조건이다.** CoT에도 같은 논리가 적용될 것으로 보이나 **측정이 아니라
추론**이다.

## X-R.2 시드 대조가 실험의 한계를 드러냈다

L' 없이 L만 있었다면 |L - M| = 6.0 %p를 용량 효과로 단정했을 것이다. 실제로는 시드만 바꿔도
2.6 %p 움직이고, M을 어느 L과 견주느냐에 따라 6.0 %p(2.3σ)와 3.4 %p(1.3σ)로 갈린다.

**부록 X.5의 검정력 계산이 부분적으로 틀렸다.** "가르려는 차이 13.6 %p는 표준오차의 7배라
500개면 충분"이라 적었는데, 그것은 *S가 논문 수준까지 무너지는가*에는 맞고(5.4σ), *L·M·S를
서로 순위 매기는 데*는 부족하다 — 인접한 크기 사이는 5 %p 규모라 2σ 언저리다. 인접 비교가
필요하면 2,000 에피소드로 다시 재야 한다.

## X-R.3 순환의 몫은 크기의 몫과 다른 종류다

```
크기를 18배 줄임   25.2% -> 13.0%   (-12.2%p)
순환을 없앰        13.0% ->  1.4%   (-11.6%p)
```

자발적 정지가 57.8 %에서 12.6 %로 무너진다. XS의 **학습** 정지 정확도는 77.5 %로 S(78.6 %)와
거의 같았는데 낯선 건물에서는 멈추지 못한다. 언제 멈출지는 "여기 오기까지 무엇을 보았는가"에
달려 있고 기억 없이는 알 수 없다.

**대비 하나가 특히 선명하다.**

```
토큰 0 대조군   시각 없음 + LSTM 2048x2   ->  3.4%
XS             시각 있음 + 순환 없음      ->  1.4%
```

**시각을 다 주고 기억을 뺀 정책이, 시각을 빼고 기억만 남긴 정책보다 못하다.** 부록 V.4의 PCA
그림이 보인 대로 표현은 방을 구분하지만, 그것을 시간에 걸쳐 쌓지 못하면 쓸모가 크게 줄어든다.

## X-R.4 실행 중 고친 것

**`policy.py`의 크기를 인자로 뺐다.** `LSTM_HIDDEN`·`LSTM_LAYERS`·`SIDE_EMBED_DIM`이 모듈
상수였다. 체크포인트에도 저장한다 — `evaluate.py`가 체크포인트에서 정책을 재구성하므로 적지
않으면 기본 크기로 만들어 가중치 적재에서 실패한다. `lstm_layers=0`은 순환을 아예 만들지 않고
요약에서 행동 헤드로 직결한다(Listing 1의 형태).

**`stage_locally`의 복사를 원자적으로 바꿨다.** 네 실행이 한 스테이징 디렉터리를 공유하는데
`shutil.copyfile`이 제자리에 쓰고 건너뛰기 판정이 "존재 + 크기 일치"였다. 한쪽이 채우는 중인
파일을 다른 쪽이 완성된 것으로 보고 40 epoch 내내 잘린 배열을 읽어도 예외가 나지 않는다.
임시 이름으로 복사한 뒤 `os.replace`로 옮긴다.

**`evaluate.py`가 순환 없는 정책을 못 받았다.** 배치 롤아웃이 LSTM 상태를 에피소드별로 잘라
나눠 갖게 하는데, XS는 `(logits, None)`을 돌려주어 `None`을 자르려다 죽었다. 정책 단독
순전파는 미리 시험했고 `validate_env.py`는 에피소드를 하나씩 처리해 `None`이 통과했으므로,
**두 평가 경로가 갈라지는 지점을 시험하지 않은 것**이 원인이다.

## X-R.5 노드 배치가 학습 시간을 두 배로 갈랐다

작은 모델 둘을 메모리 적은 노드에 보낸 것이 실수였다.

```
              노드         총 소요   읽기   계산
L' 2048x2   compute08     244s     2s   242s     메모리 128 GB
M  1024x2   compute08     184s     7s   177s
S   512x1   compute10     503s   338s   166s     메모리  64 GB
XS  512x0   compute10     532s   402s   130s
```

임베딩이 80 GB다. compute08은 페이지 캐시에 통째로 들어가 읽기가 2~7초로 소멸하고, compute10은
들어가지 않아 매 epoch 다시 읽는다. **계산만 보면 S(166초)가 L'(242초)보다 빠른데 총합은 두 배**다.
CPU 경합을 피하려 2대2로 나눌 때 두 노드의 메모리가 두 배 차이라는 것을 셈에 넣지 않았다.

## X-R.6 VC-1 원문 대조 (2026-09-03)

용량 실험이 끝난 뒤 VC-1([arXiv:2303.18240](https://arxiv.org/abs/2303.18240)) PDF를 받아
`docs/papers/VC1_arxiv_2303.18240.pdf`에 두고, 이 프로젝트가 그 논문을 근거로 정한 값들을
하나씩 대조했다.

**원문과 일치 (부록 A.2·A.3)**

```
AdamW, weight decay 1e-6      "we employed a weight decay of 10−6 ... with the AdamW optimizer"
400M / 25,000 / 512 env       "approximate total of 400 million steps, utilizing 25,000
                               updates and 512 parallel environments"
평균 159 스텝                  "an average of ∼159 steps per episode"        우리 실측 159.1
77k 시연 / 80 장면 / 12.1M     "77k demonstrations for 80 scenes ... ∼12.1 million steps"
높이 0.88 반경 0.18 640x480 79° 그대로
500 스텝 제한                  "within 500 steps"
HM3D v0.1 val, 검증 20장면     "v0.1 HM3D-SEM VAL split ... 20 validation scenes"
성공 판정                      "stopping within 0.1m of a viewpoint that is (a) within 1m of any
                               instance of the target object and (b) from which the object is visible"
LSTM 2층                       "which is a 2-layer LSTM for navigation"
```

배치 16,384도 확인된다. 나눗셈만 하면 400M / 25,000 = **16,000**이지만, 512 x 32 = **16,384**가
25,000회로 409.6M이 되어 "approximately 400 million"에 더 맞는다.

**틀렸던 것 — 인용 사슬** (X.2에 정정을 적었다). 층수 2는 VC-1이 명시하고, **폭 2048은 VC-1에
없으며 PIRLNav은 구조의 출처로 인용되지 않는다.**

**새로 알게 된 것 — 체크포인트 선택.** VC-1 부록 A.3:

> "we evaluated checkpoints after every 10M steps and **only reported metrics for the
> checkpoints with the highest validation success rate**"

**VC-1은 최고 체크포인트를 보고한다.** 우리는 최종 40 epoch을 보고한다. PR2L이 "the same
optimizer, scheduler, and associated hyperparameters as Majumdar et al."이라 했을 때 평가
규약까지 물려받았는지는 밝히지 않았고, Table 3 각주의 "VC-1 policies' performance saturated at
120"은 학습 중 성능을 추적했음을 보인다.

**이것은 격차를 설명하지 않는다** — 최고값 보고는 상대 수치를 *높이는* 방향이다. 우리 L도
epoch 30에서 포화해(25.2 → 25.2) 최고와 최종이 같다. 그래도 비교할 때 밝혀야 할 규약 차이다.

**곁가지 — 같은 인코더가 60.3 %와 13.6 %로 갈린다.**

```
VC-1 frozen ObjectNav      60.3 %   데이터 10배 + 이미지 증강 (VC-1 Table 2)
VC-1 adapted               67.7 %
PIRLNav (prior best)       70.4 %
PR2L + CoT (논문)          41.9 %
VC-1 + Patch (PR2L 재현)   13.6 %   데이터 1/10, 증강 없음, 40 epoch
```

표현이 같은데 학습 설정만으로 4.4배가 갈린다. **이 과제에서 표현보다 정책·학습 쪽 변수가 더
크게 좌우한다는 사례가 논문들 안에 이미 있으며**, 우리가 용량 하나로 25.2 → 13.0 %를 움직인
것과 같은 성격이다.

## X-R.7 정정의 정정 — 2048은 유추가 아니라 VC-1이 실제로 쓴 값이다 (2026-09-04)

X-R.6과 X.2에 "폭 2048은 추적이 아니라 유추"라고 적었다. **본문만 놓고 보면 맞지만 코드까지
보면 틀렸다.** VC-1 공개 저장소 `facebookresearch/eai-vc`의 ObjectNav 학습 설정
(`cortexbench/habitat_vc/configs/experiments/objectnav_il.yaml`):

```yaml
  RGB_ENCODER:
    hidden_size: 512          <- 시각 인코더 출력 폭. 순환망이 아니다
  STATE_ENCODER:
    hidden_size: 2048
    rnn_type: LSTM
    num_recurrent_layers: 2
```

**층수 2 · 폭 2048 · 셀 LSTM 셋 다 우리 L과 같다.** 논문 본문(A.2)이 층수만 적고 폭을 생략했을
뿐, 저자들이 쓴 값 자체는 우리가 채택한 것과 일치한다.

**PIRLNav 경유는 불필요했고, 게다가 다른 경로였다.** PIRLNav §3과 그 출처 OVRL[30]
(arXiv:2204.13226)은 ObjectNav에 **2-layer 2048-d GRU**를 쓴다. OVRL은 과제별로 갈라서
ImageNav에는 **2-layer 512-d LSTM**을 쓴다. 우리가 가져온 2048은 **GRU 쪽 숫자**였고,
VC-1은 같은 폭의 **LSTM**을 쓴다. **숫자가 우연히 같았을 뿐 계보가 다르다.**

`RGB_ENCODER`의 512를 순환망 폭으로 잘못 읽었다면 정반대 결론에 갔을 것이다. 같은 파일 안에
`hidden_size`가 두 번 나오고 블록만 다르다.

### 이것이 X-R의 ② 판정에 미치는 영향

```
L  2048x2 LSTM   VC-1 코드와 일치        <- 논문이 따랐다고 밝힌 설정
W  2048x1        깊이만 벗어남
N   512x2        폭만 벗어남
S   512x1        둘 다 벗어남 -> 13.0 % ~= 논문 11.6 %
```

**"논문이 우리보다 작은 LSTM을 썼을 것"이라는 해석이 약해진다.** VC-1이 2048x2 LSTM을 썼고
PR2L 본문 4.2가 "the same LSTM-based recurrent architecture used by that work"라 했으므로,
PR2L도 2048x2였을 개연성이 높다. 그러면 격차는 용량이 아닌 다른 데서 와야 한다.

**다만 확정은 아니다.** PR2L은 코드를 공개하지 않았고, 부록 C.2에서 VC-1과 다르게 한 것 일곱
가지를 열거하면서 순환망은 넣지 않았다. "언급하지 않았으니 그대로"가 자연스러운 읽기이나
증명은 아니다.

**X-R의 실험 결과 자체는 그대로 유효하다** — 용량을 줄이면 성공률이 단조로 내려가고 512x1에서
논문 값에 닿는다는 것은 측정된 사실이다. 바뀐 것은 그 사실의 **해석**이다. "논문이 작은 망을
썼다"의 근거로 쓰기 어려워졌고, 대신 **"이 과제에서 순환 용량이 성공률을 크게 좌우한다"**는
독립적인 발견으로 남는다.

## X-R.8 폭과 깊이의 분해 — 폭이 전부다 (2026-09-04)

X-R의 S(512x1)는 L(2048x2)에서 **폭과 깊이를 동시에** 줄인 실행이라, 13.0 %가 어느 쪽에서
왔는지 말할 수 없었다. 두 실행을 추가해 2x2 격자를 채웠다. 부가 임베딩은 32로 고정했다
(S만 16이었으므로 격자에서 제외되는 값이다).

```
              2층      1층    깊이 효과
폭 2048     25.2%   24.4%    +0.8%p   0.3σ
폭  512     14.0%   13.0%    +1.0%p   0.5σ
────────────────────────────────────
폭 효과    +11.2p  +11.4p    4.5σ / 4.7σ
```

**두 행의 폭 효과가 11.2 %p와 11.4 %p로 사실상 같고, 두 열의 깊이 효과가 0.8 %p와 1.0 %p로
둘 다 노이즈 안이다.** 교호작용이 없다. **층을 하나 빼는 것은 사실상 공짜다** — LSTM
파라미터를 59.8 M에서 26.2 M으로 절반 넘게 줄이는데 성공률은 0.3σ 내려간다.

### 파라미터 수로는 설명되지 않는다

```
              LSTM     학습 손실   성공률
W  2048x1   26.2M     0.5899    24.4%
M  1024x2   17.3M     0.7262    19.2%    파라미터 1.5배 적은데 5.2%p 낮다 (2.0σ)
N   512x2    5.5M     0.9389    14.0%
S   512x1    3.3M     1.0171    13.0%    파라미터 1.7배 적은데 1.0%p 낮다 (0.5σ)
```

**학습 손실은 파라미터 순서를 정확히 따르는데(0.59 < 0.73 < 0.94 < 1.02) 성공률은 따르지
않는다.** N과 S는 파라미터가 1.7배 차이인데 성공률이 붙고, W는 M보다 폭이 같은 L 쪽에 붙는다.
epoch 20 시점에 "손실이 파라미터 순서를 정확히 따른다"를 보고 성공률도 그러리라 적었는데
틀렸다. **용량이 아니라 폭이 결정한다.**

**해석은 추측이다.** 한 스텝에서 은닉 상태가 담는 정보량이 폭이고, 층을 쌓는 것은 그 상태를 더
복잡하게 변환하는 것이다. 탐색 과제에서는 "어디를 이미 지나왔는가"를 담을 자리가 필요하지
복잡한 변환이 필요한 것이 아닐 수 있다. XS(순환 없음, 1.4 %)가 이 읽기와 맞는다 — 기억할 자리가
0이면 무너진다. 다만 이것은 결과에 붙인 이야기이지 시험한 명제가 아니다.

### 실행 중 있었던 일

슬럼이 W를 5시간 만에 다른 노드로 재큐잉했고, 슬럼 스크립트에 `--resume`이 없어 **epoch 1부터
다시 시작했다.** 체크포인트에 옵티마이저 모멘트·`seen`·`history`가 다 있었는데도 쓰지 않았다.
스크립트를 고쳤다 — `--resume`은 플래그가 아니라 경로를 받는 인자이므로(`type=Path`), 파일이
있을 때만 붙인다.

또 두 실행을 `--nodelist=bml-compute08`로 묶어 던졌다가 **20시간을 대기했다.** 다른 노드에
GPU 4장이 놀고 있었다. 메모리 125 GB 노드에서 epoch이 2.7배 빠르다는 실측이 근거였으나,
20시간 대기는 그 이득을 훨씬 넘는다. 노드 고정은 자리가 확실할 때만 쓸 것.

---

# 부록 Z-③. 지도교수 조언 ③에 대한 답 (2026-09-05)

지도교수는 네 가지를 순서대로 보라 했고, ①②는 부록 V에서 닫혔다. ③은 두 항목이었다.

## Z-③.1 policy memory 크기 — 원인이 아니다

**실험은 답했다.** 순환망 폭이 성공률을 크게 좌우한다.

```
폭 2048 -> 512    25.2% -> 14.0%    -11.2%p   4.5σ
깊이 2층 -> 1층    영향 없음          0.3~0.5σ
순환 제거          1.4%              바닥
```

**그러나 이것이 논문과의 격차를 설명하지는 않는다.** VC-1 공개 코드
(`cortexbench/habitat_vc/configs/experiments/objectnav_il.yaml`)가
`STATE_ENCODER: {hidden_size: 2048, rnn_type: LSTM, num_recurrent_layers: 2}`이고, PR2L 본문
4.2가 "the same LSTM-based recurrent architecture used by that work"라 했다. **논문도 2048이었을
개연성이 높다.** 그러면 폭은 우리와 논문을 가르는 변수가 아니다.

축소 실행이 논문 값에 닿는 것은 사실이나(S 13.0 %, N 14.0 %, 논문 11.6 %), 그것은 "우리가 폭을
줄이면 논문 수치가 나온다"이지 "논문이 폭을 줄였다"가 아니다. **원인 후보에서 내린다.**

**남는 것은 독립적인 발견이다** — 이 과제에서 정책의 순환 폭이 표현만큼이나 성공률을 좌우한다.
CoT 60.2 % 대 이미지 인코더 24.2 %(표현 차이 36 %p)와, 폭 2048 대 512(11.2 %p)를 나란히 놓으면
정책 쪽 변수의 크기를 가늠할 수 있다.

## Z-③.2 demonstration 재생 방식 — 아직 안 함

부록 Y.4의 설계 그대로다. 내비메시 재계산을 끄고 렌더한 데이터로 이미지 인코더를 다시 학습해
11.6 % 부근이 나오는지 본다. 렌더 12 h + 학습 7 h + 평가 1 h.

**Y.2에서 이미 약해진 후보다** — "7.5 %"를 "시연의 92.5 %가 망가졌다"로 읽은 것이 틀렸고, 실제로는
전진 79.7 % · 회전 98.8/99.7 %에 궤적 누적 오차 중앙값 0.173 m다. 성공 반경 0.1 m라 17 cm만
밀려도 실패로 찍히는 것이지 관찰-행동 대응이 무너진 것이 아니다.

## Z-③.3 그래서 지금 남은 후보

①②③이 모두 닫히거나 약해졌다.

```
평가 프로토콜      닫힘 (1-A, 1.0σ)
평가 표본 수       닫힘 (2,000개, 1.0%p 차)
수렴 정도          닫힘 (epoch 10에서 이미 논문 초과)
정책 용량          원인 아님 (논문도 2048로 보임)
재생 품질          약함, 미실행
```

**남은 것은 우리가 아직 재현하지 않은 조건들과, CoT 경로에만 있는 변수들이다.**

1. **CoT 조건에서 용량 실험을 반복하지 않았다.** 위 결론은 전부 이미지 인코더에서 얻었다.
   보고서의 주 수치는 CoT(60.2 % 대 41.9 %)이고, 그 조건으로 옮겨가는지는 추론이다.
2. **BOS 한 칸 밀림(Z.1)이 CoT 수치에 그대로 남아 있다.** 코드는 고쳤으나 재인코딩하지 않았고,
   수정 전후 표현이 결정적 블록 기준 코사인 0.72로 다르다. 방향은 성능을 **낮추는** 쪽이라
   격차를 키우지 줄이지 않는다.
3. **논문이 보고한 조건 다섯 중 셋을 우리는 돌리지 않았다** — PR2L without CoT(27.8 %),
   VC-1 + CLS(6.8 %), VC-1 + Patch Embeds(13.6 %). 특히 VC-1 + Patch는 우리 이미지 인코더와
   구조가 가장 가깝고 같은 LSTM을 쓰므로, 우리 파이프라인으로 재현해 13.6 %가 나오는지 보는
   것이 남은 후보 중 가장 직접적인 검사다.

# 부록 C. 학습 레시피 차이 — 기울기 클리핑 (2026-09-08)

## C.1 발견

VC-1 공개 저장소의 ObjectNav 모방학습 설정
(`cortexbench/habitat_vc/configs/experiments/objectnav_il.yaml`):

```yaml
  BehaviorCloning:
    lr: 0.001
    encoder_lr: 0.0001
    eps: 1.0e-5
    wd: 1.0e-6
    clip_param: 0.2          # PPO의 ε. BC에는 비율이 없어 쓰이지 않는 잔재
    max_grad_norm: 0.2       # 기울기 노름 클리핑 — BC에서 실제로 작동한다
    use_linear_lr_decay: True
```

**우리 `train.py`에는 기울기 클리핑이 없다.** `optimiser.step()` 앞에 `clip_grad_norm_` 호출이
없고, 역전파한 기울기를 그대로 적용한다.

0.2는 통상값(0.5~5.0)보다 훨씬 빡빡하다.

**곁가지로 AdamW의 eps도 다르다** — VC-1은 `1.0e-5`, 우리는 설정하지 않아 PyTorch 기본값
`1e-8`이다. 1,000배 차이이며 eps가 크면 분모가 커져 유효 학습률이 낮아진다. 같은 방향이지만
클리핑보다 영향은 작을 것으로 본다.

`lr 1e-4`(PR2L이 VC-1의 1e-3에서 바꿈)와 `wd 1e-6`은 우리와 일치한다.

## C.2 왜 우리 성공률을 올렸을 수 있나

> **2026-09-08 정정.** 이 절은 처음에 "기울기 노름을 0.2로 자르면 한 번의 갱신이 움직일 수
> 있는 거리에 상한이 걸린다"고 적었다. **Adam에서는 그렇지 않다.** 아래가 옳은 설명이며,
> 이 정정이 C1에 대한 예측을 낳았고 그 예측이 맞았다(C.6).

Adam의 갱신은 `lr × m̂/(√v̂ + ε)`이다. `m`은 기울기의 이동평균, `v`는 기울기 제곱의
이동평균이므로, **기울기를 전부 `c`배 하면 분자와 분모가 함께 `c`배가 되어 비율이 변하지
않는다.** 즉 Adam은 기울기의 절대 크기를 스스로 정규화하며, 보폭은 크기가 아니라 방향의
일관성이 정한다. 클리핑은 기울기를 상수배 축소하는 연산이므로 **`ε`이 무시할 만큼 작으면
갱신을 거의 바꾸지 못한다.**

`ε`이 그 성질을 깨는 지점이다.

```
√v̂ ≫ ε    갱신 ≈ lr × m̂/√v̂     기울기 크기와 무관       ← Adam 본래 거동
√v̂ ≪ ε    갱신 ≈ lr × m̂/ε      기울기 크기에 비례       ← SGD에 가까워짐
```

`ε`을 1e-8에서 1e-5로 1,000배 키우면 두 번째 영역이 넓어지고, **그제서야 클리핑이 갱신을
실제로 자른다.** 두 설정은 독립적으로 더해지는 것이 아니라 **상호작용한다** — `ε`은 클리핑이
물릴 수 있는 조건을 만들고, 클리핑은 그 조건 아래서 보폭을 깎는다.

따라서 우리 쪽이 더 깊이 학습된다는 방향 자체는 유지되지만, 원인은 "상한이 없어서"가 아니라
**두 설정이 함께 빠져 있어서**다. 실제로 우리 학습 정확도는 93~95 %에 이르렀다.

**이 후보는 지금까지 세운 필터를 통과하는 몇 안 되는 것이다.**

* 학습 레시피이므로 **CoT와 이미지 인코더에 똑같이 작용**한다. 두 조건이 나란히 2배 오른
  관측과 맞는다.
* 방향이 맞다 — 클리핑이 없으면 더 학습되고, 더 학습되면 성공률이 오른다.
* **PR2L이 물려받았을 근거가 명시적이다.** 부록 C.2가 VC-1과 다르게 한 것 일곱 가지를
  열거하는데 클리핑은 없고, 본문은 "the same optimizer, scheduler, and **associated
  hyperparameters** as Majumdar et al."이라 적는다. `max_grad_norm`은 그 안에 든다.

용량 가설과 대비된다 — 용량은 VC-1 코드가 2048을 쓰는 것이 확인되어 후보에서 내려갔지만,
클리핑은 **VC-1이 쓰고 우리가 쓰지 않는다**는 것이 양쪽 모두 확인된다.

## C.3 설계

조건은 **이미지 인코더**. 이유는 부록 X.3과 같다 — 재인코딩 불필요, 80 GB라 빠름, VLM
체크포인트 변종으로 설명되지 않는 유일한 격차.

| 실행 | max_grad_norm | AdamW eps | 비고 |
|---|---|---|---|
| 기준 | 없음 | 1e-8 | 현재. **500 에피소드 25.2 % 이미 있음** (2000 에피소드로는 24.2 %) |
| **기준′** | 없음 | 1e-8 | **시드만 다른 재실행** |
| **C1** | 0.2 | 1e-8 | 클리핑만 |
| **C2** | 없음 | 1e-5 | eps만 |
| **C3** | 0.2 | 1e-5 | 둘 다 = VC-1 레시피 |

**C1과 C2를 나누는 것이 핵심이다.** 한 번에 둘을 바꾸면 어느 쪽이 효과인지 가릴 수 없다.
부록 X의 S가 폭과 깊이를 동시에 줄여 나중에 2x2 격자를 다시 채워야 했던 것과 같은 실수다.

**기준′도 빠뜨릴 수 없다.** 용량 실험에서 시드만 바꿔도 2.6 %p(1.0σ)가 움직였다. 그보다 작은
차이는 읽을 수 없다.

## C.4 판정 기준 (실행 전에 못박는다)

```
③ |기준 - 기준′| 이 |기준 - C3| 만큼 크다   -> 검정력 부족. 아무것도 읽을 수 없다   <- 먼저
① C3가 11.6 % 부근(±3 %p)                  -> 레시피 차이가 격차를 설명한다
② C1만 크게 떨어진다                        -> 클리핑이 주범
② C2만 크게 떨어진다                        -> eps가 주범
④ 셋 다 24 % 근처                          -> 레시피는 원인이 아니다. 후보에서 제외
```

**실행 전에 못박은 예측** (`merge_recipe.py` 상단 주석에 기록): C.2의 메커니즘이 옳다면
**C3 ≫ C1**이어야 한다. 즉 클리핑만 켠 C1은 기준과 구분되지 않아야 한다. C1이 C3만큼
떨어지면 C.2의 읽기가 틀린 것이다. 이 예측은 ②의 "클리핑이 주범"과 정면으로 어긋나므로,
어느 쪽이든 반증 가능한 형태다.

## C.5 비용

```
학습 4회 (기준′·C1·C2·C3) x 약 4시간   GPU 4장 병렬이면 4시간
평가 4회 x 500 에피소드                약 1.5시간
합계 약 6시간
```

코드 변경은 세 줄 — `optimiser.step()` 앞의 `clip_grad_norm_`, `AdamW(..., eps=...)`, 그리고 두
값을 CLI 인자와 체크포인트에 넣기(`evaluate.py`가 체크포인트에서 정책을 재구성하므로).

## C.6 결과 — 다섯 실행 완료 (2026-09-09)

전부 이미지 인코더 조건, 500 에피소드 층화 부분집합, 40 epoch 최종 체크포인트.

| 실행 | clip | eps | 시드 | 성공률 | SPL | 자발적 정지 |
|---|---|---|---|---|---|---|
| 기준 | 없음 | 1e-8 | 0 | **25.2 %** | 0.110 | 76.4 % |
| 기준′ | 없음 | 1e-8 | 1 | **24.8 %** | — | — |
| C1 | 0.2 | 1e-8 | 0 | **25.0 %** | 0.112 | 73.2 % |
| C2 | 없음 | 1e-5 | 0 | **25.0 %** | 0.115 | 72.8 % |
| C3 | 0.2 | 1e-5 | 0 | **13.6 %** | 0.063 | 52.8 % |
| 논문 Table 3 | — | — | — | 11.6 % | | |

### 판정 (C.4의 순서대로)

**③ 통과.** 시드만 바꾼 차이 0.4 %p에 대해 레시피 차이는 11.6 %p — **29배**다. 나머지를
읽을 수 있다.

```
시드   |기준 − 기준′|    +0.4 %p   ±2.74p   0.1σ
레시피 |기준 − C3|      +11.6 %p   ±2.47p   4.7σ
C3 − 논문                +2.0 %p   ±1.69p   1.2σ
```

**① 성립.** C3(13.6 %)와 논문(11.6 %)은 **1.2σ**로 구분되지 않는다. 기준에서 논문까지의
격차 13.6 %p 중 **11.6 %p (85 %)** 를 이 레시피 하나가 덮는다.

**② 분해 — 순수한 상호작용이다.**

```
클리핑만  +0.2 %p   0.1σ    ← 구분 불가
eps만     +0.2 %p   0.1σ    ← 구분 불가
단독의 합 +0.4 %p
둘 다    +11.6 %p   4.7σ    ← 단독 합의 29배
```

**두 설정 중 어느 것도 혼자서는 아무 일도 하지 않는다.** 함께 켜야만 효과가 나타나며, 이는
C.2가 실행 전에 예측한 메커니즘 그대로다 — `ε`이 커야 Adam의 분모에서 `√v̂`을 밀어내고,
그제서야 갱신이 기울기 크기에 비례하게 되어 클리핑이 물린다. 예측은 `merge_recipe.py`
주석에 실행 전 기록돼 있고, 원래의 판정 기준 ②("클리핑이 주범" / "eps가 주범")를 **양쪽 다
기각**하는 형태였으므로 반증 가능했다.

학습 곡선도 같은 답을 냈다(동일 epoch 22):

```
기준    손실 1.0200   정확도 76.7 %
C1      손실 1.0430   정확도 76.3 %     클리핑만 — 거의 동일
C3      손실 1.3532   정확도 70.3 %     둘 다 — 확연히 다름
```

### 이것이 뜻하는 것

이미지 인코더 조건에서 **우리가 논문의 2.17배였던 이유는 VC-1의 학습 레시피 두 항목을 함께
빠뜨렸기 때문이다.** 되돌려 놓으면 논문 수치가 나온다.

이 후보는 지금까지 조사한 것들과 성격이 다르다. 부록 X의 용량 가설은 수치로는 비슷하게
착지했지만 VC-1 코드가 2048을 쓰는 것이 확인되어 문서적 근거가 무너졌다. 여기서는 반대로
**VC-1이 쓰고 우리가 쓰지 않았다는 것이 양쪽 모두 코드로 확인**되고, 메커니즘이 사전 예측을
낳았으며, 그 예측이 맞았다.

**주의 — 기준′은 GPU가 다르다.** 기준은 3090, 기준′은 5090에서 학습했으므로 시드와 GPU가
섞여 있다. 이는 측정된 잡음을 **넓히는** 방향이므로 ③을 느슨하게가 아니라 엄격하게 만든다.
0.4 %p로 통과했으니 시드 단독 잡음은 그보다도 작다.

### 남은 검사 — CoT 조건

이 결론은 아직 **이미지 인코더 한 조건**에서만 확인됐다. 레시피는 조건과 무관한 학습 설정
이므로 CoT에도 같게 작용해야 하는데, 비율이 맞지 않는다.

```
                이미지    CoT     비율
우리            25.2 %   60.2 %   2.39배
논문            11.6 %   41.9 %   3.61배
```

레시피 효과가 조건에 무관하게 균일하다면(×0.540) CoT는 **32.5 %** 로 내려가야 하고, 이는
논문의 41.9 %와 맞지 않는다. 따라서 CoT에 레시피를 적용하는 실험은 확인이 아니라 **반증
가능한 검사**다.

```
CoT ≈ 32 %   ->  레시피는 균일하게 작용한다. CoT 격차에는 별도 원인이 더 있다
CoT ≈ 42 %   ->  레시피가 조건별로 다르게 작용한다. 예측이 틀렸고 설명을 다시 짜야 한다
```

## C.7 CoT·no-CoT에 레시피를 적용한 결과 (2026-09-14)

두 조건 모두 `cot_fixed`/`nocot` 인코딩(BOS 수정 후), 클리핑 0.2 · eps 1e-5, 40 epoch, 500
에피소드 층화 부분집합. 집계는 `merge_campaign.py`. 두 실행의 샤드 파일이 에피소드 목록과
필드 구성을 공유하므로, 합산 전에 md5로 서로 다른 파일임을 확인했다.

| 조건 | 성공률 | SPL | 자발적 정지 | 논문 Table 3 | 논문과의 차이 |
|---|---|---|---|---|---|
| CoT + 레시피 | **47.4 %** | 0.256 | 55.4 % | 41.9 % | +5.5 %p, 2.2σ |
| no-CoT + 레시피 | **46.6 %** | 0.248 | 54.6 % | 27.8 % | **+18.8 %p, 7.7σ** |
| CoT − no-CoT | **+0.8 %p, 0.3σ** | | | +14.1 %p | |

### ① 곱셈 모형 예측은 반증됐고, 덧셈 모형이 맞았다

C.6이 세운 예측(×0.540 → 32.5 %)은 47.4 %와 **4.9σ** 떨어져 기각된다. 대신 레시피가 깎는
**절대량**이 두 조건에서 같다.

```
              레시피 없음   적용 후    절대 효과
이미지 인코더   25.2 %  →   13.6 %    −11.6 %p
CoT            60.2 %  →   47.4 %    −12.8 %p      두 효과의 차이 1.2 %p, 0.3σ
```

클리핑과 eps는 학습이 얼마나 진행되는지를 제한하는 설정이지 표현의 품질에 비례해 작용할
이유가 없으므로, 덧셈 모형이 메커니즘과도 맞는다. 곱셈 모형을 고른 것은 근거 없는 선택이었다.

### ② CoT는 no-CoT보다 나을 것이 없다 — 이것이 새로운 격차다

논문은 "Why or why not?"을 붙이면 성공률이 27.8 % → 41.9 %로 **14.1 %p** 오른다고 보고한다.
PR2L의 핵심 주장이다. 우리 재현에서는 **0.8 %p (0.3σ)** 로 사라진다. no-CoT가 논문보다
**18.8 %p (7.7σ)** 높고, 이것이 레시피 이후 남은 격차의 대부분이다.

샤드별로 봐도 네 샤드 모두 두 조건이 붙어 있다(36.6/33.3, 49.6/52.0, 49.2/53.3, 50.8/50.8).
물체별·거리별 분해도 같다.

```
거리     2-4 m          4-6 m          6+ m
CoT      68 % (111)     60 % (133)     32 % (256)
no-CoT   72 % (111)     59 % (133)     29 % (256)
```

### ③ 유력한 원인 — 최소 생성 길이가 no-CoT를 CoT로 만든다

인코딩 매니페스트에 저장된 표본 답변을 보면, **"Why or why not?"이 없는데도 no-CoT의 답이
이유를 설명한다.**

```
CoT     Yes, a tv_monitor would be found here because it is a living room. It is common for
        living rooms to have televisions. The presence of a fireplace and couch also supports this.
no-CoT  Yes, a tv_monitor is present in the image. It is placed on the mantle above the
        fireplace. The monitor is currently turned off.
no-CoT  No, there is no plant found in the image. The image contains a dining room with a
        table, chairs, and a window, but no plants are visible.
```

`MIN_NEW_TOKENS = 32`가 두 조건에 똑같이 걸려 있다(`vlm_features.py`). "Yes." 한 단어로 끝날
질문에 32토큰을 강제하면 모델은 무엇이든 더 써야 하고, 그 내용은 장면 묘사와 근거가 된다.
나머지는 "Would a tv_monitor be found in a kitchen? No..."처럼 스스로 질문을 이어 가거나
"(interrogative) (ro) (best)" 같은 퇴행 출력이다. 즉 우리 no-CoT는 **질문만 짧을 뿐 생성은
CoT와 같은 길이의 설명**이다.

논문 부록 C.2 PR2L 4의 "32–48 new tokens"가 no-CoT 조건에도 적용됐는지는 **논문이 밝히지
않는다.** 논문이 no-CoT에서 짧은 생성을 허용했다면 그 조건의 표현에는 추론이 거의 없고,
그것이 27.8 %와 41.9 %를 가른 차이일 수 있다. **아직 가설이다.**

> **2026-09-14 정정 — 논문 원문 재확인.** 위 문단의 "밝히지 않는다"는 과한 표현이었다.
> 부록 C.2는 이 설정을 조건별이 아니라 **"For PR2L-specific design choices" 목록**에 넣었다.
>
> > "4. For generating text in response to our task-relevant prompt, we use sample-based decoding
> > with fixed random seed prior to the decoding with temperature 0.4 and 32 − 48 new tokens
> > generated."
>
> 그리고 no-CoT 조건은 본문 4.3에서 **프롬프트의 뒷부분만 뺀 것**으로만 정의된다.
>
> > "we train PR2L policies both with and without the second part of the prompt."
>
> 조건 간 차이로 적힌 것이 프롬프트뿐이므로, **가장 충실한 읽기는 32–48이 두 조건 모두에
> 적용됐다는 것**이고 우리 구현이 그 읽기다. 조건별 예외는 어디에도 적혀 있지 않다(Listing 2의
> "Any other generation parameters (min / max tokens, temp, etc)"도 일반 서술이다).
>
> 따라서 "논문은 no-CoT에서 짧게 생성했을 것"이라는 가설은 **원문과 어긋나는 가정**이 된다.
> 최소 길이를 없앤 no-CoT 실험은 여전히 "생성 길이가 성공률을 좌우하는가"를 보는 **진단용
> 절제 실험**으로는 의미가 있지만, **논문이 한 일을 재현하는 실험은 아니며** 결과가 27.8 %
> 근처로 떨어져도 "논문의 27.8 %가 왜 나왔는가"를 설명하지 못한다 — 논문은 32–48로 그 수치를
> 얻었기 때문이다.
>
> **2026-09-15 재정정 — 위 정정이 과했다.** no-CoT의 생성 길이는 논문 어디에도 적혀 있지 않다. 부록
> C.2 PR2L 4는 32–48을 **"our task-relevant prompt"에 대한 생성**으로 적는다.
>
> > "4. For generating text in response to **our task-relevant prompt**, we use sample-based decoding
> > with fixed random seed prior to the decoding with temperature 0.4 and 32 − 48 new tokens generated."
>
> 그리고 본문 §4.3이 정한 그 프롬프트는 CoT 문장이다.
>
> > "For Habitat, we choose the prompt "Would a [target object] be found here? Why or why not?""
>
> 논문은 이 표현을 설정마다 "고른 그 프롬프트"를 가리키는 데 쓴다 — 부록 D는 "What room is this?"를
> "our task-relevant prompt"라고 부른다. no-CoT는 그 프롬프트에서 뒷부분을 뺀 **절제 실험**으로만 등장하고
> 생성 설정은 따로 적히지 않는다. 따라서 **두 읽기가 모두 가능하다**: (가) 32–48이 PR2L 공통 설정이라 두
> 조건에 적용됐다, (나) 32–48은 CoT 프롬프트의 설정이고 no-CoT의 생성 길이는 미명시다. 앞의 "원문과
> 어긋나는 가정"이라는 판정은 (가)만 옳다고 본 것이어서 틀렸다. no-CoT의 최소 길이는 **논문 미명시 사항**이고,
> 최소 길이를 없앤 no-CoT 실험은 사양 이탈이 아니라 (나)의 읽기를 따르는 실험이다.

### ④ 판별 실험 (미실행)

```
no-CoT 재인코딩, 최소 생성 길이 제거 (min_new_tokens 0, max 48 유지)
  → no-CoT ≈ 28~35 %   : 최소 길이가 원인. CoT 효과가 복원된다
  → no-CoT ≈ 46 %      : 원인이 아니다. 다른 곳을 찾아야 한다
```

비용은 no-CoT 인코딩 한 번(약 77 GPU시간, 짧은 답이면 그보다 적음)과 학습·평가다. 인코딩
전에 **표본 수백 프레임으로 답변 길이 분포만 먼저 뽑아** 가설이 성립하는지 싸게 확인할 수
있다 — 짧은 답이 대부분 "Yes."/"No." 수준이면 진행할 가치가 있다.

### ⑤ 싼 확인 결과 — 최소 길이가 no-CoT 답을 네 배로 늘린다 (2026-09-14, 작업 16076)

학습 궤적 40개에서 5장씩, 같은 200프레임에 같은 프레임 시드로 네 방식으로 물었다
(`probe_answer_length.py`, rtx2080 두 장에 float16 분할 — 인코딩의 bfloat16과 개별 표본은 다를
수 있으나 네 설정이 공유한다).

| 설정 | 평균 토큰 | 중앙 | ≤10 | 32 미만 | 48 도달 | 이유 어휘 | 2문장 이상 |
|---|---|---|---|---|---|---|---|
| no-CoT, 최소 0 | **10.6** | 11 | 17 % | 100 % | 0 % | **0 %** | **0 %** |
| no-CoT, 최소 32 (인코딩과 같음) | 39.6 | 38 | 0 % | 0 % | 21 % | 13.5 % | **99.5 %** |
| CoT, 최소 0 | 19.0 | 16 | 15 % | 88.5 % | 1.5 % | 60.5 % | 1.5 % |
| CoT, 최소 32 (인코딩과 같음) | 43.4 | 46 | 0 % | 0 % | 47.5 % | 67 % | 90 % |

(이유 어휘: because · since · due to · typically · usually · common 등. 문장 수: `.!?` 기준.)

**가설의 전제는 성립한다.** 스스로 멈추게 두면 no-CoT의 답은 **200개 전부 한 문장, 평균 10.6
토큰**("No, it's not mentioned in the image.")이고 이유가 하나도 없다. 32토큰을 강제하면 99.5 %가
두 문장 이상이 되고, 덧붙는 내용은 장면 묘사("The image only describes a hardwood floor,
furniture, and various objects in the room. There is no mention of a bed.")이거나 퇴행 출력
("v.set(0, 0, 0)", "(3, 177) (100, 185)")이다.

**다만 no-CoT에 강제로 덧붙는 것은 인과 추론이 아니라 장면 묘사다.** 이유 어휘는 13.5 %로 CoT의
67 %에 한참 못 미친다. 따라서 최소 길이가 no-CoT를 "CoT로 만든다"는 C.7 ③의 표현은 과하다.
정확히는 **답에 장면 묘사를 붙여 표현이 담는 내용을 CoT 쪽으로 끌어올린다.** 성공률이 같다는
관측과 함께 읽으면, 정책에 도움이 되는 것이 "왜"라는 추론이 아니라 **방에 무엇이 있는지에 대한
서술**일 가능성을 시사한다. 이것은 논문의 해석과 다르다.

**fp16 대조**: 최소 32 두 설정의 답이 매니페스트에 저장된 bfloat16 표본과 같은 문체·같은 퇴행
양상을 보인다("No, it's not mentioned in the image. Would a tv_monitor be found in..."). 정밀도가
답의 성격을 바꾸지 않았다.

**판별 실험(④)의 근거가 섰다.** 최소 길이 없이 인코딩한 no-CoT는 생성 토큰이 약 11개로 지금의
약 40개보다 훨씬 얇은 표현이 된다. 생성이 짧아 인코딩도 빨라진다 — 생성이 인코딩 시간의 대부분
이므로 약 30 GPU시간, 저장은 프레임당 토큰 약 72 → 약 40개로 약 200 GB로 추정한다.

**사양 문제 (위 ③의 재정정 참조)**: 32–48은 논문이 "our task-relevant prompt", 즉 CoT 프롬프트에 대해
명시했다. no-CoT에 같은 길이를 썼는지는 적혀 있지 않다 — **no-CoT의 최소 길이는 논문 미명시**이고 사용자
결정 사항이다.

## C.8 no-CoT 격차의 다른 원인 후보 (2026-09-14)

최소 길이 가설이 원문과 어긋나 보류된 뒤(C.7 ③ 정정), 남은 사실은 이것이다.

```
레시피 적용 후 논문과의 차이   이미지 +2.0 %p (1.2σ)   CoT +5.5 %p (2.2σ)   no-CoT +18.8 %p (7.7σ)
CoT − no-CoT                  우리 +0.8 %p            논문 +14.1 %p
```

**물체별로 보면 no-CoT의 초과분은 한두 물체가 아니라 전반적이다** — 우리 − 논문, no-CoT:
toilet +20, bed +27, sofa +10, chair +19, tv +16, plant +21 (plant는 21 에피소드). 같은 비교가
CoT에서는 +7, +10, +5, +2, 0, +14다. 특정 장면·물체의 인공물이 아니라 **표현 수준의 체계적
차이**를 가리킨다. 후보는 no-CoT에 **비대칭적으로** 작용해야 한다.

### ① 샘플링 절단(top-k 50) — 가장 유력

논문의 추출 예시(Listing 2·3)는 HF `model.generate(**inputs, output_hidden_states=True,
return_dict_in_generate=True)`를 부르고, 생성 인자는 "Any other generation parameters (min / max
tokens, temp, etc)"라고만 적는다. HF는 `do_sample=True`일 때 **기본값 `top_k=50`** 을 건다
(`GenerationConfig()` 확인: top_k 50, top_p 1.0). Prismatic은 추론용 LLM을 `_from_config`로 만들어
Llama-2의 generation_config를 읽지 않으므로 이 기본값이 그대로 남는다(`base_llm.py:131-132`).
Prismatic 자체의 `generate`도 `super().generate(**kwargs)`로 같은 경로를 탄다.

**우리 `_sample`은 절단 없이 어휘 전체에서 뽑는다.** 논문은 top-k를 밝히지 않았지만, 논문이
보인 코드 경로라면 top-k 50이 걸렸을 가능성이 높다.

**왜 no-CoT에 비대칭인가.** 모델이 끝내고 싶은 지점에서 32토큰을 강제하면 다음 토큰 분포가
평평해진다. 그 꼬리에서 우리 샘플러는 새 내용(장면 묘사)이나 쓰레기("v.set(0, 0, 0)")를 집고,
top-k 50은 꼬리를 잘라 **반복**("No, it's not mentioned in the image." 되풀이)으로 몰 가능성이
높다. 반복은 장면 정보를 담지 않으므로 no-CoT 표현이 얇아진다. CoT는 할 말이 있어 덜 영향받는다.

**방향 조건과 약점 (사용자 지적으로 명시).** 이 후보가 격차를 설명하려면 절단이 no-CoT 성공률을
**낮춰야** 한다 — 논문(절단 추정)이 낮고 우리(절단 없음)가 높기 때문이다. 그런데 그 방향은 보장되지
않는다. 반대 논리도 똑같이 그럴듯하다: 절단은 "v.set(0, 0, 0)" 같은 쓰레기를 없애 **더 깨끗한**
묘사를 남길 수 있고, 그러면 no-CoT가 오히려 좋아지거나 그대로여야 한다. 또 강제 토큰 수의 비대칭도
위 서술만큼 크지 않다 — 최소 0 대비 최소 32에서 늘어나는 토큰이 no-CoT 약 29개, CoT 약 24개로
비슷하다. 비대칭은 개수가 아니라 **강제 구간에서 무엇을 쓰느냐**에서 나와야 한다. 논문 Table 4의
CoT 예시("… It has a red pillow on it. It is a large sectional couch.")는 쓰레기도 반복도 없는 묘사로,
논문 파이프라인이 강제 구간에서 깨끗한 묘사를 냈다는 쪽의 증거다(다만 CoT 프롬프트다).

**판별**: 탐침(`probe_answer_length.py`)에 top-k 50을 추가해, 최소 32의 두 조건에서 반복률·장면
묘사율을 절단 유무로 비교한다. **no-CoT만 반복으로 무너지면 방향이 맞고, 절단 후에도 no-CoT가 장면을
묘사하면 이 후보는 기각한다.** rtx2080 두 장, 약 1.5시간. no-CoT만 반복으로 무너지면 top-k 50으로
재인코딩한다 — 이번엔 논문 코드 경로에 **더 가까워지는** 방향이다(사용자 합의 필요, 논문 미명시).

### ② 논문의 PCA 기저 공유

부록 C.2 PR2L 2는 기저 적합을 조건 구분 없이 한 번의 절차로 서술한다. 논문이 CoT 출력으로 적합한
기저를 no-CoT에 재사용했다면 no-CoT가 정보를 더 잃는다. **사전 확률은 낮다** — 우리 조건별 기저의
설명분산이 CoT 95.98 %, no-CoT 96.12 %로 기하가 거의 같아, 교차 적용해도 손실이 작을 것이다.
**판별**: 탐침 프레임의 원시 4096차원 은닉 상태에 CoT 기저를 적용해 보존 분산을 잰다. ①과 같은
작업에 넣을 수 있다.

### ③ 프롬프트 포장

우리는 Prismatic의 "In: …\nOut: " 형식으로 감싼다. 논문 Listing 3는 InstructBLIP **예시**에 날
문자열을 넘긴다. Habitat 코드도 날 문자열이었다면 pure Llama는 답하는 대신 질문을 이어 쓸 수
있고, "Why or why not?"이 없는 no-CoT가 더 크게 무너질 수 있다. **근거가 약하다** — 그 코드는 다른
모델의 예시다. **판별**: ① 탐침에 포장 없는 설정을 추가한다.

### ④ 시드 분산 — 우리 쪽 잡음 바닥이 없다

PR2L 조건의 시드 대조가 없다(이미지 인코더만 0.4 %p). 논문 수치의 분산은 알 수 없지만, CoT −
no-CoT = +0.8 %p를 해석하려면 우리 쪽 잡음부터 알아야 한다. 18.8 %p를 시드로 설명할 가능성은
낮다. **판별**: no-CoT 시드 1 학습. compute02에 스테이징(661 GB)이 남아 있어 복사 없이 2080에서
약 13시간, 평가 3090에서 약 6시간.

### ⑤ 모든 조건에 공통인 오프셋 — no-CoT 특이 원인은 아니다

세 조건 모두 논문보다 약간 높다(+2.0, +5.5). 500 에피소드 부분집합, 체크포인트 선택 같은 공통
원인이 있다면 no-CoT 초과분 중 수 %p를 설명할 뿐이다. 부분집합의 대표성은 이미지 인코더에서
500 대 2000이 1.0 %p로 확인됐다. 우선순위 낮음.

### 해석에 걸린 미해결 교란 — CoT 레시피 효과

C.7 ①의 "레시피는 조건에 무관하게 약 12 %p를 깎는다"는 CoT 쪽 수치가 **BOS 수정 전(60.2 %)과
수정 후 + 레시피(47.4 %)** 의 차이다. 레시피 없는 `cot_fixed` 대조군을 자원 문제로 뺐기 때문에
BOS 수정의 몫이 섞여 있다. epoch 7 체크포인트가 남아 있어 약 10시간에 복원할 수 있다.

### C.8 탐침 결과 (2026-09-14, 작업 16117, `probe_decoding.py`)

같은 200프레임 · 같은 프레임 시드 · 최소 32 / 최대 48 / temperature 0.4. `*_full` 두 설정은 앞선
탐침(16076)의 최소 32 답과 **200개 전부 글자까지 같아** 하네스가 재현됨을 확인했다.

| 설정 | 토큰 | 4-gram 반복 | 반복 문장 | 쓰레기 | 이유 어휘 | 첫 문장 뒤 장면 명사 |
|---|---|---|---|---|---|---|
| no-CoT, 절단 없음 | 39.6 | 2.8 % | 3.0 % | 25.0 % | 11.5 % | 2.35 |
| no-CoT, top-k 50 | 39.4 | 2.8 % | 3.0 % | 25.5 % | 12.0 % | 2.34 |
| CoT, 절단 없음 | 43.4 | 0.4 % | 0.5 % | 17.0 % | 65.0 % | 2.55 |
| CoT, top-k 50 | 43.4 | 0.4 % | 0.5 % | 16.5 % | 65.5 % | 2.53 |
| no-CoT, 포장 없음 | 43.5 | **27.3 %** | **40.5 %** | 16.5 % | 1.0 % | **1.44** |
| CoT, 포장 없음 | 41.3 | 13.3 % | 23.5 % | 19.0 % | 67.0 % | **1.46** |

**① top-k 50 — 기각.** 절단 유무로 답이 **글자까지 같은 비율이 두 조건 모두 95.5 %** 이고, 모든
지표가 오차 안에서 같다. no-CoT의 쓰레기 답 50개는 절단 후에도 **50개 전부** 쓰레기다. 즉
"v.set(0, 0, 0)" 같은 출력은 꼬리 샘플링의 탈선이 아니라 **상위 50개 안에 드는, 모델 스스로의
고확률 출력**이다. 앞서 쓰레기를 꼬리 탈선으로 설명한 것은 틀렸다. temperature 0.4가 이미 꼬리를
눌러 두어 top-k가 할 일이 거의 없다.

**② PCA 기저 공유 — 중간 정도의 근거.** 같은 토큰을 두 기저에 투영한 보존 분산:

```
토큰     층    자기 기저   상대 기저   손실
no-CoT   −2    97.03 %     95.18 %    −1.9 %p
no-CoT   −1    90.11 %     83.74 %    −6.4 %p   잔차 9.9 % → 16.3 %
CoT      −2    96.83 %     94.50 %    −2.3 %p
CoT      −1    90.71 %     84.20 %    −6.5 %p
```

자기 기저 수치가 bfloat16 적합 로그(97.19 / 90.81)와 가까워 float16이 비교를 흐리지 않았다. 손실은
**마지막 층에 몰려** 잔차가 1.65배가 된다. 논문이 CoT로 적합한 기저 하나를 두 조건에 썼다면 이
손실이 no-CoT에만 떨어진다. 이것만으로 18.8 %p가 설명될지는 알 수 없다.
**판별(재인코딩 불필요)**: 저장된 no-CoT 토큰을 자기 기저로 복원한 뒤 CoT 기저로 다시 투영하는
아핀 사상(층마다 1024 → 1024)을 학습과 평가 입력에 똑같이 걸어 no-CoT를 재학습·재평가한다. 자기
부분공간 밖의 잔차는 이미 없으므로 실제 교차 투영의 근사지만, 부분공간 불일치의 효과는 그대로 잰다.

**③ 프롬프트 포장 — 방향은 맞고, 논문이 그렇게 했다는 근거는 약하다.** 포장을 벗기면 두 조건 모두
장면 명사가 1.4개대로 줄지만, 무너지는 모양이 다르다. no-CoT는 반복 문장이 40.5 %("a bed would be
out of place." 세 번), 이유 어휘가 1 %로 사라지고 "[0.36, 0.54, 0.48, 0.74] executive order 13982"
같은 출력이 나온다. CoT는 반복이 절반 수준이고 이유 어휘 67 %를 유지한다. 즉 포장이 없으면 **두
조건의 표현 내용 차이가 커져** 논문의 CoT ≫ no-CoT 패턴과 방향이 맞는다. 다만 논문 코드가 포장
없이 불렀다는 근거는 InstructBLIP 예시(Listing 3)뿐이다. 판별하려면 두 조건을 모두 재인코딩해야
해서(각 수십 GPU시간) 비용이 크다.

**2026-09-14 정정 — 위 표의 "쓰레기" 열.** 판정식에 밑줄(`_`)이 들어 있어 목표 이름 `tv_monitor`를
언급한 답이 모두 쓰레기로 잡혔다. 목표 이름을 지우고 다시 세면 no-CoT 절단 없음 14.0 % / top-k 50
14.5 %, CoT 두 설정 0.0 %, 포장 없음 두 조건 6.0 %다. ① 기각은 그대로다 — 보정한 no-CoT 쓰레기 답
28개가 top-k 50에서도 28개 전부 남는다.

## C.9 답변 형식 — 논문 예시와 우리 인코딩 비교 (2026-09-14)

**비교 자료.** 논문은 Table 4에 CoT 답 6개, Figure 1에 1개(축약으로 보임)를 싣는다. **no-CoT 답은 하나도
싣지 않는다.** 우리 쪽은 탐침이 아니라 실제 인코딩(bfloat16)이 매니페스트에 남긴 `sample_answers` —
궤적마다 첫 세 프레임, 조건별 23,472개다. 집계는 `compare_answers.py`, 토큰은 Llama-2 토크나이저로 셌다.

| | 논문 Table 4 (6) | 우리 CoT (23,472) | 우리 no-CoT (23,472) |
|---|---|---|---|
| 토큰 평균 · 범위 | 43.7 · 33–49 | 42.6 · 32–48 | 40.0 · 19–48 |
| 문장 수 | 2.33 | 2.56 | 3.14 |
| 예/아니오로 시작 | 100 % | 99.8 % | 95.6 % |
| 방 종류 언급 | 100 % | 68.6 % | 36.1 % |
| 이유 어휘 | 83 % | 80.8 % | 25.0 % |
| "설명·목록·언급" 어휘 | 17 % | 15.0 % | **80.8 %** |
| "not mentioned in the image" | 0 % | 2.4 % | **77.6 %** |
| 문장 도중 끊김 | 17 % | 42.2 % | 31.7 % |
| 쓰레기(좌표 등) · 문장 반복 | 0 % · 0 % | 0.1 % · 0.3 % | 9.2 % · 8.2 % |

**CoT는 논문과 형식이 거의 같다.** 길이(43.7 대 42.6토큰, 범위도 32–48에 맞음), 문장 수, 예/아니오
시작, 이유 어휘(83 대 81 %), 설명형 어휘(17 대 15 %)가 모두 가깝다. 논문 예시 하나가 "rather than
sleeping"에서 끊겨 있어 **최대 48토큰 상한이 논문 파이프라인에도 걸렸음**을 보여 준다. 방 종류 언급
(100 대 69 %)과 끊김(17 대 42 %)의 차이는 표본 6개짜리 선별 예시와의 비교라 해석하지 않는다. 우리
CoT에서 가장 흔한 첫 문장은 "No, there is no <T> in the image."(11.2 %)로, 논문 예시가 모두 방 종류를
근거로 드는 것보다 탐지형이다.

**no-CoT는 틀에 박혀 있다.** 74.1 %가 글자까지 같은 "No, it's not mentioned in the image."로 시작하고,
이어서 "The objects listed are mostly related to lighting and walls. A chair is not mentioned. So, the
instruction is misleading." 같은 말이 붙는다. 존재하지 않는 "목록"과 "설명"을 참조하고, 질문을
"오도하는 지시"로 규정하며, "(154, 147, 1005, 105)" 같은 영역 좌표를 낸다. CoT에서는 이 틀이 2.4 %뿐이다.

### 새 후보 ⑥ — 체크포인트: 같은 이름 조건에 맞는 모델이 둘이다

설치된 Prismatic 레지스트리(`prismatic/models/registry.py`)에 224px DINOSigLIP 7B가 두 개 있다.

| 체크포인트 | 학습 데이터 | epoch |
|---|---|---|
| `prism-dinosiglip-224px+7b` (**우리**) | LLaVA v1.5 Instruct + LVIS-Instruct-4V + **LRV-Instruct** | 2 |
| `prism-dinosiglip-224px-controlled+7b` | LLaVA v1.5 Instruct만 | 1 |

논문 부록 C.2의 "Dino+SigLIP as a vision backbone and Llama2-7B-pure as the language backbone. We use
the 224px version"은 **둘 다에 맞는다.** 우리 사양표(§2.5)는 이것을 "논문"으로 분류했는데 틀렸다 —
실제로는 미명시 사항에서 우리가 고른 것이다(위 표에서 정정).

LRV-Instruct의 원 논문 제목은 "Mitigating hallucination in large multi-modal models via robust
instruction tuning"(Prismatic 논문 참고문헌 Liu et al., 2023a)이다. 우리 no-CoT의 "the instruction is
misleading", "not mentioned in the image"는 **존재하지 않는 대상을 묻는 지시에 거절하도록 가르치는
환각 억제 데이터**의 응답 형식과 맞아떨어진다. 다만 LRV 데이터 원문으로 직접 대조하지는 않았으므로
**추론**이다. "Would a X be found here?"는 X가 대개 화면에 없는 질문이라, 이 틀을 강하게 부른다.
"Why or why not?"이 붙으면 설명 요청으로 읽혀 틀이 거의 사라진다.

**비대칭은 뚜렷하지만 방향은 모른다.** 이 틀은 no-CoT에만 몰려 있다(77.6 % 대 2.4 %). controlled
체크포인트라면 no-CoT 답이 달라질 것이지만, 그것이 성공률을 올릴지 내릴지는 형식만으로 말할 수 없다
— 우리 no-CoT는 틀에 박혔는데도 CoT와 같은 성공률을 냈다. 논문 예시는 CoT뿐이라 체크포인트를 가르는
증거가 되지 못한다(예시 4의 "the room is described as a bedroom"은 설명형 버릇이 논문 모델에도 있음을
보이지만 두 체크포인트 중 어느 쪽인지는 알 수 없다).

**판별.** 싼 쪽은 controlled 체크포인트(약 14 GB)를 받아 탐침 하네스로 같은 200프레임의 no-CoT·CoT
답 형식을 보는 것(rtx2080 두 장, 약 2시간). 성공률까지 보려면 두 조건 재인코딩·학습·평가가 필요하다.

### 부수 발견 — 목표 이름

프롬프트에 Habitat 범주 문자열이 그대로 들어가 "Would a tv_monitor be found here?"가 된다. tv_monitor
답의 약 절반(CoT 47 %, no-CoT 53 %)이 밑줄 붙은 `tv_monitor`를 그대로 되풀이한다. 논문 본문은 목표를
"television"으로 부르지만 프롬프트에 어떤 문자열을 넣었는지는 적지 않았다. 두 조건에 똑같이 걸리고
평가 에피소드의 14 %에만 해당해 no-CoT 격차의 원인은 아니지만, 사양표에 미명시 선택으로 올려야 한다.

### C.9 탐침 결과 — controlled 체크포인트 (2026-09-14, 작업 16277, `compare_probe_models.py`)

`prism-dinosiglip-224px-controlled+7b`(config `dataset_id: llava-v15`, finetune 1 epoch)를 받아 앞선
탐침(16076)과 같은 200프레임 · 같은 시드 · 같은 float16 하네스로 물었다. 두 체크포인트는 파일 크기가
바이트까지 같지만(같은 구조) 가중치 중간 64 MB의 md5가 다르다.

**거절 틀은 우리 체크포인트의 성질이다.** no-CoT, 최소 32(인코딩 조건):

| | 우리 (prism) | controlled |
|---|---|---|
| "not mentioned in the image" | 59.0 % | **0.0 %** |
| "instruction / misleading" | 16.0 % | **0.0 %** |
| 설명·목록·언급 어휘 | 62.5 % | 5.5 % |
| 좌표 등 쓰레기 | 14.0 % | 0.0 % |
| 방 종류 언급 | 36.0 % | 64.0 % |
| 가장 흔한 첫 문장 | "No, it's not mentioned in the image." 56 % | "No, a <T> would not be found in this room." 14 % |

controlled는 LRV·LVIS가 빠진 만큼 거절 틀이 완전히 사라지고, 질문을 "이 방에 있을 법한가"로 읽어
방을 근거로 답한다. 또 **말이 길다** — 최소 길이 없이도 no-CoT 평균 34.7토큰(48 도달 40.5 %), CoT는
200개 전부 48토큰에 닿아 최소 0과 최소 32의 답이 100 % 같다.

**그런데 논문 예시는 controlled가 아니라 우리 체크포인트처럼 생겼다.** 논문 Table 4의 CoT 답 6개와
두 체크포인트의 CoT(최소 32)를 같은 지표로 대조했다. 논문 예시의 토큰 수는 49, 49, 48, 47, 36, 33이다
(PDF에서 옮긴 문장이라 ±1 오차가 있다).

| 지표 | 논문 Table 4 (6) | 우리 (prism) | controlled |
|---|---|---|---|
| 예/아니오로 시작 | 100 % | 99.5 % | 61.5 % |
| 48토큰 전에 스스로 멈춤 | 약 50 % (47·36·33) | 52.5 % | **0 %** |
| 문장 도중 끊김 | 17 % | 44 % | 96.5 % |
| 짧은 "It is / It has" 문장 | 33 % | 24.5 % | 1.5 % |
| "described" | 17 % | 6.5 % | 3.0 % |
| "Yes/No, a X would/is (not) found" 틀 | 83 % | 38.5 % | 58.5 % |

여섯 지표 중 다섯이 우리 체크포인트 쪽이다. 결정적인 것은 길이다 — controlled는 CoT 답 200개가 **전부**
48토큰 상한에 닿는데, 논문 예시에는 36토큰과 33토큰짜리가 있다. 선별된 예시라 해도 한 번도 48 전에
멈추지 않는 모델에서 그런 답을 고르기는 어렵다. 논문 예시 6("Yes, there is a black leather sofa in the
living room. It has a red pillow on it. It is a large sectional couch.")의 짧은 탐지형 문장은 우리 모델의
"Yes, there is a chair in the corner. It is gray and red."와 문체가 같다.

**판정: 후보 ⑥ 약화.** 논문은 우리와 같은 `prism-dinosiglip-224px+7b`를 썼을 가능성이 높다. 그렇다면
no-CoT의 거절 틀은 논문 쪽에도 똑같이 있었을 것이므로 격차를 설명하지 못한다. 한계: 논문 예시 6개,
다른 프레임, PDF 전사, 그리고 생성 설정이 달랐을 가능성(controlled라도 최대 길이를 다르게 뒀다면 짧은
답이 나올 수 있다 — 다만 논문은 32–48을 명시했다).

## C.10 no-CoT 최소 길이 0 — 판별 실험 (2026-09-15 제출)

**근거.** 부록 C.2 PR2L 4의 "32 − 48 new tokens"는 "our task-relevant prompt", 즉 CoT 프롬프트에 대한
설정이고 no-CoT의 생성 길이는 논문에 없다(C.7 ③ 재정정). CoT의 최소 32는 설정이었다는 증거가 강하다 —
같은 체크포인트로 최소 없이 생성하면 CoT 답의 88.5 %가 32토큰 미만(중앙 16)인데 논문 예시 6개는 모두
33–49토큰이고 하나가 48에서 끊긴다(무작위라면 0.115⁶ ≈ 2×10⁻⁶). no-CoT는 예시가 없어 같은 논리를 쓸 수
없다. 최소 없는 no-CoT 답은 평균 10.6토큰 한 문장("No, it's not mentioned in the image.")이고, 최소 32는
보이는 물건을 나열하는 설명을 덧붙여 CoT와 내용이 비슷해진다(첫 문장 뒤 장면 명사 2.35 대 2.55).

**구현.** 조건 `nocot_min0` 추가 — `vlm_features.NO_MINIMUM_CONDITIONS`와 `min_new_tokens_for()`로 인코딩
(`encode.py` fit/encode)과 평가(`evaluate.py` 롤아웃)에 같은 최소 길이 0을 건다. 최대 48, temperature 0.4,
프레임 시드, 포장, 체크포인트, BOS 수정, PCA 절차는 `nocot`과 같고 **기저는 새로 적합**한다(토큰 분포가
달라진다). 기존 조건의 동작은 바뀌지 않는다(기본값 32).

| 단계 | 작업 | 자원 |
|---|---|---|
| PCA 적합 | 16381 | 3090/4090 bf16 |
| 인코딩 12샤드 | 16382 (`%4`) | 3090/4090 bf16 |
| 스테이징+학습 | 16383, `stageB_nocotMin0C3` — 클리핑 0.2 · eps 1e-5 · fp32 · 요약 조각 256 | rtx2080 |
| 평가 4샤드 | 16384 (`%2`), 500 에피소드 | 3090/4090 |

학습 설정은 비교 대상인 `stageB_nocotC3`(46.6 %)와 같다 — 그 실행도 2080 fp32 · 요약 조각 256이었다.

**실행 전에 정한 판정.** 500 에피소드 표준오차 약 ±2.2 %p, 논문 2000 에피소드 ±1.0 %p.

```
x ≤ 37 %        최소 길이가 no-CoT 격차를 설명한다. 논문의 27.8 %는 최소 길이 없는 no-CoT로 재현된다
                (27.8과의 차이를 σ로 함께 보고한다)
x ≥ 40 %        최소 길이는 원인이 아니다 (46.6과 구분되지 않거나 약간 낮음)
37 < x < 40     판단 보류 — 부분 기여. 시드 대조(④) 없이는 해석하지 않는다
```

### C.10 결과 — 최소 길이는 원인이 아니다 (2026-10-01)

**실제로 돈 작업.** 위 표의 16381–16384는 공유 HF 캐시 문제(샤드마다 SigLIP 3.5 GB를 다시 받다 멈춤)로
취소하고 다시 제출했다. 인코딩 16809(샤드 1–11)·19600(샤드 0 재실행), 학습 16810(rtx2080, fp32, 요약 조각 256,
epoch당 약 1,130초, 읽기 3–5초), 평가 16811(4샤드, 3090/4090). 인코딩 7,824 궤적 · 193 GB(최소 32의 356 GB 대비),
PCA 적합 표본 645,792토큰(프레임당 약 41 = 시각 16 + 질문 13 + 생성 약 12), 설명 분산 98.81 %. 학습 최종
epoch 40 손실 1.1029, 정확도 74.8 %.

| 실행 | 성공률 | SPL | 자발적 정지 |
|---|---|---|---|
| `stageB_nocotMin0C3` (최소 0) | **46.0 %** | 0.254 | 55.2 % |
| `stageB_nocotC3` (최소 32) | 46.6 % | 0.248 | 54.6 % |
| 논문 no-CoT | 27.8 % | | |

```
최소 0 − 최소 32     −0.6 %p   ±3.15p   0.2σ     ← 구분 불가
최소 0 − 논문        +18.2 %p  ±2.44p   7.5σ
```

**판정: x = 46.0 % ≥ 40 % → 최소 길이는 no-CoT 격차의 원인이 아니다.** 생성 답변이 평균 약 12토큰
한 문장으로 줄어 표현이 크게 달라졌는데도(저장 용량 46 % 감소) 성공률은 움직이지 않았다. 거리별로도 같다.

```
거리     2-4 m          4-6 m          6+ m
최소 0   69 % (111)     58 % (133)     30 % (256)
최소 32  72 % (111)     59 % (133)     29 % (256)
```

**이것이 뜻하는 것.** 우리 정책의 성공률은 생성 답변의 길이와 내용에 거의 무감하다. CoT(47.4 %), no-CoT
최소 32(46.6 %), 최소 0(46.0 %)이 1.4 %p 안에 모인다. 논문이 보고한 CoT 효과(+14.1 %p)는 답변 내용이 정책에
영향을 준다는 뜻인데, 우리 재현에서는 그 경로 자체가 작동하지 않는 것으로 보인다. no-CoT 격차의 남은 후보는
C.8의 ② PCA 기저, ③ 프롬프트 포장, ④ 시드 분산이다.
