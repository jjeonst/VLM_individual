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
| 모델 | `prism-dinosiglip-224px+7b` | 논문 |
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

**수정**: 평가에서도 `reduced.astype(np.float16)`을 거치게 한다. 재인코딩은 필요 없고 평가만
다시 돌리면 된다.

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

**판정**: 같은 100 에피소드에서 우리 루프와 성공률이 **±3 %p 안에서 일치**해야 한다. 어긋나면
그 차이가 곧 우리 평가 루프의 편향이다.

**비용**: 래퍼 약 100줄 + 실행 2시간. **통과하면 "평가가 관대해서 높다"는 반론이 통째로 닫힌다.**

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
