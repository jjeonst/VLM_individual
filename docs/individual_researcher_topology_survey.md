# 개별연구생 과제: TopoVLM topology survey와 시각화

## 과제 목적

이번 과제의 목표는 새로운 모델이나 loss를 바로 만드는 것이 아닙니다. 먼저
navigation과 robotics 환경 안에 실제로 `topology`라고 부를 만한 구조가 있는지,
그리고 그 구조를 어떻게 시각화하고 benchmark로 만들 수 있을지 정리해 주세요.

특히 이번 발표의 중심은 **navigation 환경에서 trajectory가 서로 겹치고, 그 겹침을
이용해 stitching할 수 있는지**를 보는 것입니다. 동시에 그런 trajectory 안에서
**branching 후보**, 즉 agent가 어느 쪽으로 갈지 선택해야 하는 지점이 어디인지도
정리해 주세요.

발표는 15--20분 정도를 목표로 준비해 주세요. 큰 training을 돌리기보다는, dataset
구조를 읽고, 작은 예시를 찾고, 그림으로 설명하는 데 집중해 주시면 됩니다.

## 먼저 답해야 할 질문

각 dataset이나 environment를 볼 때 아래 질문에 답해 주세요.

1. **Observation space**
   - agent가 실제로 보는 입력은 무엇인가요?
   - 예: RGB frame, panorama, depth, semantic map, language instruction, object state,
     proprioception, scene graph.

2. **Action space**
   - agent가 실제로 선택하는 action은 무엇인가요?
   - 예: adjacent viewpoint 선택, `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`,
     continuous end-effector command, discrete skill/option.

3. **Topology object**
   - 어떤 구조를 topology라고 부를 수 있나요?
   - 예: environment graph, trajectory overlap graph, junction, bottleneck, branch point,
     merge point, option graph, precondition graph.
   - 이 구조가 dataset annotation에 직접 들어 있나요, trajectory에서 복원해야 하나요,
     아니면 연구자가 새로 정의해야 하나요?

4. **Composable components**
   - task를 어떤 단위로 쪼갤 수 있나요?
   - 예: shared route segment, room transition, doorway crossing, object search,
     placement, activation, prerequisite skill.

5. **Visualization**
   - 이 topology를 사람이 한눈에 볼 수 있게 어떻게 그릴 수 있나요?
   - 예: trajectory overlap graph, colored route overlay, junction/branch marker,
     option transition graph, task program graph, valid/invalid branch diagram.

## 이번 발표에서 꼭 보면 좋은 navigation 예시

### 예시 1: R2R / VLN trajectory overlap과 stitching

R2R-style 또는 VLN-style dataset은 가장 먼저 볼 만한 예시입니다. 여러 language
instruction trajectory가 같은 environment graph 위를 지나가면, 서로 다른 trajectory
사이에 같은 node, edge, subpath가 반복해서 나타날 수 있습니다.

아래와 같은 concrete example을 하나 찾아와 주세요.

```text
trajectory A: A -> B -> C -> D -> E
trajectory B: X -> B -> C -> Y

shared subpath: B -> C
possible stitched route: A -> B -> C -> Y
```

이 예시에서 정리할 내용은 다음입니다.

- `B -> C`가 정말 같은 physical / graph subpath인지 확인해 주세요.
- 두 trajectory가 같은 node를 공유하는지, 아니면 시각적으로 비슷하지만 graph상으로는
  다른 node인지 구분해 주세요.
- shared subpath 앞뒤에 어떤 branch / merge가 있는지 표시해 주세요.
- stitch한 route가 graph connectivity를 보존하는지 확인해 주세요.
- 가능하면 같은 environment graph 위에 trajectory A/B를 색으로 overlay한 그림을
  만들어 주세요.

발표 그림은 복잡할 필요 없습니다. node와 edge만 있는 작은 graph라도 괜찮습니다.
중요한 것은 “왜 이것이 stitching 후보인지”가 보이는 것입니다.

### 예시 2: navigation branching 후보

같은 navigation dataset에서 branching 후보를 찾아 주세요. 여기서 branch는 단순히
길이 갈라지는 곳이 아니라, **agent의 다음 선택이 future route나 success를 바꾸는
지점**입니다.

우선 아래 후보들을 찾아보면 좋습니다.

- **Junction / fork**: 여러 outgoing edge가 있고, instruction에 따라 올바른 edge가 달라지는 지점.
- **Doorway / bottleneck**: 방과 방을 잇는 좁은 통로처럼 여러 trajectory가 모이는 지점.
- **Merge point**: 다른 route가 다시 같은 node나 corridor로 합쳐지는 지점.
- **Dead-end branch**: 들어가면 다시 돌아나와야 하거나, goal과 멀어지는 branch.
- **Landmark-conditioned branch**: 같은 junction이어도 instruction의 landmark cue에 따라
  선택해야 하는 edge가 달라지는 경우.

각 branch 후보마다 아래 정보를 표로 정리해 주세요.

| Candidate | Observation | Action choices | Why topology? | How to visualize? |
| --- | --- | --- | --- | --- |
| Junction | panorama + instruction | next viewpoint candidates | route choice changes future path | node with outgoing colored edges |
| Doorway bottleneck | panorama / map node | pass through or turn away | many paths share the connector | thick shared edge or heatmap |
| Dead-end branch | graph + trajectory | enter or avoid | invalid branch creates recovery cost | red invalid edge |

### 예시 3: Habitat ObjectNav / PR2L-style navigation

TopoVLM repo의 현재 implementation은 HM3D ObjectNav 기반 PR2L-style path를 포함하고
있습니다. 이 경우 R2R처럼 language route graph가 바로 주어지지 않을 수 있으므로,
shortest-path expert trajectory와 scene/object structure에서 topology 후보를 복원해야
합니다.

가능하면 아래 예시 중 하나를 찾아 주세요.

- 여러 ObjectNav shortest-path trajectory가 같은 doorway나 corridor를 공유하는 경우.
- 같은 object category를 향하지만 서로 다른 room entry / route branch를 쓰는 경우.
- target 근처에서 `STOP` 여부가 갈리는 경우.
- wrong turn을 하면 다시 돌아나와야 하는 dead-end 또는 recovery branch.

정리할 때는 다음을 포함해 주세요.

- observation: egocentric RGB, optional depth/semantic state, target object category,
  VLM prompt-conditioned representation.
- action: Habitat discrete actions such as `STOP`, `MOVE_FORWARD`, `TURN_LEFT`,
  `TURN_RIGHT`.
- topology candidate: navigable space graph, shortest-path graph, bottleneck,
  room/object semantic adjacency.
- visualization: top-down path overlay가 가능하면 좋고, 어렵다면 action sequence와
  branch 후보를 작은 graph로 그려도 됩니다.

## Robotics / kitchen 쪽은 확장 예시입니다

Robotics도 중요하지만, 이번 첫 발표에서는 navigation 예시가 중심입니다. Kitchen task의
PCA-style visualization이나 RoboCasa choice-point graph는 좋은 예시가 될 수 있지만,
거기에만 제한하지 않아도 됩니다.

가능하면 아래 중 1개 정도만 짧게 비교해 주세요.

- RoboCasa / kitchen manipulation task.
- RoboSuite-style manipulation task.
- CALVIN-style language-conditioned manipulation task.
- BEHAVIOR / CookBench처럼 object, receptacle, prerequisite가 잘 보이는 benchmark.
- 다른 VLA benchmark 중 observation, action, success condition이 명확한 것.

Robotics에서 topology는 physical map보다 아래 구조로 나타날 수 있습니다.

- object-receptacle graph.
- precondition graph.
- skill transition graph.
- valid/invalid option branch.
- task-stage graph.

예시:

```text
choice point 1: choose object
  valid: target mug
  invalid: distractor object

choice point 2: choose placement
  valid: coffee machine area
  invalid: counter or wrong appliance

choice point 3: activate / finish
  valid: press correct button after placement
  invalid: premature activation or no-op
```

Kitchen PCA-style visualization을 본다면, 그것은 “representation이 task stage나 option을
분리해서 보여주는지”를 확인하는 하나의 방법으로만 다뤄 주세요. PCA가 필수 방법은
아니고, choice-point graph, option graph, task program graph, valid/invalid edge diagram도
모두 가능한 시각화 방법입니다.

## Benchmark-first framing

발표의 결론은 새로운 model을 제안하는 것이 아니라, 아래 질문에 답하는 benchmark
proposal이어도 충분합니다.

> Existing VLM/VLA models가 explicit topology information을 받았을 때, shared subpath,
> branch point, valid/invalid option, stitchable trajectory를 이해할 수 있을까요?

가능한 probe task는 다음과 같습니다.

1. **Shared-subpath detection**
   - 두 navigation trajectory를 주고, 공통 subpath와 branch / merge point를 찾게 합니다.

2. **Trajectory stitching**
   - 여러 route fragment를 주고, graph connectivity를 보존하는 stitched route가 가능한지
     묻습니다.

3. **Next valid branch prediction**
   - current node, instruction, candidate next nodes를 주고 success-connected next branch를
     고르게 합니다.

4. **Branch validity judgment**
   - candidate branch가 goal route인지, dead-end인지, recovery가 필요한 wrong turn인지
     판단하게 합니다.

5. **Robotics choice-point labeling**
   - task description과 candidate actions를 주고, 현재 choice point와 valid/invalid option을
     구분하게 합니다. 이 항목은 navigation 예시가 정리된 뒤 확장으로 보면 됩니다.

## 발표 deliverables

발표에는 아래 산출물이 들어가면 좋습니다.

1. **Comparison table**

| Environment / dataset | Observation | Action | Topology candidate | Example to inspect | Visualization | Risk / missing info |
| --- | --- | --- | --- | --- | --- | --- |
| R2R / VLN | panorama + instruction + graph | adjacent viewpoint / stop | trajectory overlap graph | shared subpath + stitched route | colored route overlay | graph/data access |
| Habitat ObjectNav | egocentric RGB + goal | discrete Habitat actions | bottleneck / branch graph | shared doorway or wrong-turn branch | top-down path or action graph | semantic map availability |
| RoboCasa / kitchen | RGB + state + instruction | continuous or skill action | precondition / option graph | object-placement-activation task | choice-point graph or PCA-style plot | action abstraction |

2. **Navigation stitching figure**
   - 적어도 하나의 shared subpath 예시를 그려 주세요.
   - 어떤 trajectory fragments를 이어붙일 수 있는지 표시해 주세요.

3. **Navigation branching figure or table**
   - junction, bottleneck, dead-end, merge point 중 2--3개 후보를 정리해 주세요.
   - 각 후보에서 action choices와 valid/invalid branch를 표시해 주세요.

4. **One benchmark proposal slide**
   - input format.
   - expected output.
   - correctness criterion.
   - 왜 이것이 generic language ability가 아니라 topology understanding을 보는지.

5. **Source list**
   - dataset paper / project page.
   - environment documentation.
   - inspected data schema 또는 example episode / trajectory file.

## Suggested work plan

### Week 1: source survey

- R2R / VLN-style navigation dataset을 먼저 확인해 주세요.
- 추가로 Habitat ObjectNav 또는 PR2L-style navigation path를 비교 후보로 확인해 주세요.
- robotics는 RoboCasa/kitchen 계열을 하나의 확장 예시로만 짧게 확인해도 됩니다.
- 각 dataset에서 observation, action, trajectory graph, instruction field가 있는지 표로 정리해 주세요.

### Week 2: navigation example 찾기

- R2R / VLN 또는 predictive branching dataset에서 overlapping trajectory 예시를 하나 찾습니다.
- shared subpath, branch point, merge point, stitched route 후보를 작은 graph로 그립니다.
- Habitat ObjectNav에서는 doorway, bottleneck, wrong-turn, stop decision 후보를 찾아봅니다.

### Week 3: benchmark sketch

- navigation 중심 probe question 3--4개를 정의합니다.
- 각 probe마다 input, output, correctness criterion, likely failure mode를 씁니다.
- training 없이 existing VLM/VLA에게 물어볼 수 있는 probe와, 나중에 policy/evaluation이 필요한 probe를 구분합니다.

### Week 4: presentation

- 15--20분 발표를 준비합니다.
- 마지막 slide에는 “첫 benchmark로 무엇이 가장 좋은가”를 추천해 주세요.
- 추천할 때는 easiest, most convincing, least dependent on new training 기준으로 판단해 주세요.

## 처음에는 하지 않아도 되는 것

- 큰 training job은 돌리지 않아도 됩니다.
- learned latent가 topology-aware하다고 바로 주장하지 않아도 됩니다.
- graph가 있다는 이유만으로 topology라고 부르지 말고, 그 graph가 decision-making에 어떤
  영향을 주는지 설명해 주세요.
- dataset topology, task topology, learned representation topology를 섞지 말고 구분해 주세요.
- Kitchen PCA-style plot에만 매달리지 않아도 됩니다. 그것은 여러 visualization 후보 중
  하나입니다.

## Success criteria

이 과제는 아래 질문에 답하면 성공입니다.

1. Navigation 환경에서 trajectory stitching이 가능한 shared subpath 예시를 찾았나요?
2. Navigation 환경에서 branching 후보를 2--3개 이상 정리했나요?
3. Observation space와 action space를 dataset별로 명확히 적었나요?
4. 이 topology를 사람이 볼 수 있는 그림이나 표로 표현했나요?
5. Existing VLM/VLA가 이 topology를 이해하는지 평가할 benchmark question을 제안했나요?
6. Robotics/kitchen 예시는 navigation 중심 논의를 보조하는 확장 예시로만 적절히 배치했나요?
