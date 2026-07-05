# Individual Researcher Assignment: Topology Survey for TopoVLM

## 목적

TopoVLM에서 바로 새로운 method를 만들기 전에, navigation과 robotics 환경 안에
실제로 `topology`라고 부를 만한 구조가 있는지 먼저 정리한다. 목표는
"VLM/VLA가 topology information을 받았을 때 decision-relevant structure를 이해하는가"를
볼 수 있는 benchmark 또는 presentation 방향을 잡는 것이다.

이 과제는 training이나 큰 implementation이 아니라, source-grounded survey,
작은 data inspection, diagram design, 발표 자료 준비가 중심이다.

## 핵심 질문

각 환경 또는 dataset마다 아래 질문에 답한다.

1. **Topology object**
   - 어떤 graph, branch, option, bottleneck, subgoal, precondition structure가 있는가?
   - 이 구조는 dataset annotation에 명시되어 있는가, trajectory에서 복원해야 하는가,
     아니면 연구자가 정의해야 하는가?

2. **Observation space**
   - agent가 보는 입력은 무엇인가?
   - 예: RGB frame, panorama, depth, semantic map, language instruction, object state,
     proprioception, scene graph.

3. **Action space**
   - agent가 선택하는 action은 무엇인가?
   - 예: adjacent viewpoint 선택, `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`,
     continuous end-effector command, discrete skill/option.

4. **Composable components**
   - task를 어떤 component로 쪼갤 수 있는가?
   - 예: room-to-room route segments, object search, placement, activation,
     skill precondition, object-receptacle relation, repeated subtrajectory.

5. **Visualization**
   - topology를 사람이 한눈에 볼 수 있게 그릴 방법은 무엇인가?
   - 예: trajectory overlap graph, choice-point graph, option transition graph,
     bottleneck heatmap, task program graph, valid/invalid branch diagram.

## Navigation 조사 항목

### R2R / trajectory-following datasets

R2R-style navigation은 topology survey의 가장 좋은 출발점이다. 여러 natural-language
instruction trajectory가 같은 environment graph 위를 지나가므로, trajectory 사이에
겹치는 node, edge, subpath가 생긴다.

조사할 것:

- observation: panorama 또는 viewpoint-level visual observation, language instruction,
  navigation graph.
- action: adjacent viewpoint 선택 또는 `STOP`.
- topology candidate:
  - environment connectivity graph,
  - overlapping trajectory subpaths,
  - branch point / merge point,
  - trajectory stitching 가능 지점.
- compositional component:
  - shared corridor / room transition,
  - instruction segment,
  - route segment,
  - start-to-bottleneck / bottleneck-to-goal subpath.
- visualization idea:
  - 같은 environment graph 위에 여러 trajectory를 색으로 overlay한다.
  - overlap edge는 두껍게 표시한다.
  - branch point와 merge point를 별도 marker로 표시한다.
  - 두 trajectory가 공유 subpath를 통해 stitch 가능한 예시를 1--2개 만든다.

예시 diagram:

```text
T1: A -> B -> C -> D -> E
T2: X -> B -> C -> Y
T3: A -> B -> C -> Y

shared subpath: B -> C
possible stitching: A -> B -> C -> Y
```

이 예시는 "새 policy를 학습하지 않아도, dataset 자체가 topological reuse와 branching을
갖고 있는가"를 보여주는 데 좋다.

### Habitat ObjectNav / PR2L-style navigation

TopoVLM repo의 현재 implementation은 HM3D ObjectNav 기반 PR2L-style path를 포함한다.
이 환경에서는 raw dataset이 R2R처럼 language route graph를 바로 주지 않을 수 있으므로,
shortest-path expert trajectory와 scene/object structure에서 topology를 복원해야 한다.

조사할 것:

- observation: egocentric RGB, optional depth/semantic state, target object category,
  VLM prompt-conditioned representation.
- action: Habitat discrete actions such as `STOP`, `MOVE_FORWARD`, `TURN_LEFT`,
  `TURN_RIGHT`.
- topology candidate:
  - navigable space graph,
  - shortest-path trajectory graph,
  - object-goal-conditioned bottleneck,
  - room/object semantic adjacency.
- compositional component:
  - approach target object,
  - enter room,
  - cross doorway,
  - turn/search local region,
  - stop decision.
- visualization idea:
  - shortest-path trajectories projected onto top-down map if available,
  - repeated doorway/bottleneck usage,
  - action sequence segments aligned to semantic route components.

## Robotics 조사 항목

Robotics에서는 topology가 physical space graph라기보다, object relation, skill order,
precondition, option validity로 나타나는 경우가 많다.

Candidate environments:

- RoboCasa / kitchen manipulation tasks.
- RoboSuite-style manipulation tasks.
- CALVIN-style language-conditioned manipulation tasks.
- Any VLA benchmark that exposes language instruction, observation, action, and
  success condition clearly.

조사할 것:

- observation: RGB camera, object state, gripper/proprioception, language instruction.
- action: continuous end-effector action, gripper action, or abstract skill/option.
- topology candidate:
  - object-receptacle graph,
  - precondition graph,
  - skill transition graph,
  - valid/invalid option branches,
  - task-stage graph.
- compositional component:
  - choose object,
  - move to receptacle,
  - place/open/close/activate,
  - satisfy precondition before final action.
- visualization idea:
  - task program graph,
  - choice point with option set,
  - valid edge vs invalid edge,
  - same abstract task graph under different object labels.

Example:

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

## Benchmark-first framing

학생 발표의 결론은 새로운 loss를 제안하는 것이 아니라, 다음 질문에 답하는 benchmark
proposal이어도 충분하다.

> Existing VLM/VLA models가 explicit topology information을 받았을 때, choice point,
> branch, shared subpath, valid/invalid option, stitchable trajectory를 이해하는가?

Possible probe tasks:

1. **Shared-subpath detection**
   - 두 navigation trajectory를 주고, 공통 subpath와 branch point를 찾게 한다.

2. **Trajectory stitching**
   - 여러 route fragment를 주고, graph connectivity를 보존하는 stitch가 가능한지 묻는다.

3. **Next valid branch prediction**
   - current node, instruction, candidate next nodes를 주고 valid next branch를 고르게 한다.

4. **Robotics choice-point labeling**
   - task description과 candidate actions를 주고, 현재 choice point와 valid/invalid option을
     구분하게 한다.

5. **Option-topology isomorphism**
   - 서로 다른 task 두 개가 같은 abstract choice-point graph를 갖는지 판단하게 한다.

## 발표 deliverables

발표에는 아래 산출물이 들어가야 한다.

1. **Comparison table**

| Environment / dataset | Observation | Action | Topology candidate | Composable components | Visualization | Risk / missing info |
| --- | --- | --- | --- | --- | --- | --- |
| R2R | panorama + instruction + graph | adjacent viewpoint / stop | trajectory overlap graph | route segment, branch, merge | colored route overlay | dataset access, graph format |
| Habitat ObjectNav | egocentric RGB + goal | discrete Habitat actions | shortest-path / bottleneck graph | room entry, search, stop | top-down path overlay | semantic map availability |
| RoboCasa | RGB + state + instruction | continuous or skill action | precondition / option graph | object, receptacle, activation | choice-point graph | action abstraction |

2. **At least three visual sketches**
   - R2R trajectory overlap / stitching graph.
   - Habitat ObjectNav shortest-path or bottleneck sketch.
   - Robotics choice-point / option graph.

3. **One benchmark proposal slide**
   - Input format.
   - Expected output.
   - What counts as correct.
   - Why this tests topology understanding rather than generic language ability.

4. **Source list**
   - Dataset papers/pages.
   - Environment documentation.
   - Any inspected data schema or example episode file.

## Suggested work plan

### Week 1: source survey

- Pick 2 navigation datasets/environments and 1--2 robotics environments.
- Fill the comparison table with source citations.
- Identify whether trajectory graph, action labels, and instruction fields are accessible.

### Week 2: topology extraction examples

- For R2R or a predictive branching dataset, find one concrete example of overlapping
  trajectories or route fragments.
- Draw a small graph showing shared subpath, branch point, and possible stitching.
- For robotics, draw one task as choice points with option sets.

### Week 3: benchmark sketch

- Define 3--5 probe question types.
- For each probe, specify input, output, correctness criterion, and likely failure mode.
- Decide which probes can be evaluated without training a new policy.

### Week 4: presentation

- Prepare a 15--20 minute presentation.
- End with a recommendation: which environment/dataset is best for a first
  TopoVLM topology-understanding benchmark, and why.

## What not to do initially

- Do not start large training jobs.
- Do not claim that a learned latent is topology-aware without a concrete test.
- Do not treat any graph as topology just because it is a graph; explain why the
  graph affects decision-making.
- Do not mix dataset topology, task topology, and learned representation topology
  without labeling them separately.

## Success criteria

The assignment succeeds if it gives a clear answer to these questions:

1. Which navigation or robotics environments expose decision-relevant topology most clearly?
2. Can this topology be visualized from existing annotations or trajectories?
3. Can we build a benchmark that asks existing VLM/VLA models to reason about that topology?
4. Which first benchmark is easiest, most convincing, and least dependent on new training?
