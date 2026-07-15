# OGBench cube benchmark — manipulation topology (robotics expansion)

Source: OGBench `cube-double-play-v0` (val split, 100 play episodes × 1001 steps).
State: qpos[0:14]=arm, cube i xyz = qpos[:, 14+i*7 : 14+i*7+3]. Action: 5-D continuous
(end-effector Δxyz + wrist + gripper). Observation: 37-D state (no image in this dataset).

Topology here is **task structure**, not a physical map: which cube to act on, the
object-on-object stacking (precondition), and the pick→place→stack skill sequence.

| Candidate | Observation cue | Action / option choices | Why topology (choice changes future)? | Valid / invalid | How to visualize | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **Choice point — which cube** | cube xyz + gripper state | pick cube 0 vs cube 1 | the cube you grasp first sets the rest of the plan | valid = the cube the goal needs first; invalid = the other (must be undone) | choice-point graph (Panel A) | first-moved: cube0 56/100, cube1 44/100; both lifted in 99/100 |
| **Precondition / stacking** | relative cube heights & xy | place-on-table vs stack-on-other | to stack, the base cube must be placed first — an ordering constraint | valid = base before top; invalid = stack before base is set | object-on-object graph (Panel C) | 87% of episodes stack; cube0-on-1 in 62, cube1-on-0 in 64 |
| **Skill / task-stage** | gripper open/close + cube z | reach→grasp→lift→transport→place | each episode is a sequence of pick-place skills; stage order defines the task | valid = correct stage order; invalid = e.g. release before transport | cube-height timeline (Panel B) | avg 9.1% of steps in a stacked config; both cubes manipulated (99, 100 eps) |

## Honest framing
- This is **play** (task-agnostic) data, so it explores **both** stacking orders (cube0-on-1 ≈ cube1-on-0). A specific goal-conditioned task fixes one order → then the precondition graph is directed. So the *dataset* shows the space of options; a *task* selects a valid branch through it.
- "cube" topology is **task/precondition topology**, distinct from R2R/ObjectNav route topology — kept separate per the assignment (dataset vs task vs representation topology).
- Figure: `results/cube_topology.png`. Larger stacking towers (precondition chains) live in `cube-triple` / `cube-quadruple` (1–1.9 GB), not downloaded here.
