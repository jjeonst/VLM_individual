# ObjectNav spatial branch candidates (deliverable #3, position-based)

Source: expert positions recovered by replaying HM3D ObjectNav actions in the
Habitat simulator (`objectnav_replay_positions.py`), 5039
trajectories over 106 scenes. Floor plane (x, z), cell 0.5 m.

These are the **spatial** candidates that action sequences alone could not give.

| Candidate | Observation cue | Action choices | Why topology? | Valid / invalid | How to visualize | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **Doorway / bottleneck** | RGB of a narrow passage many routes use | pass through vs turn away | many distinct routes funnel through one narrow cell — the connectivity chokepoint | valid = pass the connector toward goal; invalid = miss it | trajectory overlay + traffic heat (Panel A) | scene `00440-wPLokgvCnuk`: busiest cell carries **41/51 routes (80%)** at xz=[1.75, -6.75] |
| **Wrong-turn / dead-end** | RGB shows a dead-end / blocked region | go in vs avoid | entering forces a there-and-back detour, changing path length & cost | valid = avoid; invalid = enter, must return | highlighted route with return point (Panel B) | **1208 routes (24.0%)** leave then return within 0.75 m after a ≥2.5 m detour; top: scene `00135-HeSYRw7eMtG`, ep …glb_19, detour 16.5 m returning within 0.64 m |
| **Shared doorway across routes** | same connector seen from different approaches | which side to enter/exit | different start rooms reuse the same doorway (shared subpath, like R2R) | valid = correct connector; invalid = wrong room exit | overlay showing convergence (Panel A) | high-traffic cells = shared connectors across 51 routes in `00440-wPLokgvCnuk` |

## Method notes
- Bottleneck traffic = # of **distinct** trajectories whose path enters a given 0.5 m cell; a doorway is the narrow, high-traffic case.
- Dead-end = a route returns within 0.75 m of an earlier point after travelling ≥2.5 m away (≥12 steps apart) — a genuine there-and-back, which the action-only view could not detect (all ≥180° spins were at spawn).
- Figure: `results/objectnav_branching/{split}/objectnav_topdown.png`
