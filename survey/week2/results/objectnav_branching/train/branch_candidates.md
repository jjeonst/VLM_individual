# ObjectNav branching candidates (deliverable #3)

Source: HM3D ObjectNav v2 PR2L expert **action** trajectories
(`/data/topovlm/habitat/episodes/pr2l_hm3d_objectnav/{split}`), 6000 episodes.
Actions: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT. No world positions in the cache
(would require a Habitat sim replay), so branches are characterised in ACTION space.

## Observation & action space (for all rows below)
- **Observation**: egocentric RGB (640×480) + target `object_category` (VLM prompt-conditioned latent in PR2L). No language route instruction.
- **Action**: discrete Habitat `STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT` (turn 30°).

## Branch candidate table

| Candidate | Observation cue | Action choices | Why topology (choice changes future)? | Valid / invalid branch | How to visualize | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| **Start-junction fork** | egocentric RGB at spawn + target category | first turn LEFT vs RIGHT (vs straight) | initial branch sets the whole route to the target; correct side depends on where the target instance is (observation-conditioned) | valid = branch toward the reachable target instance; invalid = away → longer path | fork node→{L,R}→target diagram (Panel A) | 102 scene+object groups split L/R; top: scene `00757-LVgQNuK8vtv`, sofa, 50 routes = 25L / 25R |
| **Spawn-orientation spin** | RGB at spawn (target behind agent) | turn-around (≥180° same-dir run) before moving | the agent must first choose a heading before any route unfolds — the earliest branch | valid = spin toward target bearing; invalid = spin away | action strip, run at step 0 (Panel C) | 376 episodes (6.3%) begin with a ≥180° spin; **100% start at step 0** (mid-route ≥180° backtracks = 0) |
| **Terminal STOP decision** | RGB shows target in view / not | STOP vs keep moving | STOP at the wrong place fails the episode; STOP is the success-defining branch | valid = STOP at target; invalid = premature/late STOP | STOP node in option graph (Panel B) | every episode ends in STOP; 961 (16.0%) STOP immediately (spawn=goal) |
| **Inflection junctions (support)** | RGB at each heading change | FORWARD vs TURN | each FORWARD→TURN is a re-decision of heading at a junction | — | option-transition graph (Panel B) | avg 14.86 FORWARD→TURN transitions per episode |

## Option-transition structure (from FORWARD)
From a FORWARD state the expert chooses:
FORWARD 55% · LEFT 21% · RIGHT 21% · STOP 2%.
These non-FORWARD choices are the decision points that create branch structure.

## Caveats (honest framing)
- This is **action-space** topology, not a metric map. "Start-junction fork" is route-level (spawn/observation-conditioned), not a single physical junction with multiple outgoing edges — reconstructing the latter needs a sim replay to recover positions.
- **Dead-end / mid-route backtrack is NOT separable from actions alone.** Every ≥180° same-direction turn run in the dataset starts at step 0 (0 forward actions before it) — they are spawn-orientation spins. A genuine dead-end recovery (go forward into a dead-end, then return near a previous location) needs **positions from a Habitat sim replay** to detect; it is left as future work, not claimed here.
- Left/right validity cannot be judged from actions alone (depends on target instance location in the RGB); it is labelled by intent, not verified geometrically.
- Figure: `results/objectnav_branching/{split}/objectnav_branching.png`
