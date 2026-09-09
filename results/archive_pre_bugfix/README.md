# Archived: pre-bugfix model & evaluation (2026-09-09)

These files are Gemini's original training/evaluation output, produced **before**
a code review found and fixed several real bugs in `envs/traci_wrapper.py` and
`agents/rule_controllers.py` (hand-rolled TTC instead of SUMO's SSM device,
yellow/all-red clearance not actually being held, Max-Pressure using a BEV-grid
heuristic instead of real lane queues, FailSafeCoordinator never wired in and
unable to recover once degraded). Kept here for reference / before-after
comparison. See the top-level `results/` folder for the current, corrected
numbers, and the session transcript for the full list of fixes.

- `best_colight_model_pre_bugfix.pt` -- the actual trained weights are still
  valid and loadable (architecture unchanged); it was just trained in an
  environment that didn't enforce all-red clearance and used a noisier,
  spec-violating safety reward signal. Re-evaluating this SAME checkpoint with
  the corrected code (`results/evaluation_metrics.csv`) shows wait-time/
  throughput/queue improvements hold up, but the previously-reported 6.3%
  safety (TTC) improvement over Fixed-Time drops to ~0.3% once measured
  correctly -- i.e. that specific claim was largely a counting-method artifact.
- `training_log_pre_bugfix.csv` / `training_curves_pre_bugfix.png` -- the
  original 10-episode training run that produced the checkpoint above.
- `evaluation_metrics_pre_bugfix.csv` / `evaluation_summary_pre_bugfix.md` --
  the original (bug-affected) evaluation numbers, superseded by the
  corrected files now in `results/`.
