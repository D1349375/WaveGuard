"""
test_run_scenario.py
Unit tests for the shockwave dissipation demo's analysis functions (pure
data-processing logic, no SUMO required).
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.run_scenario import calculate_dissipation, calculate_excess_congestion


def _make_df(steps, queues):
    return pd.DataFrame({"step": steps, "j2_incident_queue": queues})


def test_dissipation_ignores_a_single_step_trough_in_an_oscillating_queue():
    """
    Reproduces the exact failure mode found when checking a real demo run: the
    baseline (pre-incident) queue oscillates with the signal cycle, so it is
    easy for a single post-incident sample to dip below the recovery threshold
    purely by oscillation, then climb straight back into double digits a few
    steps later. The original single-sample check reported "dissipated
    instantly" (0 seconds) for exactly this shape of data, which is not a
    credible claim. calculate_dissipation must require the queue to *stay*
    low for several consecutive steps before declaring recovery.
    """
    trigger_step = 250
    duration = 45
    incident_end = trigger_step + duration  # 295

    # Baseline window [200, 250): oscillates between 2 and 10, mean ~= 6 -> threshold ~7.5
    baseline_steps = list(range(200, 250))
    baseline_queue = [2, 10] * (len(baseline_steps) // 2)

    # Post-incident: one lucky low sample (4) right at incident_end that would
    # trivially satisfy a single-sample check, then it spikes back up into a
    # real, sustained jam (kept well clear of the ~8-vehicle threshold so the
    # assertion doesn't depend on exact margin tuning) before finally settling
    # down for good at step 305.
    post_steps = list(range(incident_end, incident_end + 60))
    post_queue = (
        [4]                                  # step 295: single-step trough (the trap)
        + [18, 22, 25, 28, 24, 20, 16]        # genuine post-incident congestion, well above threshold
        + [3] * 52                           # genuinely settled from here on
    )
    post_queue = post_queue[:len(post_steps)]

    df = _make_df(baseline_steps + post_steps, baseline_queue + post_queue)

    dissipation = calculate_dissipation(df, trigger_step, duration, sustain_steps=10)

    # The single trough at step 295 must NOT be reported as the recovery point.
    assert dissipation != 0, "A single noisy low sample was wrongly accepted as full recovery"
    # Real recovery starts once the queue is genuinely and durably low again
    # (step 303 onward in this fixture), not at the single decoy trough.
    assert dissipation >= 8


def test_dissipation_still_detects_genuine_fast_recovery():
    """Sanity check: a queue that drops and *stays* down should still be
    reported as a fast recovery -- the sustained-window fix must not make the
    metric spuriously pessimistic."""
    trigger_step = 100
    duration = 20
    incident_end = trigger_step + duration  # 120

    baseline_steps = list(range(50, 100))
    baseline_queue = [4] * len(baseline_steps)  # stable baseline, threshold = 5.0

    post_steps = list(range(incident_end, incident_end + 40))
    post_queue = [4] * len(post_steps)  # recovers immediately and stays recovered

    df = _make_df(baseline_steps + post_steps, baseline_queue + post_queue)

    dissipation = calculate_dissipation(df, trigger_step, duration, sustain_steps=10)
    assert dissipation == 0


def test_dissipation_does_not_penalize_a_well_controlled_low_baseline():
    """
    Reproduces a real bug found comparing an actual Fixed-Time vs WaveGuard-AI
    demo run: a pure 25%-of-baseline threshold degenerates when the baseline
    itself is small. A corridor that's well-controlled enough to have a tiny
    ~3-vehicle baseline queue gets only a ~0.75-vehicle recovery band -- tighter
    than ordinary signal-cycle noise -- so normal fluctuation between 2 and 5
    vehicles (never anywhere near the original incident's severity) kept
    failing the threshold and made the well-controlled run look like it took
    far longer to "recover" than a run with a larger, looser baseline. The
    `min_margin` floor exists so a low baseline doesn't get an unrealistically
    tight recovery band.
    """
    trigger_step = 250
    duration = 45
    incident_end = trigger_step + duration

    baseline_steps = list(range(200, 250))
    baseline_queue = [3] * len(baseline_steps)  # baseline = 3.0 -> pure 25% band is only 0.75 vehicles wide

    # Ordinary post-recovery cycle noise oscillating 2-5 vehicles -- clearly
    # "back to normal" for a corridor whose baseline is 3, not a lingering jam.
    post_steps = list(range(incident_end, incident_end + 30))
    post_queue = ([2, 5, 3, 4, 2, 5, 3, 4] * 4)[:len(post_steps)]

    df = _make_df(baseline_steps + post_steps, baseline_queue + post_queue)

    dissipation = calculate_dissipation(df, trigger_step, duration, sustain_steps=10, min_margin=2.0)
    assert dissipation == 0, (
        "Ordinary cycle noise around a low baseline was misread as unresolved congestion"
    )


def test_excess_congestion_is_not_fooled_by_a_brief_calm_patch_before_a_real_second_wave():
    """
    Reproduces a real bug found in an actual demo run: right after the
    incident ended, the queue happened to sit low for exactly the length of
    calculate_dissipation()'s sustain window, then surged into a second,
    clearly-still-congested wave shortly after -- so the threshold-crossing
    metric declared "recovered" while real congestion was still visibly
    ongoing a few samples later. calculate_excess_congestion integrates area
    above baseline over a fixed window instead of picking one instant, so it
    must reflect that second wave rather than being fooled by the early dip.
    """
    trigger_step = 250
    duration = 45
    incident_end = trigger_step + duration  # 295

    baseline_steps = list(range(200, 250))
    baseline_queue = [3] * len(baseline_steps)  # baseline = 3.0

    # First 10 steps look "recovered" (values near baseline), then a real,
    # sustained second wave hits before finally settling down.
    calm_patch = [3, 4, 3, 4, 3, 4, 3, 4, 3, 4]
    second_wave = [9, 12, 15, 18, 20, 18, 15, 12, 9, 6]
    settled = [3] * 40
    post_queue = calm_patch + second_wave + settled
    post_steps = list(range(incident_end, incident_end + len(post_queue)))

    df = _make_df(baseline_steps + post_steps, baseline_queue + post_queue)

    # The threshold-crossing metric is expected to be fooled here (that's the
    # documented weakness) -- confirm the failure mode exists...
    fooled = calculate_dissipation(df, trigger_step, duration, sustain_steps=10, min_margin=2.0)
    assert fooled == 0

    # ...and confirm the robust metric correctly reflects the real, substantial
    # second wave instead of reporting near-zero excess congestion.
    excess, peak = calculate_excess_congestion(df, trigger_step, duration, window=60)
    assert peak >= 20
    assert excess > 50
