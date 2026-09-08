"""
rule_controllers.py
Classical and Fail-Safe Controllers for WaveGuard:
1. FixedTimeController: Traditional fixed-time pre-timed signal cycle (Baseline).
2. MaxPressureController: Decentralized Max-Pressure rule controller (Section 2.5 Fail-Safe).
3. FailSafeCoordinator: Watchdog layer that automatically switches to Max-Pressure upon sensor timeout or network degradation.
"""

import numpy as np

class FixedTimeController:
    """
    Fixed-Time (Pre-timed) Signal Controller.
    Cycles through green phases with fixed duration.
    """
    def __init__(self, tls_ids, phase_durations=None):
        self.tls_ids = tls_ids
        # Default: 30s arterial, 20s cross-street, repeated
        self.phase_durations = phase_durations or [30, 20]
        self.current_phase_idx = {tid: 0 for tid in tls_ids}
        self.timer = {tid: 0 for tid in tls_ids}

    def reset(self):
        self.current_phase_idx = {tid: 0 for tid in self.tls_ids}
        self.timer = {tid: 0 for tid in self.tls_ids}

    def act(self, obs=None):
        """Returns action dictionary mapping tls_id -> action index."""
        actions = {}
        for tid in self.tls_ids:
            self.timer[tid] += 1
            curr_idx = self.current_phase_idx[tid]
            duration = self.phase_durations[curr_idx % len(self.phase_durations)]
            
            if self.timer[tid] >= duration:
                self.current_phase_idx[tid] = (curr_idx + 1) % len(self.phase_durations)
                self.timer[tid] = 0

            # Map to 0 (Arterial) or 1 (Cross-street)
            actions[tid] = self.current_phase_idx[tid]
        return actions


class MaxPressureController:
    """
    Max Pressure Traffic Signal Controller.
    Selects the phase that maximizes outgoing pressure differential.
    Formula: a* = argmax_p (sum_{l in in_p} queue_l - sum_{l in out_p} queue_l)
    """
    def __init__(self, tls_ids, min_green=10):
        self.tls_ids = tls_ids
        self.min_green = min_green
        self.active_phase = {tid: 0 for tid in tls_ids}
        self.phase_timer = {tid: 0 for tid in tls_ids}

    def reset(self):
        self.active_phase = {tid: 0 for tid in self.tls_ids}
        self.phase_timer = {tid: 0 for tid in self.tls_ids}

    def act(self, obs):
        actions = {}
        for tid in self.tls_ids:
            self.phase_timer[tid] += 1
            curr_action = self.active_phase[tid]

            # Obey minimum green duration to prevent fluttering
            if self.phase_timer[tid] < self.min_green:
                actions[tid] = curr_action
                continue

            # Max Pressure: argmax_p (sum_{l in in_p} queue_l - sum_{l in out_p} queue_l),
            # grouped by movement axis. Computed from real per-lane queues via
            # TraciWrapper.get_movement_pressures (see envs/traci_wrapper.py),
            # not approximated from BEV grid occupancy cells.
            movement = obs[tid]["movement_pressure"]
            ew_pressure = movement["arterial"]
            ns_pressure = movement["cross"]

            # If motorcycle stop box is saturated (>80%), prioritize arterial clearance
            stop_box_sat = obs[tid]["scalar_feats"][0]
            if stop_box_sat > 0.8:
                best_phase = 0
            elif ew_pressure >= ns_pressure:
                best_phase = 0
            else:
                best_phase = 1

            if best_phase != curr_action:
                self.active_phase[tid] = best_phase
                self.phase_timer[tid] = 0

            actions[tid] = self.active_phase[tid]
        return actions


class FailSafeCoordinator:
    """
    Safety Fallback Mechanism (Section 2.5).
    Monitors communication / sensor health.
    If timeout or anomaly detected, triggers fail-safe switch to Max Pressure.
    """
    def __init__(self, marl_agent, max_pressure_ctrl, heartbeat_timeout=3):
        self.marl_agent = marl_agent
        self.fallback_ctrl = max_pressure_ctrl
        self.heartbeat_timeout = heartbeat_timeout
        self.consecutive_misses = 0
        self.is_fallback_active = False
        self._last_good_actions = None

    def act(self, obs, sim_step=0, force_degrade=False):
        """
        Chooses action via MARL, or degrades to Max Pressure once
        `heartbeat_timeout` consecutive misses have been observed.

        The primary agent is probed on every call (unless force_degrade),
        including while already degraded, so the coordinator recovers
        automatically as soon as inputs are healthy again -- it must not
        permanently latch into fallback mode after the first sustained outage.
        A short run of misses below the timeout is tolerated by re-using the
        last known-good MARL action rather than immediately dropping to the
        rule-based controller, to avoid flapping on a single transient blip.
        """
        if not force_degrade:
            try:
                actions = self.marl_agent.act(obs, explore=False)
                self.consecutive_misses = 0
                self.is_fallback_active = False
                self._last_good_actions = actions
                return actions, "MARL_COOPERATIVE"
            except Exception:
                self.consecutive_misses += 1
        else:
            self.consecutive_misses += 1

        if self.consecutive_misses < self.heartbeat_timeout and self._last_good_actions is not None:
            self.is_fallback_active = False
            return self._last_good_actions, "MARL_COOPERATIVE"

        self.is_fallback_active = True
        actions = self.fallback_ctrl.act(obs)
        return actions, "FAILSAFE_MAX_PRESSURE"
