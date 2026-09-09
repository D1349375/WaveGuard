"""
test_simulation.py
Tests the Taiwan Xinyi Corridor network and incident injection mechanism.
"""

import pytest
from pathlib import Path
from envs.waveguard_env import WaveGuardEnv
from agents.colight_agent import WaveGuardCoLightAgent
from agents.rule_controllers import FixedTimeController, MaxPressureController

def test_xinyi_corridor_simulation():
    xinyi_cfg = Path(__file__).resolve().parent.parent / "sumo_nets" / "taiwan_xinyi" / "xinyi_corridor.sumocfg"
    assert xinyi_cfg.exists(), "Xinyi sumocfg must exist"

    env = WaveGuardEnv(sumocfg_path=xinyi_cfg, gui=False)
    obs, info = env.reset(seed=123)

    assert len(env.tls_ids) == 6, f"Expected 6 intersections along corridor, got {len(env.tls_ids)}"

    # Run 15 simulation steps
    agent = WaveGuardCoLightAgent(tls_ids=env.tls_ids, grid_size=12, action_dim=4)
    for _ in range(15):
        actions = agent.act(obs, explore=True)
        obs, rewards, term, trunc, info = env.step(actions)

    assert info["step"] == 15
    assert info["throughput_vehicles"] >= 0

    # Test incident injection
    target_edge = "E_J1_J2"
    veh = env.wrapper.inject_incident(target_edge, speed=0.0, duration=10)
    # veh can be None if no vehicle on edge at step 15, or str
    env.close()


def test_phase_transition_actually_holds_yellow_and_all_red():
    """
    Section 2.2 says yellow + all-red is a hard safety constraint the RL agent
    cannot override. The original execute_phase() only called
    traci.trafficlight.setPhase() once at the start of a transition and then
    just counted a Python-side timer down without touching the signal again --
    meaning SUMO's own auto-generated static program (whose yellow phase has a
    fixed 4s duration baked into the .net.xml) could silently auto-advance
    past our configured yellow_time/all_red_time, and there was no all-red
    phase defined at all, so vehicles could see two conflicting greens with no
    real clearance interval between them. This drives an actual phase switch
    and inspects the live signal string every step to confirm: the network is
    genuinely all-red at some point during the transition, and the commanded
    target phase isn't reached before yellow_time + all_red_time have elapsed.
    """
    import traci
    xinyi_cfg = Path(__file__).resolve().parent.parent / "sumo_nets" / "taiwan_xinyi" / "xinyi_corridor.sumocfg"
    env = WaveGuardEnv(sumocfg_path=xinyi_cfg, gui=False)
    obs, info = env.reset(seed=123)

    tid = env.tls_ids[0]
    yellow_time = env.cfg["environment"]["yellow_time"]
    all_red_time = env.cfg["environment"]["all_red_time"]

    # Command a switch to the other movement to force a transition through
    # yellow -> all-red -> new green.
    starting_active = env.wrapper.tls_transition[tid]["active_phase"]
    other_action = 1 if starting_active == 0 else 0

    saw_all_red = False
    reached_target_before_deadline = False
    deadline = yellow_time + all_red_time

    for step in range(deadline + 5):
        env.wrapper.execute_phase(tid, other_action, yellow_time=yellow_time, all_red_time=all_red_time)
        state = traci.trafficlight.getRedYellowGreenState(tid)
        if all(ch == "r" for ch in state):
            saw_all_red = True
        if step < deadline - 1 and not env.wrapper.tls_transition[tid]["in_transition"]:
            reached_target_before_deadline = True
        traci.simulationStep()

    assert saw_all_red, "Transition never produced a genuine all-red signal state"
    assert not reached_target_before_deadline, (
        "Signal reached the target green before yellow_time + all_red_time elapsed"
    )

    env.close()
