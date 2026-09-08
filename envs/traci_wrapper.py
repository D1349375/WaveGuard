"""
traci_wrapper.py
Base TraCI interface wrapper for WaveGuard.
Handles SUMO subprocess lifecycle, TraCI commands, sublane feature queries,
Grid-based BEV feature extraction, and Surrogate Safety Measures (TTC tracking).
"""

import os
import sys
import math
import numpy as np
from pathlib import Path

# Auto-configure SUMO_HOME
if "SUMO_HOME" not in os.environ:
    try:
        import sumo
        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError:
        pass

import traci
import sumolib

class TraciWrapper:
    def __init__(self, sumocfg_path, gui=False, step_length=1.0, lateral_resolution=0.8):
        self.sumocfg_path = str(Path(sumocfg_path).resolve())
        self.gui = gui
        self.step_length = step_length
        self.lateral_resolution = lateral_resolution
        self.is_running = False
        self.tls_ids = []
        self.tls_lane_map = {}
        self.tls_pos_map = {}
        self.step_count = 0
        
        # Performance & Safety tracking
        self.total_completed_trips = 0
        self.ttc_conflict_count = 0
        self.waiting_time_cars = []
        self.waiting_time_motos = []
        
        # Transition state machine for hard yellow/all-red constraints
        # Each TLS tracks: {current_green, target_green, transition_state, remaining_yellow, remaining_allred}
        self.tls_transition = {}

    def start(self, seed=42):
        """Starts SUMO simulation via TraCI."""
        try:
            traci.close()
        except Exception:
            pass

        sumo_binary = "sumo-gui" if self.gui else "sumo"
        cmd = [
            sumo_binary,
            "-c", self.sumocfg_path,
            "--step-length", str(self.step_length),
            "--lateral-resolution", str(self.lateral_resolution),
            "--seed", str(seed),
            "--no-warnings", "true",
            "--no-step-log", "true",
            "--quit-on-end", "true"
        ]

        traci.start(cmd)
        self.is_running = True
        self.step_count = 0
        self.total_completed_trips = 0
        self.ttc_conflict_count = 0
        self.waiting_time_cars = []
        self.waiting_time_motos = []

        # Cache TLS and their controlled lanes & junction coordinates
        self.tls_ids = list(traci.trafficlight.getIDList())
        for tid in self.tls_ids:
            lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tid)))
            self.tls_lane_map[tid] = lanes
            
            # Approximate junction center from junction shape or inbound lanes
            try:
                # Traci junction position
                pos = traci.junction.getPosition(tid)
                self.tls_pos_map[tid] = pos
            except Exception:
                # Fallback to mean position of controlled lane stop points
                xs, ys = [], []
                for lane in lanes:
                    shape = traci.lane.getShape(lane)
                    if shape:
                        xs.append(shape[-1][0])
                        ys.append(shape[-1][1])
                cx = sum(xs) / len(xs) if xs else 0.0
                cy = sum(ys) / len(ys) if ys else 0.0
                self.tls_pos_map[tid] = (cx, cy)

            self.tls_transition[tid] = {
                "active_phase": traci.trafficlight.getPhase(tid),
                "in_transition": False,
                "yellow_timer": 0,
                "red_timer": 0,
                "target_phase": 0
            }

        return self.tls_ids

    def step(self):
        """Advances simulation by one step and tracks metrics."""
        if not self.is_running:
            return

        traci.simulationStep()
        self.step_count += 1

        # Track completed vehicles
        arrived = traci.simulation.getArrivedIDList()
        self.total_completed_trips += len(arrived)

        # Track TTC safety conflicts and waiting times across all active vehicles
        self._track_safety_and_equity()

    def _track_safety_and_equity(self):
        """Calculates TTC safety conflicts and car vs motorcycle waiting time."""
        active_vehs = traci.vehicle.getIDList()
        
        for vid in active_vehs:
            # Equity: record waiting time by vehicle type
            vtype = traci.vehicle.getTypeID(vid)
            wait_time = traci.vehicle.getWaitingTime(vid)
            
            if "moto" in vtype.lower():
                self.waiting_time_motos.append(wait_time)
            else:
                self.waiting_time_cars.append(wait_time)

            # Safety: TTC (Time to Collision) calculation with leading vehicle
            leader_info = traci.vehicle.getLeader(vid, dist=50.0)
            if leader_info is not None:
                lead_id, dist = leader_info
                v_speed = traci.vehicle.getSpeed(vid)
                try:
                    lead_speed = traci.vehicle.getSpeed(lead_id)
                except traci.TraCIException:
                    lead_speed = 0.0

                rel_speed = v_speed - lead_speed
                if rel_speed > 0.5 and dist > 0.1:
                    ttc = dist / rel_speed
                    if ttc < 3.0:  # Section 2.4 specification: TTC < 3.0s
                        self.ttc_conflict_count += 1

    def get_grid_bev(self, tls_id, grid_size=12, grid_range=60.0):
        """
        Implements Section 2.1: Grid-based Bird's Eye View (BEV) Occupancy Matrix.
        Outputs:
          - car_grid: (grid_size, grid_size) occupancy matrix for passenger cars
          - moto_grid: (grid_size, grid_size) occupancy matrix for motorcycles
          - scalar_features: [motorcycle_stop_box_saturation, hook_turn_waiting_count, phase_elapsed_ratio]
        """
        cx, cy = self.tls_pos_map.get(tls_id, (0.0, 0.0))
        car_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        moto_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        stop_box_motos = 0
        stop_box_capacity = 15.0  # Normalized stop-box capacity estimate
        hook_turn_count = 0

        # Query all vehicles near the intersection
        lanes = self.tls_lane_map.get(tls_id, [])
        nearby_vehicles = set()
        for lane in lanes:
            vehs = traci.lane.getLastStepVehicleIDs(lane)
            nearby_vehicles.update(vehs)

        cell_size = (2.0 * grid_range) / grid_size

        for vid in nearby_vehicles:
            try:
                vx, vy = traci.vehicle.getPosition(vid)
                dx = vx - cx
                dy = vy - cy
                
                # Check within grid bounding box [-grid_range, +grid_range]
                if abs(dx) <= grid_range and abs(dy) <= grid_range:
                    gx = int((dx + grid_range) / cell_size)
                    gy = int((dy + grid_range) / cell_size)
                    gx = min(max(gx, 0), grid_size - 1)
                    gy = min(max(gy, 0), grid_size - 1)

                    vtype = traci.vehicle.getTypeID(vid).lower()
                    speed = traci.vehicle.getSpeed(vid)

                    if "moto" in vtype:
                        moto_grid[gx, gy] = min(moto_grid[gx, gy] + 0.33, 1.0)
                        # Stop box check: within 15m of stop line and halted
                        if abs(dx) <= 15.0 and abs(dy) <= 15.0 and speed < 0.5:
                            stop_box_motos += 1
                        # Hook-turn check: waiting in right quadrant for cross turn
                        if dx > 5.0 and dy > 5.0 and speed < 0.5:
                            hook_turn_count += 1
                    else:
                        car_grid[gx, gy] = min(car_grid[gx, gy] + 0.5, 1.0)
            except traci.TraCIException:
                continue

        stop_box_sat = min(float(stop_box_motos) / stop_box_capacity, 1.0)
        hook_turn_norm = min(float(hook_turn_count) / 10.0, 1.0)
        
        curr_phase = traci.trafficlight.getPhase(tls_id)
        phase_ratio = float(curr_phase % 4) / 4.0

        scalar_features = np.array([stop_box_sat, hook_turn_norm, phase_ratio], dtype=np.float32)

        return car_grid, moto_grid, scalar_features

    def get_tls_halting(self, tls_id):
        """Returns the total number of halted vehicles across inbound lanes."""
        lanes = self.tls_lane_map.get(tls_id, [])
        return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    def get_intersection_pressure(self, tls_id):
        """
        Computes the traffic pressure for Max Pressure control and GAT state aggregation.
        Pressure = Sum(Incoming lane queues) - Sum(Outgoing lane queues).
        """
        lanes = self.tls_lane_map.get(tls_id, [])
        in_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        
        # Outgoing lanes estimation
        out_lanes = []
        for lane in lanes:
            edge_id = traci.lane.getEdgeID(lane)
            # Find reverse or downstream edge
            if "E_" in edge_id:
                parts = edge_id.split("_")
                if len(parts) == 3 and parts[2] == tls_id:
                    out_edge = f"E_{parts[2]}_{parts[1]}"
                    out_lanes.append(out_edge)

        out_queue = 0
        for edge in out_lanes:
            try:
                out_queue += traci.edge.getLastStepHaltingNumber(edge)
            except traci.TraCIException:
                pass

        return max(float(in_queue - out_queue * 0.5), 0.0)

    def execute_phase(self, tls_id, target_phase_idx, yellow_time=3, all_red_time=2):
        """
        Enforces Section 2.2 hard safety constraints:
        Agent commands a green phase. If different from current,
        the wrapper automatically inserts Yellow (3s) + All-Red (2s).
        """
        trans = self.tls_transition[tls_id]
        curr_phase = traci.trafficlight.getPhase(tls_id)

        # Map agent green action (0, 1, 2, 3) to SUMO TLS phases
        # For our 4-phase network:
        # Green 0 -> SUMO Phase 0
        # Yellow 0 -> SUMO Phase 1
        # Green 1 -> SUMO Phase 2
        # Yellow 1 -> SUMO Phase 3
        sumo_target = (target_phase_idx * 2) % 4

        if trans["in_transition"]:
            if trans["yellow_timer"] > 0:
                trans["yellow_timer"] -= 1
                if trans["yellow_timer"] == 0:
                    # Transition to all-red
                    trans["red_timer"] = all_red_time
            elif trans["red_timer"] > 0:
                trans["red_timer"] -= 1
                if trans["red_timer"] == 0:
                    # Complete transition to target green
                    trans["in_transition"] = False
                    trans["active_phase"] = trans["target_phase"]
                    traci.trafficlight.setPhase(tls_id, trans["target_phase"])
        else:
            if sumo_target != trans["active_phase"]:
                # Initiate transition: switch to yellow
                yellow_phase = (trans["active_phase"] + 1) % 4
                traci.trafficlight.setPhase(tls_id, yellow_phase)
                trans["in_transition"] = True
                trans["yellow_timer"] = yellow_time
                trans["target_phase"] = sumo_target
            else:
                traci.trafficlight.setPhase(tls_id, sumo_target)

    def get_aggregate_metrics(self):
        """Returns comprehensive summary metrics."""
        avg_wait_car = float(np.mean(self.waiting_time_cars)) if self.waiting_time_cars else 0.0
        avg_wait_moto = float(np.mean(self.waiting_time_motos)) if self.waiting_time_motos else 0.0
        
        all_waits = self.waiting_time_cars + self.waiting_time_motos
        overall_avg_wait = float(np.mean(all_waits)) if all_waits else 0.0

        total_halting = 0
        for lanes in self.tls_lane_map.values():
            for l in lanes:
                total_halting += traci.lane.getLastStepHaltingNumber(l)

        return {
            "step": self.step_count,
            "overall_avg_wait_sec": round(overall_avg_wait, 2),
            "car_avg_wait_sec": round(avg_wait_car, 2),
            "moto_avg_wait_sec": round(avg_wait_moto, 2),
            "equity_wait_delta": round(abs(avg_wait_moto - avg_wait_car), 2),
            "throughput_vehicles": self.total_completed_trips,
            "ttc_conflicts": self.ttc_conflict_count,
            "current_halting_queue": total_halting
        }

    def inject_incident(self, edge_id, speed=0.0, duration=45):
        """
        Injects an incident to simulate collision shockwave.
        Forces the leading vehicle on target edge to halt.
        """
        try:
            vehs = traci.edge.getLastStepVehicleIDs(edge_id)
            if vehs:
                target_veh = vehs[0]
                traci.vehicle.setSpeed(target_veh, speed)
                return target_veh
        except traci.TraCIException:
            pass
        return None

    def close(self):
        """Safely shuts down TraCI."""
        if self.is_running:
            try:
                traci.close()
            except Exception:
                pass
            self.is_running = False
