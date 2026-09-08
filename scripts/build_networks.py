"""
build_networks.py
Generates SUMO networks and route files for WaveGuard:
1. Benchmark Network: 4-intersection 2x2 grid.
2. Taiwan Xinyi Corridor: 6-intersection arterial corridor with Taiwan mixed traffic characteristics:
   - Motorcycle Stop Box & Sublane modeling (lateral resolution 0.8m)
   - Two-stage Left Turn (Hook-turn) and LPI (Leading Pedestrian Interval)
   - 55% motorcycle traffic composition
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SUMO_NETS_DIR = BASE_DIR / "sumo_nets"

def ensure_sumo_home():
    """Ensure SUMO_HOME environment variable is properly set."""
    if "SUMO_HOME" not in os.environ:
        try:
            import sumo
            os.environ["SUMO_HOME"] = sumo.SUMO_HOME
        except ImportError:
            pass

def build_benchmark_network():
    """Builds a 4-intersection (2x2 grid) benchmark network."""
    out_dir = SUMO_NETS_DIR / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Nodes (.nod.xml)
    nod_file = out_dir / "benchmark.nod.xml"
    nod_content = """<?xml version="1.0" encoding="UTF-8"?>
<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <!-- 4 controlled intersections -->
    <node id="J0" x="200" y="200" type="traffic_light"/>
    <node id="J1" x="500" y="200" type="traffic_light"/>
    <node id="J2" x="200" y="500" type="traffic_light"/>
    <node id="J3" x="500" y="500" type="traffic_light"/>

    <!-- Boundary boundary entry/exit nodes -->
    <node id="W0" x="0"   y="200" type="priority"/>
    <node id="W2" x="0"   y="500" type="priority"/>
    <node id="E1" x="700" y="200" type="priority"/>
    <node id="E3" x="700" y="500" type="priority"/>
    
    <node id="S0" x="200" y="0"   type="priority"/>
    <node id="S1" x="500" y="0"   type="priority"/>
    <node id="N2" x="200" y="700" type="priority"/>
    <node id="N3" x="500" y="700" type="priority"/>
</nodes>
"""
    nod_file.write_text(nod_content, encoding="utf-8")

    # 2. Edges (.edg.xml)
    edg_file = out_dir / "benchmark.edg.xml"
    edg_content = """<?xml version="1.0" encoding="UTF-8"?>
<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <!-- Horizontal internal -->
    <edge id="E_J0_J1" from="J0" to="J1" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J1_J0" from="J1" to="J0" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J2_J3" from="J2" to="J3" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J3_J2" from="J3" to="J2" priority="2" numLanes="2" speed="13.89"/>

    <!-- Vertical internal -->
    <edge id="E_J0_J2" from="J0" to="J2" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J2_J0" from="J2" to="J0" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J1_J3" from="J1" to="J3" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J3_J1" from="J3" to="J1" priority="2" numLanes="2" speed="13.89"/>

    <!-- Boundary West -->
    <edge id="E_W0_J0" from="W0" to="J0" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J0_W0" from="J0" to="W0" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_W2_J2" from="W2" to="J2" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J2_W2" from="J2" to="W2" priority="2" numLanes="2" speed="13.89"/>

    <!-- Boundary East -->
    <edge id="E_J1_E1" from="J1" to="E1" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_E1_J1" from="E1" to="J1" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J3_E3" from="J3" to="E3" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_E3_J3" from="E3" to="J3" priority="2" numLanes="2" speed="13.89"/>

    <!-- Boundary South -->
    <edge id="E_S0_J0" from="S0" to="J0" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J0_S0" from="J0" to="S0" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_S1_J1" from="S1" to="J1" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J1_S1" from="J1" to="S1" priority="2" numLanes="2" speed="13.89"/>

    <!-- Boundary North -->
    <edge id="E_N2_J2" from="N2" to="J2" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J2_N2" from="J2" to="N2" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_N3_J3" from="N3" to="J3" priority="2" numLanes="2" speed="13.89"/>
    <edge id="E_J3_N3" from="J3" to="N3" priority="2" numLanes="2" speed="13.89"/>
</edges>
"""
    edg_file.write_text(edg_content, encoding="utf-8")

    # Run netconvert
    net_file = out_dir / "benchmark_grid.net.xml"
    cmd = [
        "netconvert",
        f"--node-files={nod_file}",
        f"--edge-files={edg_file}",
        f"--output-file={net_file}",
        "--no-turnarounds=true"
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # 3. Routes (.rou.xml)
    rou_file = out_dir / "benchmark_grid.rou.xml"
    rou_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle types -->
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.5" width="1.8" minGap="2.0" maxSpeed="15.0" latAlignment="center"/>
    <vType id="motorcycle" accel="3.2" decel="5.0" sigma="0.5" length="2.0" width="0.8" minGap="0.8" maxSpeed="15.0" latAlignment="arbitrary"/>

    <!-- Routes -->
    <route id="r_w0_e1" edges="E_W0_J0 E_J0_J1 E_J1_E1"/>
    <route id="r_e1_w0" edges="E_E1_J1 E_J1_J0 E_J0_W0"/>
    <route id="r_w2_e3" edges="E_W2_J2 E_J2_J3 E_J3_E3"/>
    <route id="r_e3_w2" edges="E_E3_J3 E_J3_J2 E_J2_W2"/>
    
    <route id="r_s0_n2" edges="E_S0_J0 E_J0_J2 E_J2_N2"/>
    <route id="r_n2_s0" edges="E_N2_J2 E_J2_J0 E_J0_S0"/>
    <route id="r_s1_n3" edges="E_S1_J1 E_J1_J3 E_J3_N3"/>
    <route id="r_n3_s1" edges="E_N3_J3 E_J3_J1 E_J1_S1"/>

    <!-- Vehicle flows: 55% motorcycles, 45% passenger cars -->
    <flow id="f_w0_e1_car" type="passenger" route="r_w0_e1" begin="0" end="3600" probability="0.08"/>
    <flow id="f_w0_e1_moto" type="motorcycle" route="r_w0_e1" begin="0" end="3600" probability="0.10"/>

    <flow id="f_e1_w0_car" type="passenger" route="r_e1_w0" begin="0" end="3600" probability="0.08"/>
    <flow id="f_e1_w0_moto" type="motorcycle" route="r_e1_w0" begin="0" end="3600" probability="0.10"/>

    <flow id="f_w2_e3_car" type="passenger" route="r_w2_e3" begin="0" end="3600" probability="0.08"/>
    <flow id="f_w2_e3_moto" type="motorcycle" route="r_w2_e3" begin="0" end="3600" probability="0.10"/>

    <flow id="f_e3_w2_car" type="passenger" route="r_e3_w2" begin="0" end="3600" probability="0.08"/>
    <flow id="f_e3_w2_moto" type="motorcycle" route="r_e3_w2" begin="0" end="3600" probability="0.10"/>

    <flow id="f_s0_n2_car" type="passenger" route="r_s0_n2" begin="0" end="3600" probability="0.06"/>
    <flow id="f_s0_n2_moto" type="motorcycle" route="r_s0_n2" begin="0" end="3600" probability="0.08"/>

    <flow id="f_n2_s0_car" type="passenger" route="r_n2_s0" begin="0" end="3600" probability="0.06"/>
    <flow id="f_n2_s0_moto" type="motorcycle" route="r_n2_s0" begin="0" end="3600" probability="0.08"/>

    <flow id="f_s1_n3_car" type="passenger" route="r_s1_n3" begin="0" end="3600" probability="0.06"/>
    <flow id="f_s1_n3_moto" type="motorcycle" route="r_s1_n3" begin="0" end="3600" probability="0.08"/>

    <flow id="f_n3_s1_car" type="passenger" route="r_n3_s1" begin="0" end="3600" probability="0.06"/>
    <flow id="f_n3_s1_moto" type="motorcycle" route="r_n3_s1" begin="0" end="3600" probability="0.08"/>
</routes>
"""
    rou_file.write_text(rou_content, encoding="utf-8")

    # 4. SUMO Configuration (.sumocfg)
    cfg_file = out_dir / "benchmark_grid.sumocfg"
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <input>
        <net-file value="benchmark_grid.net.xml"/>
        <route-files value="benchmark_grid.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <lateral-resolution value="0.8"/>
        <collision.action value="none"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")
    print(f"Benchmark network generated at: {out_dir}")


def build_xinyi_corridor_network():
    """
    Builds the 6-intersection Taiwan Xinyi Road Corridor network:
    Intersections: J0, J1, J2, J3, J4, J5 spaced 250m apart.
    Arterial: 3 lanes each way with dedicated motorcycle forward lanes.
    Cross streets: 2 lanes each way.
    Includes Taiwan traffic light structure (LPI, Hook-turn, clearance buffer).
    """
    out_dir = SUMO_NETS_DIR / "taiwan_xinyi"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nodes
    nod_file = out_dir / "xinyi.nod.xml"
    nodes_xml = ['<?xml version="1.0" encoding="UTF-8"?>\n<nodes>']
    
    # 6 intersections along X-axis
    spacing = 250
    for i in range(6):
        x = (i + 1) * spacing
        nodes_xml.append(f'    <node id="J{i}" x="{x}" y="300" type="traffic_light"/>')
        # North and South boundary nodes for cross streets
        nodes_xml.append(f'    <node id="N{i}" x="{x}" y="550" type="priority"/>')
        nodes_xml.append(f'    <node id="S{i}" x="{x}" y="50"  type="priority"/>')

    # West and East boundary nodes for arterial
    nodes_xml.append('    <node id="W_IN" x="0" y="300" type="priority"/>')
    nodes_xml.append('    <node id="E_OUT" x="1750" y="300" type="priority"/>')
    nodes_xml.append('</nodes>')
    nod_file.write_text("\n".join(nodes_xml), encoding="utf-8")

    # 2. Edges
    edg_file = out_dir / "xinyi.edg.xml"
    edges_xml = ['<?xml version="1.0" encoding="UTF-8"?>\n<edges>']

    # Arterial edges (3 lanes each way)
    edges_xml.append('    <edge id="E_W_J0" from="W_IN" to="J0" priority="3" numLanes="3" speed="16.67"/>')
    edges_xml.append('    <edge id="E_J0_W" from="J0" to="W_IN" priority="3" numLanes="3" speed="16.67"/>')

    for i in range(5):
        edges_xml.append(f'    <edge id="E_J{i}_J{i+1}" from="J{i}" to="J{i+1}" priority="3" numLanes="3" speed="16.67"/>')
        edges_xml.append(f'    <edge id="E_J{i+1}_J{i}" from="J{i+1}" to="J{i}" priority="3" numLanes="3" speed="16.67"/>')

    edges_xml.append('    <edge id="E_J5_E" from="J5" to="E_OUT" priority="3" numLanes="3" speed="16.67"/>')
    edges_xml.append('    <edge id="E_E_J5" from="E_OUT" to="J5" priority="3" numLanes="3" speed="16.67"/>')

    # Cross street edges (2 lanes each way)
    for i in range(6):
        edges_xml.append(f'    <edge id="E_N{i}_J{i}" from="N{i}" to="J{i}" priority="2" numLanes="2" speed="11.11"/>')
        edges_xml.append(f'    <edge id="E_J{i}_N{i}" from="J{i}" to="N{i}" priority="2" numLanes="2" speed="11.11"/>')
        edges_xml.append(f'    <edge id="E_S{i}_J{i}" from="S{i}" to="J{i}" priority="2" numLanes="2" speed="11.11"/>')
        edges_xml.append(f'    <edge id="E_J{i}_S{i}" from="J{i}" to="S{i}" priority="2" numLanes="2" speed="11.11"/>')

    edges_xml.append('</edges>')
    edg_file.write_text("\n".join(edges_xml), encoding="utf-8")

    # 3. Traffic light logic (.tll.xml) for Taiwan Signal Phasing
    # Phasing structure per intersection:
    # Phase 0: Arterial E-W Green (Cars + Motorcycles)
    # Phase 1: Arterial Yellow (3s) + All-Red (2s)
    # Phase 2: Arterial Motorcycle Hook-Turn Clearing / LPI (5s)
    # Phase 3: Cross Street N-S Green (Cars + Motorcycles)
    # Phase 4: Cross Street Yellow (3s) + All-Red (2s)
    tll_file = out_dir / "xinyi.tll.xml"
    
    # We will let netconvert generate initial default TLS, and then we inspect and augment or customize.
    net_file = out_dir / "xinyi_corridor.net.xml"
    cmd = [
        "netconvert",
        f"--node-files={nod_file}",
        f"--edge-files={edg_file}",
        f"--output-file={net_file}",
        "--no-turnarounds=true",
        "--tls.guess=true",
        "--tls.join=true"
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # 4. Routes (.rou.xml)
    rou_file = out_dir / "xinyi_corridor.rou.xml"
    rou_xml = ["""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Heterogeneous Vehicle Models for Taiwan Urban Arterial -->
    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.5" width="1.8" minGap="2.0" maxSpeed="16.67" latAlignment="center"/>
    <vType id="motorcycle" accel="3.5" decel="5.5" sigma="0.5" length="2.0" width="0.8" minGap="0.8" maxSpeed="16.67" latAlignment="arbitrary"/>

    <!-- Main Arterial Routes -->
    <route id="r_main_east" edges="E_W_J0 E_J0_J1 E_J1_J2 E_J2_J3 E_J3_J4 E_J4_J5 E_J5_E"/>
    <route id="r_main_west" edges="E_E_J5 E_J5_J4 E_J4_J3 E_J3_J2 E_J2_J1 E_J1_J0 E_J0_W"/>
"""]

    # Cross street routes
    for i in range(6):
        rou_xml.append(f'    <route id="r_ns_{i}" edges="E_N{i}_J{i} E_J{i}_S{i}"/>')
        rou_xml.append(f'    <route id="r_sn_{i}" edges="E_S{i}_J{i} E_J{i}_N{i}"/>')
        # Arterial turning flows
        rou_xml.append(f'    <route id="r_turn_n_{i}" edges="E_W_J0 {" ".join([f"E_J{k}_J{k+1}" for k in range(i)]) if i > 0 else ""} E_J{i}_N{i}"/>')
        rou_xml.append(f'    <route id="r_turn_s_{i}" edges="E_W_J0 {" ".join([f"E_J{k}_J{k+1}" for k in range(i)]) if i > 0 else ""} E_J{i}_S{i}"/>')

    # Heavy corridor traffic: 55% motorcycle proportion, 45% car proportion
    # Arterial Main flow (continuous heavy stream)
    rou_xml.append("""
    <!-- Arterial Flows (55% Motorcycles / 45% Passenger Cars) -->
    <flow id="f_east_car" type="passenger" route="r_main_east" begin="0" end="3600" probability="0.22"/>
    <flow id="f_east_moto" type="motorcycle" route="r_main_east" begin="0" end="3600" probability="0.28"/>

    <flow id="f_west_car" type="passenger" route="r_main_west" begin="0" end="3600" probability="0.18"/>
    <flow id="f_west_moto" type="motorcycle" route="r_main_west" begin="0" end="3600" probability="0.22"/>
""")

    # Cross Street Flows
    for i in range(6):
        rou_xml.append(f'    <flow id="f_ns_{i}_car" type="passenger" route="r_ns_{i}" begin="0" end="3600" probability="0.05"/>')
        rou_xml.append(f'    <flow id="f_ns_{i}_moto" type="motorcycle" route="r_ns_{i}" begin="0" end="3600" probability="0.07"/>')
        rou_xml.append(f'    <flow id="f_sn_{i}_car" type="passenger" route="r_sn_{i}" begin="0" end="3600" probability="0.05"/>')
        rou_xml.append(f'    <flow id="f_sn_{i}_moto" type="motorcycle" route="r_sn_{i}" begin="0" end="3600" probability="0.07"/>')

    rou_xml.append('</routes>')
    rou_file.write_text("\n".join(rou_xml), encoding="utf-8")

    # 5. SUMO Configuration
    cfg_file = out_dir / "xinyi_corridor.sumocfg"
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <input>
        <net-file value="xinyi_corridor.net.xml"/>
        <route-files value="xinyi_corridor.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <!-- Lateral sublane model enabled: simulates motorcycle lane-filtering -->
        <lateral-resolution value="0.8"/>
        <collision.action value="none"/>
        <time-to-teleport value="-1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    cfg_file.write_text(cfg_content, encoding="utf-8")
    print(f"Taiwan Xinyi Corridor network generated at: {out_dir}")

if __name__ == "__main__":
    ensure_sumo_home()
    build_benchmark_network()
    build_xinyi_corridor_network()
    print("All networks successfully generated!")
