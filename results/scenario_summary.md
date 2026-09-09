# WaveGuard Ghost Traffic Jam Scenario: Shockwave Impact Report

Incident: stalled `passenger` vehicle on `E_J1_J2`, T=250s to 295s.

## Primary metric: excess queuing at the incident junction (J2)

Total queue-vehicles above each mode's own pre-incident baseline, integrated over the 150s after the incident ends. Chosen over a single-instant "recovery time" because that number turned out to be sensitive to normal signal-cycle noise (see calculate_dissipation() docstring in this script for the specific false positive it produced).

| Metric | Fixed-Time | WaveGuard AI | Change |
|---|---|---|---|
| Peak queue (vehicles) | 33 | 12 | -21 |
| Excess congestion (vehicle-seconds, 150s window) | 657 | 136 | -79.3% |

## Reference only (noise-sensitive, not the headline claim)

Threshold-crossing recovery time: ~11s (Fixed-Time) vs ~0s (WaveGuard AI).
