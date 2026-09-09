# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 501, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |   WaveGuard+FailSafe |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|---------------------:|
| Average Waiting Time (s)   |                    1.53 |           0.68 |             0.45 | 70.6%                     |                 0.54 |
| Motorcycle Avg Wait (s)    |                    1.58 |           0.66 |             0.48 | 69.6%                     |                 0.53 |
| Car Avg Wait (s)           |                    1.45 |           0.7  |             0.42 | 71.0%                     |                 0.56 |
| Equity Wait Gap (s)        |                    0.13 |           0.04 |             0.06 | 53.8%                     |                 0.03 |
| Throughput (Vehicles)      |                  466    |         464    |           506    | +8.6%                     |               480    |
| TTC Safety Conflicts (<3s) |                  688    |         639    |           665    | 3.3%                      |               641    |
| Mean Halting Queue         |                   36.6  |          25.57 |            16.32 | 55.4%                     |                18.72 |

**Fail-Safe Watchdog**: a simulated sensor/communication outage was injected on steps [100, 200) during the WaveGuard+FailSafe run. The watchdog spent 98 of those steps in `FAILSAFE_MAX_PRESSURE` mode before recovering to `MARL_COOPERATIVE` automatically once the simulated outage ended.

---
*Generated automatically by WaveGuard Evaluation Pipeline*
