# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 501, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |   WaveGuard+FailSafe |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|---------------------:|
| Average Waiting Time (s)   |                    1.53 |           0.68 |             0.44 | 71.2%                     |                 0.52 |
| Motorcycle Avg Wait (s)    |                    1.58 |           0.66 |             0.44 | 72.2%                     |                 0.53 |
| Car Avg Wait (s)           |                    1.45 |           0.7  |             0.44 | 69.7%                     |                 0.52 |
| Equity Wait Gap (s)        |                    0.13 |           0.04 |             0    | 100.0%                    |                 0.01 |
| Throughput (Vehicles)      |                  466    |         464    |           500    | +7.3%                     |               490    |
| TTC Safety Conflicts (<3s) |                  688    |         639    |           674    | 2.0%                      |               657    |
| Mean Halting Queue         |                   36.6  |          25.57 |            15.8  | 56.8%                     |                16.79 |

**Fail-Safe Watchdog**: a simulated sensor/communication outage was injected on steps [100, 200) during the WaveGuard+FailSafe run. The watchdog spent 98 of those steps in `FAILSAFE_MAX_PRESSURE` mode before recovering to `MARL_COOPERATIVE` automatically once the simulated outage ended.

---
*Generated automatically by WaveGuard Evaluation Pipeline*
