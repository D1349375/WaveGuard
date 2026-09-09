# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 501, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |   WaveGuard+FailSafe |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|---------------------:|
| Average Waiting Time (s)   |                    1.53 |           0.68 |             0.49 | 68.0%                     |                 0.57 |
| Motorcycle Avg Wait (s)    |                    1.58 |           0.66 |             0.53 | 66.5%                     |                 0.57 |
| Car Avg Wait (s)           |                    1.45 |           0.7  |             0.43 | 70.3%                     |                 0.56 |
| Equity Wait Gap (s)        |                    0.13 |           0.04 |             0.09 | 30.8%                     |                 0.01 |
| Throughput (Vehicles)      |                  466    |         464    |           498    | +6.9%                     |               478    |
| TTC Safety Conflicts (<3s) |                  688    |         639    |           691    | -0.4%                     |               679    |
| Mean Halting Queue         |                   36.6  |          25.57 |            16.55 | 54.8%                     |                18.02 |

**Fail-Safe Watchdog**: a simulated sensor/communication outage was injected on steps [100, 200) during the WaveGuard+FailSafe run. The watchdog spent 98 of those steps in `FAILSAFE_MAX_PRESSURE` mode before recovering to `MARL_COOPERATIVE` automatically once the simulated outage ended.

---
*Generated automatically by WaveGuard Evaluation Pipeline*
