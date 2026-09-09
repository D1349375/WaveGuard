# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 100, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |   WaveGuard+FailSafe |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|---------------------:|
| Average Waiting Time (s)   |                    1.54 |           0.58 |             0.45 | 70.8%                     |                 0.51 |
| Motorcycle Avg Wait (s)    |                    1.54 |           0.53 |             0.44 | 71.4%                     |                 0.5  |
| Car Avg Wait (s)           |                    1.55 |           0.66 |             0.45 | 71.0%                     |                 0.52 |
| Equity Wait Gap (s)        |                    0.01 |           0.13 |             0    | 100.0%                    |                 0.01 |
| Throughput (Vehicles)      |                  462    |         479    |           513    | +11.0%                    |               490    |
| TTC Safety Conflicts (<3s) |                  673    |         661    |           706    | -4.9%                     |               691    |
| Mean Halting Queue         |                   35.65 |          21.6  |            14.61 | 59.0%                     |                17.66 |

**Fail-Safe Watchdog**: a simulated sensor/communication outage was injected on steps [100, 200) during the WaveGuard+FailSafe run. The watchdog spent 98 of those steps in `FAILSAFE_MAX_PRESSURE` mode before recovering to `MARL_COOPERATIVE` automatically once the simulated outage ended.

---
*Generated automatically by WaveGuard Evaluation Pipeline*
