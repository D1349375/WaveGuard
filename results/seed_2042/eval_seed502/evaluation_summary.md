# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 502, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |   WaveGuard+FailSafe |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|---------------------:|
| Average Waiting Time (s)   |                    1.57 |           0.55 |             0.4  | 74.5%                     |                 0.44 |
| Motorcycle Avg Wait (s)    |                    1.59 |           0.56 |             0.4  | 74.8%                     |                 0.43 |
| Car Avg Wait (s)           |                    1.55 |           0.53 |             0.4  | 74.2%                     |                 0.46 |
| Equity Wait Gap (s)        |                    0.04 |           0.02 |             0    | 100.0%                    |                 0.04 |
| Throughput (Vehicles)      |                  412    |         442    |           446    | +8.3%                     |               443    |
| TTC Safety Conflicts (<3s) |                  645    |         617    |           613    | 5.0%                      |               634    |
| Mean Halting Queue         |                   33.78 |          18.79 |            13.69 | 59.5%                     |                15.7  |

**Fail-Safe Watchdog**: a simulated sensor/communication outage was injected on steps [100, 200) during the WaveGuard+FailSafe run. The watchdog spent 98 of those steps in `FAILSAFE_MAX_PRESSURE` mode before recovering to `MARL_COOPERATIVE` automatically once the simulated outage ended.

---
*Generated automatically by WaveGuard Evaluation Pipeline*
