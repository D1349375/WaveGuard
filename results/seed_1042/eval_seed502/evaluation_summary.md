# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 502, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |   WaveGuard+FailSafe |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|---------------------:|
| Average Waiting Time (s)   |                    1.57 |           0.55 |             0.38 | 75.8%                     |                 0.58 |
| Motorcycle Avg Wait (s)    |                    1.59 |           0.56 |             0.38 | 76.1%                     |                 0.61 |
| Car Avg Wait (s)           |                    1.55 |           0.53 |             0.37 | 76.1%                     |                 0.54 |
| Equity Wait Gap (s)        |                    0.04 |           0.02 |             0.01 | 75.0%                     |                 0.07 |
| Throughput (Vehicles)      |                  412    |         442    |           439    | +6.6%                     |               425    |
| TTC Safety Conflicts (<3s) |                  645    |         617    |           616    | 4.5%                      |               598    |
| Mean Halting Queue         |                   33.78 |          18.79 |            13.01 | 61.5%                     |                16.96 |

**Fail-Safe Watchdog**: a simulated sensor/communication outage was injected on steps [100, 200) during the WaveGuard+FailSafe run. The watchdog spent 98 of those steps in `FAILSAFE_MAX_PRESSURE` mode before recovering to `MARL_COOPERATIVE` automatically once the simulated outage ended.

---
*Generated automatically by WaveGuard Evaluation Pipeline*
