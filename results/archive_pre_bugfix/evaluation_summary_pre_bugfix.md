# WaveGuard Quantitative Evaluation Report

**Network**: taiwan_xinyi (Seed: 100, Steps: 300)

| Metric                     |   Fixed-Time (Baseline) |   Max-Pressure |   WaveGuard (AI) | Improvement vs Baseline   |
|:---------------------------|------------------------:|---------------:|-----------------:|:--------------------------|
| Average Waiting Time (s)   |                    1.38 |           0.47 |             0.7  | 49.3%                     |
| Motorcycle Avg Wait (s)    |                    1.4  |           0.52 |             0.73 | 47.9%                     |
| Car Avg Wait (s)           |                    1.35 |           0.4  |             0.66 | 51.1%                     |
| Equity Wait Gap (s)        |                    0.05 |           0.11 |             0.07 | -40.0%                    |
| Throughput (Vehicles)      |                  469    |         497    |           494    | +5.3%                     |
| TTC Safety Conflicts (<3s) |                 2842    |        2418    |          2664    | 6.3%                      |
| Mean Halting Queue         |                   33.04 |          14.05 |            15.88 | 51.9%                     |

---
*Generated automatically by WaveGuard Evaluation Pipeline*
