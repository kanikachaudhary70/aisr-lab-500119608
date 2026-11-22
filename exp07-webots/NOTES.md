# Experiment 7: Webots (reactive swarm and lidar navigation)

Webots R2025a does not run on my laptop (macOS 27.0.1, Apple M1). As agreed with the course teacher, the simulations
were run on 15 October 2026 by my laboratory partner **Aditya Pratap Singh Sisodia (SAP ID 500121985)** on his computer
(Windows 11 Home, Intel Core i5-11400H, Python 3.12.10, Webots R2025a).

- `figs/*_partner.png`: his screenshot and plots (Figures 7.1 to 7.4 of the report).
- The values in Tables 7.1 to 7.3 of the report are his measurements.
- Runs: swarm default and cruise = 0.9 (3 repeats each, 120 s); Pioneer with bubble 0.45 (default), 0.15 and 0.60 m (180 s);
  Exercise 1 (cruise 0.2 to 1.0, 3 repeats each); Exercise 5 (lidar noise 0.001 and 0.003 m).

`run_all.sh` and `analyse_runs.py` are my scripts for repeating the runs and computing the metrics:
path length = sum of distances between logged poses; mean speed = path / duration; stuck % = share of samples
slower than 1 cm/s; coverage = share of grid cells visited (10 cm cells in the 1.6 m arena, 50 cm in the 8 m room).
