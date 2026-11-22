#!/usr/bin/env bash
# Experiment 7: every run needed for Tables 7.1-7.3, three repeats per setting.
# Run from inside a clone of github.com/reshamrajs/webots-robotics-sim with Webots R2025a installed:
#   bash /path/to/aisr-lab-500119608/exp07-webots/run_all.sh
# Each run is saved under a unique --name so repeats do not overwrite each other.
# Then copy analysis/runs/*.csv into exp07-webots/runs/ and run analyse_runs.py.
set -e
PY=${PY:-python3}

# Part A: e-puck swarm, 120 s, default vs cruise = 0.9 (Table 7.1)
for r in 1 2 3; do
  $PY tools/run_experiment.py swarm --seconds 120 --name swarm_default_r$r
  $PY tools/run_experiment.py swarm --seconds 120 --set cruise=0.9 --name swarm_cruise-0.9_r$r
done

# Part B: Pioneer 3-DX with lidar, 180 s (Table 7.2)
$PY tools/run_experiment.py pioneer --seconds 180 --name pioneer_default
$PY tools/run_experiment.py pioneer --seconds 180 --set bubble=0.15 --name pioneer_bubble-0.15
$PY tools/run_experiment.py pioneer --seconds 180 --set k_turn=0.5 --name pioneer_k_turn-0.5

# Exercise 1 (GUIDE.md section 10): speed vs safety, cruise sweep, 3 repeats each (Table 7.3, Figure 7.4)
for c in 0.2 0.4 0.6 0.8 1.0; do
  for r in 1 2 3; do
    $PY tools/run_experiment.py swarm --seconds 120 --set cruise=$c --name ex1_cruise-${c}_r$r
  done
done

# Exercise 4 (GUIDE.md section 10): time step. Edit worlds/swarm_arena.wbt: basicTimeStep 64, run, then 8, run.
#   $PY tools/run_experiment.py swarm --seconds 120 --name ex4_dt64_r1   (repeat r2, r3)
#   $PY tools/run_experiment.py swarm --seconds 120 --name ex4_dt8_r1    (repeat r2, r3)
echo "All runs done. CSVs are in analysis/runs/."
