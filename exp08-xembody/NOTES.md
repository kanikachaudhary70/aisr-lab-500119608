# Experiment 8: cross-embodiment transfer (PyBullet)

Run on 29 Oct 2026 on my laptop (macOS 27.0.1, Apple M1), project venv: Python 3.11.17, PyBullet 3.2.7,
NumPy 2.4.6, scikit-learn 1.9.1, Matplotlib 3.11.2, Pillow 13.0.0.

## Default runs (results/)
```
python experiments/00_check_setup.py            # SETUP OK
python experiments/01_explore_embodiments.py
python experiments/02_joint_space_transfer.py --episodes 10
python experiments/03_task_space_transfer.py --episodes 20
python experiments/04_learn_once_deploy_many.py
python experiments/05_multi_embodiment.py
```
All seeds in the project are fixed, so these numbers match the reference run in MANUAL.md Appendix A.

## Exercises (exercises/)
- **E2, fewer demonstrations:** Part 4 with `--demos 10, 20, 40` -> `E2_demos*.csv`.
- **E4, smallest KUKA that still succeeds:** `scale` of kuka_mini in `xembody/robots.py` set to 0.7 to 1.0,
  then Part 1 (`E4_emb*.csv`) and Part 3 for kuka_mini with 20 episodes (`E4_scale*.csv`).
  This was done in a copy of the project so the default results were not overwritten.
- `run_exercises.sh` runs both; `plot_exercises.py` writes `summary.json` and `../figs/fig8_6_exercises.png`.
