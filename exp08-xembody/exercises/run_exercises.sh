#!/usr/bin/env bash
# Exercises E2 and E4 of Experiment 8. Run from the root of a COPY of AISR-Cross-Embodiment-PyBullet
# (E4 edits xembody/robots.py; the original is restored at the end). Outputs go to $OUT.
#   OUT=/path/to/aisr-lab-500119608/exp08-xembody/exercises bash run_exercises.sh
set -e
PY=${PY:-.venv/bin/python}
OUT=${OUT:-exercises_out}
mkdir -p "$OUT"
# E2: fewer demonstrations (60 is the default run)
for n in 10 20 40; do
  $PY experiments/04_learn_once_deploy_many.py --demos $n
  cp results/04_learn_once_deploy_many_kuka_iiwa.csv "$OUT/E2_demos$n.csv"
done
# E4: smallest KUKA that still succeeds (scale of kuka_mini)
cp xembody/robots.py robots_orig.py
for sc in 0.7 0.75 0.8 0.85 0.9 1.0; do
  sed "s/scale=0.7,/scale=$sc,/" robots_orig.py > xembody/robots.py
  $PY experiments/03_task_space_transfer.py --robots kuka_mini --episodes 20
  cp results/03_task_space_transfer.csv "$OUT/E4_scale$sc.csv"
  $PY experiments/01_explore_embodiments.py
  cp results/01_embodiments.csv "$OUT/E4_emb$sc.csv"
done
mv robots_orig.py xembody/robots.py
echo "Done. Now run: python plot_exercises.py (in exp08-xembody/exercises)"
