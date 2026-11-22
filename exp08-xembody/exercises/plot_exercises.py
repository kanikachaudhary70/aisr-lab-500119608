"""Exercise figure for Experiment 8 (E2: number of demonstrations, E4: scale of kuka_mini).
Run from exp08-xembody/:  python exercises/plot_exercises.py  ->  figs/fig8_6_exercises.png"""
import csv, json, os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 10, 'axes.grid': True, 'grid.color': '0.88', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
                     'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
ROBOTS = ['kuka_iiwa', 'franka_panda', 'xarm6', 'kuka_mini']

def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))

# E2: BC-agnostic success on each robot against the number of KUKA demonstrations
demos = [10, 20, 40, 60]
files = {10: 'exercises/E2_demos10.csv', 20: 'exercises/E2_demos20.csv', 40: 'exercises/E2_demos40.csv',
         60: 'results/04_learn_once_deploy_many_kuka_iiwa.csv'}
e2 = {r: [] for r in ROBOTS}
e2p = {r: [] for r in ROBOTS}
for d in demos:
    for row in rows(files[d]):
        if row['policy'] == 'BC-agnostic':
            e2[row['robot']].append(100*float(row['success_rate']))
        if row['policy'] == 'BC-proprio':
            e2p[row['robot']].append(100*float(row['success_rate']))

# E4: expert task-space success and task coverage against the scale of kuka_mini
scales = [0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
e4_succ = [100*float(rows(f'exercises/E4_scale{s}.csv')[0]['success_rate']) for s in scales]
e4_cov = [100*float([r for r in rows(f'exercises/E4_emb{s}.csv') if r['robot'] == 'kuka_mini'][0]['task_coverage'])
          for s in scales]
e4_reach = [float([r for r in rows(f'exercises/E4_emb{s}.csv') if r['robot'] == 'kuka_mini'][0]['max_reach_m'])
            for s in scales]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
for r, c, m, ls in zip(ROBOTS, C, 'osD^', ['-', '--', '-.', ':']):
    a1.plot(demos, e2[r], marker=m, color=c, ls=ls, ms=5, label=r)
a1.set_xlabel('KUKA demonstrations used for training'); a1.set_ylabel('BC-agnostic success (%)')
a1.set_xticks(demos); a1.set_ylim(0, 105); a1.legend(fontsize=7, loc='lower right')
a1.set_title('E2: fewer demonstrations (20 test episodes)', fontsize=9)
a2.plot(scales, e4_succ, 'o-', color=C[0], ms=5, label='expert success, task space (20 ep.)')
a2.plot(scales, e4_cov, 's--', color=C[1], ms=5, label='task coverage, Part 1 (90 points)')
a2.set_xlabel('scale of kuka_mini (1.0 = full-size KUKA)'); a2.set_ylabel('percentage (%)')
a2.set_ylim(0, 105); a2.legend(fontsize=7, loc='lower right')
a2.set_title('E4: smallest KUKA that still succeeds', fontsize=9)
fig.tight_layout()
os.makedirs('figs', exist_ok=True)
fig.savefig('figs/fig8_6_exercises.png')
json.dump(dict(demos=demos, e2_agnostic=e2, e2_proprio=e2p, scales=scales, e4_success=e4_succ,
               e4_coverage=e4_cov, e4_reach=e4_reach), open('exercises/summary.json', 'w'), indent=1)
print(json.dumps(dict(e2_agnostic=e2, e2_proprio=e2p, e4_success=e4_succ, e4_coverage=e4_cov, e4_reach=e4_reach), indent=1))
