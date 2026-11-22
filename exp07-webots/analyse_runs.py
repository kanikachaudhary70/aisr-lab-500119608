"""Experiment 7 analysis: turn the run CSVs into the numbers for Tables 7.1-7.3 and Figures 7.2-7.4.

Usage (from exp07-webots/, after copying the Webots analysis/runs/*.csv into runs/):
    python analyse_runs.py
Metrics follow analysis/metrics.py of the webots-robotics-sim repository:
  path (m), mean speed = path / duration, stuck % = share of samples slower than 1 cm/s,
  coverage = share of grid cells visited (10 cm cells on the 1.6 m arena, 50 cm cells in the 8 m room).
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 10, 'axes.grid': True, 'grid.color': '0.88', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.4,
                     'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
GRID = {'swarm': (0.8, 0.1), 'pioneer': (4.0, 0.5)}
os.makedirs('figs', exist_ok=True)


def load(path):
    tracks = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            tracks[r['robot']].append((float(r['t']), float(r['x']), float(r['y'])))
    return tracks


def metrics(path, world):
    half, cell = GRID[world]
    n = int(round(2*half/cell))
    tracks, visited, per_robot = load(path), set(), {}
    for name, p in sorted(tracks.items()):
        length, stuck = 0.0, 0
        for (t0, x0, y0), (t1, x1, y1) in zip(p, p[1:]):
            d = math.hypot(x1 - x0, y1 - y0)
            length += d
            stuck += d/max(t1 - t0, 1e-9) < 0.01
        for _, x, y in p:
            visited.add((min(n - 1, int((x + half)/cell)), min(n - 1, int((y + half)/cell))))
        dur = p[-1][0] - p[0][0]
        per_robot[name] = dict(path=length, speed=length/max(dur, 1e-9), stuck=100*stuck/max(len(p) - 1, 1))
    return dict(robots=per_robot, coverage=100*len(visited)/n**2,
                path=statistics.mean(r['path'] for r in per_robot.values()),
                speed=statistics.mean(r['speed'] for r in per_robot.values()),
                stuck_mean=statistics.mean(r['stuck'] for r in per_robot.values()),
                stuck_worst=max(r['stuck'] for r in per_robot.values()))


def mean_sd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


def plot_tracks(path, out, title):
    tracks = load(path)
    world = 'pioneer' if 'pioneer' in os.path.basename(path) else 'swarm'
    half, cell = GRID[world]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.5))
    allp = []
    for (name, p), c in zip(sorted(tracks.items()), C):
        p = np.array([(x, y) for _, x, y in p]); allp.append(p)
        a1.plot(p[:, 0], p[:, 1], color=c, lw=1.0, label=name); a1.plot(*p[0], 'o', color=c, ms=4)
    a1.set(xlim=(-half, half), ylim=(-half, half), aspect='equal', xlabel='x (m)', ylabel='y (m)')
    a1.legend(fontsize=7, loc='upper right'); a1.set_title('paths (dot = start)', fontsize=9)
    allp = np.vstack(allp); bins = int(round(2*half/cell))
    H, _, _ = np.histogram2d(allp[:, 0], allp[:, 1], bins=bins, range=[[-half, half], [-half, half]])
    im = a2.imshow((H.T > 0).astype(float), origin='lower', extent=(-half, half, -half, half), cmap='Blues', vmin=0, vmax=1.4)
    a2.set(xlabel='x (m)', ylabel='y (m)'); a2.grid(False)
    a2.set_title(f'visited {int(cell*100)} cm cells: {100*(H > 0).mean():.0f}% coverage', fontsize=9)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(out); plt.close(fig)


# ---------- Table 7.1: swarm, default vs cruise = 0.9 ----------
print('Table 7.1  Part A, e-puck swarm (mean over 5 robots per run; stuck = worst robot)')
for setting in ('swarm_default', 'swarm_cruise-0.9'):
    files = sorted(glob.glob(f'runs/{setting}_r*.csv'))
    if not files:
        continue
    ms = [metrics(f, 'swarm') for f in files]
    for i, m in enumerate(ms, 1):
        print(f'  {setting:<18} run {i}: path {m["path"]:.2f} m, speed {m["speed"]:.3f} m/s, '
              f'stuck {m["stuck_worst"]:.1f} % (worst), coverage {m["coverage"]:.0f} %')
    for key, lab in (('path', 'path'), ('speed', 'speed'), ('stuck_worst', 'stuck'), ('coverage', 'coverage')):
        mu, sd = mean_sd([m[key] for m in ms])
        print(f'    mean ± SD {lab}: {mu:.3f} ± {sd:.3f}')

# ---------- Table 7.2: Pioneer ----------
print('\nTable 7.2  Part B, Pioneer 3-DX')
for f in sorted(glob.glob('runs/pioneer_*.csv')):
    m = metrics(f, 'pioneer')
    print(f'  {os.path.basename(f):<28} path {m["path"]:.1f} m, stuck {m["stuck_worst"]:.0f} %, coverage {m["coverage"]:.0f} %')

# ---------- Exercise 1: cruise sweep -> Figure 7.4 ----------
sweep = defaultdict(list)
for f in glob.glob('runs/ex1_cruise-*_r*.csv'):
    c = float(re.search(r'cruise-([\d.]+)_r', f).group(1))
    sweep[c].append(metrics(f, 'swarm'))
if sweep:
    cs = sorted(sweep)
    cov = [mean_sd([m['coverage'] for m in sweep[c]]) for c in cs]
    stk = [mean_sd([m['stuck_worst'] for m in sweep[c]]) for c in cs]
    print('\nExercise 1  cruise sweep (mean ± SD over repeats)')
    for c, (cm, csd), (sm, ssd) in zip(cs, cov, stk):
        print(f'  cruise {c:.1f} ({len(sweep[c])} runs): coverage {cm:.1f} ± {csd:.1f} %, worst-robot stuck {sm:.1f} ± {ssd:.1f} %')
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.errorbar(cs, [m for m, _ in cov], yerr=[s for _, s in cov], fmt='o-', color=C[0], capsize=3, label='coverage (%)')
    ax.errorbar(cs, [m for m, _ in stk], yerr=[s for _, s in stk], fmt='s--', color=C[1], capsize=3, label='worst-robot stuck (%)')
    ax.set_xlabel('cruise speed (fraction of maximum wheel speed)'); ax.set_ylabel('percentage (%)')
    ax.set_ylim(0, 105); ax.legend(fontsize=8, loc='center right')
    fig.savefig('figs/fig7_4_exercise1_cruise.png'); plt.close(fig)

# ---------- Figures 7.2 and 7.3 ----------
first = sorted(glob.glob('runs/swarm_default_r*.csv'))
if first:
    plot_tracks(first[0], 'figs/fig7_2_swarm.png', 'e-puck swarm, default parameters, 120 s')
if os.path.exists('runs/pioneer_default.csv'):
    plot_tracks('runs/pioneer_default.csv', 'figs/fig7_3_pioneer.png', 'Pioneer 3-DX, default parameters, 180 s')
print('\nFigures written to figs/')
