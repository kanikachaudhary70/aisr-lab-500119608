# CSAI4014 Algorithms for Intelligent Systems and Robotics: Laboratory Work

- **Student:** Kanika Chaudhary, SAP ID 500119608
- **Batch:** B (B.Tech CSE AI & ML, B2), Semester VII, Autumn 2026–27
- **Roll No.:** R2142230730
- **Course teacher:** Dr. Resham Raj Shivwanshi, School of Computer Science, UPES Dehradun

Code, results and figures for Experiments 1 to 8 of the laboratory report.

| Folder | Experiment | Contents |
|---|---|---|
| `exp01-signals/` | 1. Time and frequency domain, the pendulum | notebook, `results.json`, figures |
| `exp02-queens/` | 2. Eight queens (backtracking vs generate and test) | notebook, `results.json`, figures |
| `exp03-search/` | 3. BFS and DFS on a grid maze | notebook, `results.json`, figures |
| `exp04-hillclimb/` | 4. Hill climbing and random restarts | notebook, `results.json`, figures |
| `exp05-rl/` | 5. Tabular Q-learning | notebook, `results.json`, figures |
| `exp06-neural/` | 6. NumPy neural network for XOR | notebook, `results.json`, figures |
| `exp07-webots/` | 7. Webots swarm and lidar navigation | run and analysis scripts, figures from my partner's runs, `NOTES.md` |
| `exp08-xembody/` | 8. Cross-embodiment transfer (PyBullet) | result CSVs and plots, exercises E2 and E4, `NOTES.md` |

## Reproducing Experiments 1 to 6
Python 3.11.17, NumPy 1.26.4, SciPy 1.16.1, Matplotlib 3.10.5. Each `expNN.py` is the notebook source
(cells marked with `# %%`); the `.ipynb` was executed top to bottom on a fresh kernel. Random seeds are
fixed in every notebook, so re-running gives the same numbers.

## Sources
- Course Laboratory Manual (Experiments 1 to 6); `MANUAL.md` of AISR-Cross-Embodiment-PyBullet (Experiment 8);
  `GUIDE.md` of [webots-robotics-sim](https://github.com/kanikachaudhary70/aisr-lab-500119608/tree/main/exp07-webots) (Experiment 7).
- Textbooks and papers are listed in the References of the report.

## AI tools
Claude (Anthropic) was used as a coding and writing assistant.
