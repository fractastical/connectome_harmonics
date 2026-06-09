"""
Connectome Harmonics Toy Simulator
----------------------------------
A small educational simulation of graph Laplacian harmonics on a brain-like
network graph.

Run:
    python connectome_harmonics_sim.py
    python connectome_harmonics_sim.py --data connectome_harmonics_data_hcp.json

Requires: numpy, scipy, matplotlib
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DEFAULT_DATA = Path(__file__).with_name('connectome_harmonics_data.json')
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='JSON data file')
parser.add_argument('--mode', type=int, default=7, help='Harmonic mode (1-indexed)')
args = parser.parse_args()

DATA_PATH = args.data
if not DATA_PATH.exists():
    DATA_PATH = Path('/mnt/data/connectome_harmonics_data.json')
DATA = json.loads(DATA_PATH.read_text())
nodes = np.array(DATA['nodes'])
edges = DATA['edges']
modes = np.array(DATA['modes'])
eigvals = np.array(DATA['eigvals'])

mode = args.mode  # 1-indexed, like the slide labels
speed = 1.0
amp = 1.0
mode_idx = mode - 1

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor('#0b0e14')
fig.patch.set_facecolor('#0b0e14')
ax.set_aspect('equal')
ax.axis('off')

for i, j, w in edges:
    ax.plot([nodes[i,0], nodes[j,0]], [nodes[i,1], nodes[j,1]], lw=0.2 + 0.8*w, alpha=0.12, color='white', zorder=1)
scatter = ax.scatter(nodes[:,0], nodes[:,1], c=np.zeros(len(nodes)), s=28, cmap='coolwarm', vmin=-1, vmax=1, edgecolors='none', zorder=2)
title = ax.set_title('', color='white')

def update(frame):
    t = frame / 30
    field = amp * modes[mode_idx] * np.cos(speed * t * (0.8 + 2.2*np.sqrt(eigvals[mode_idx])))
    scatter.set_array(field)
    title.set_text(f'Connectome harmonic mode {mode}   λ={eigvals[mode_idx]:.4f}')
    return scatter, title

ani = FuncAnimation(fig, update, frames=180, interval=33, blit=False)
plt.show()
