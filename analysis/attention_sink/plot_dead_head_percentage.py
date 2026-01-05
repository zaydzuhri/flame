"""Plot dead head percentage over training for 1.8B and 340M models."""

import matplotlib.pyplot as plt
import matplotlib
import re
from edd_utils import register_edd_style

register_edd_style()

# ACL 2-column template: column width is ~3.25 inches
# Use appropriate font sizes for readability
matplotlib.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

# Parse the data from the result file
data_1_8b = {
    'steps': [],
    'dead_pct': []
}

data_340m = {
    'steps': [],
    'dead_pct': []
}

with open('/home/coder/Python_project/flame/resources/head_dead_result.txt', 'r') as f:
    content = f.read()

# Parse 1.8B data
lines = content.split('\n')
current_model = None

for line in lines:
    if '# 1.8B' in line:
        current_model = '1.8B'
    elif '# 340M' in line:
        current_model = '340M'
    elif 'Step ' in line and 'dead=' in line:
        # Extract step number and dead percentage
        match = re.search(r'Step (\d+): dead=\d+ \(([\d.]+)%\)', line)
        if match:
            step = int(match.group(1))
            pct = float(match.group(2))
            if current_model == '1.8B':
                data_1_8b['steps'].append(step)
                data_1_8b['dead_pct'].append(pct)
            elif current_model == '340M':
                data_340m['steps'].append(step)
                data_340m['dead_pct'].append(pct)

# Create figure with ACL-appropriate size
fig, ax = plt.subplots(figsize=(3.25, 2.4))

# Plot both lines
ax.plot(data_1_8b['steps'], data_1_8b['dead_pct'],
        marker='o', markersize=3, linewidth=1.2,
        color='#1f77b4', label='1.8B')
ax.plot(data_340m['steps'], data_340m['dead_pct'],
        marker='s', markersize=3, linewidth=1.2,
        color='#ff7f0e', label='340M')

# Format x-axis with k notation
ax.set_xticks([0, 50000, 100000, 150000, 200000])
ax.set_xticklabels(['0', '50k', '100k', '150k', '200k'])

ax.set_xlabel('Training step')
ax.set_ylabel(r'Dead heads (\%)')
ax.set_title('Dead attention heads over training')

ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Set axis limits with some padding
ax.set_xlim(0, 210000)
ax.set_ylim(0, 50)

plt.tight_layout()

# Save figure
output_path = '/home/coder/Python_project/flame/analysis/attention_sink/outputs/figures/dead_head_percentage.pdf'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.1)

print(f"Saved to {output_path}")
print(f"1.8B data points: {len(data_1_8b['steps'])}")
print(f"340M data points: {len(data_340m['steps'])}")
