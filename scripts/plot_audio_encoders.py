import matplotlib.pyplot as plt
import numpy as np
import os

datasets = ['VSD', 'CREMA-D', 'ESC-50']
architectures = ['YAMNet', 'VGGish', 'PANNs', 'Proposed Hybrid']

metrics = {
    'Accuracy': {
        'YAMNet': [0.9770, 0.8343, 0.9770],
        'VGGish': [0.9695, 0.8237, 0.8965],
        'PANNs': [0.9770, 0.8249, 0.9745],
        'Proposed Hybrid': [0.9773, 0.8691, 0.9125]
    },
    'Precision': {
        'YAMNet': [0.9844, 0.5108, 0.8272],
        'VGGish': [0.9844, 0.4909, 0.4285],
        'PANNs': [0.9846, 0.4917, 0.8441],
        'Proposed Hybrid': [0.9848, 0.5782, 0.4781]
    },
    'Recall': {
        'YAMNet': [0.9922, 0.7184, 0.9000],
        'VGGish': [0.9844, 0.8041, 0.8750],
        'PANNs': [0.9922, 0.6924, 0.8438],
        'Proposed Hybrid': [0.9922, 0.8726, 0.8313]
    },
    'F1-Score': {
        'YAMNet': [0.9882, 0.5968, 0.8618],
        'VGGish': [0.9843, 0.6094, 0.5743],
        'PANNs': [0.9882, 0.5747, 0.8407],
        'Proposed Hybrid': [0.9883, 0.6952, 0.6050]
    }
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

x = np.arange(len(datasets))
width = 0.2

for idx, (metric_name, metric_data) in enumerate(metrics.items()):
    ax = axes[idx]
    for i, arch in enumerate(architectures):
        offset = (i - 1.5) * width
        ax.bar(x + offset, metric_data[arch], width, label=arch)
    ax.set_title(f'{metric_name} across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim([0.0, 1.1])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    if idx == 2: # Add legend on bottom left
        ax.legend(loc='lower center', bbox_to_anchor=(1.0, -0.2), ncol=4)

plt.tight_layout()
fig.subplots_adjust(bottom=0.15) # Leave space for the legend
os.makedirs('references/Springer_Nature_LaTeX_Template/figures', exist_ok=True)
plt.savefig('references/Springer_Nature_LaTeX_Template/figures/audio_encoder_metrics_grid.png', dpi=300)
print('Saved to references/Springer_Nature_LaTeX_Template/figures/audio_encoder_metrics_grid.png')
