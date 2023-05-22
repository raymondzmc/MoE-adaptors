import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pdb

data = np.zeros((3, 12))
for task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']:
    task_dir = os.path.join('output', 'glue', 'roberta', f"moe_gnn_gumbel_{task}_prune_gates_attn")

    for file in ['gates_probs.500.pt', 'gates_probs.pt']:
        file_path = os.path.join(task_dir, file)
        if os.path.exists(file_path):
            probs = torch.load(file_path)
            for layer, p in enumerate(probs):
                index = np.argmax(p)
                if index < 3:
                    data[index, layer] += 1

X = np.arange(12)
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.9,0.9])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25, label='sem')
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25, label='syn')
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25, label='pos')

ax.set_xticks(np.arange(12) + 0.25, [str(x) for x in np.arange(1, 13)])
ax.set_xlabel('Layer')
ax.legend(labels=['sem', 'syn', 'pos'])
# ax.set_yticks(np.arange(0, max(data), 2))
# ax.set_xlabel(np.arange(1, 13))
# plt.ylabel('Layers')
plt.axis('on')
plt.savefig('test.png')
    