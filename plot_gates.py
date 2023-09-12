import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.functional import F
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from scipy.interpolate import make_interp_spline
import os
import pdb
import seaborn as sns
import numpy as np





def jenson_shannon_divergence(dist1, dist2):
    jensenshannon
    F.kl_div(dist1, dist2)
    total_m = 0.5 * (dist1 + dist2)
    jsd = 0.5 * F.kl_div(F.log_softmax(dist1, dim=1), total_m, reduction='none') \
        + F.kl_div(F.log_softmax(dist2, dim=1), total_m, reduction='batch')
    return jsd


# for task_name in ['cola', 'mrpc', 'qnli', 'rte', 'sst2', 'stsb']:
for task_name in ['qnli', 'mnli']:
    data_dir = os.path.join('output', 'glue_roberta', task_name)
    prob_files = [f for f in os.listdir(data_dir) if (f.startswith('gates_probs') and f.split('.')[-1] == 'pt')]
    prob_files = sorted(prob_files, key=lambda x: int(x.split('.')[1]))

    prev_gates = None
    jsd = []
    smr = []
    for f in prob_files:
        step = f.split('.')[1]
        plot_name = f.split('.pt')[0] + '.png'
        gates = torch.load(os.path.join(data_dir, f))
        try:
            gates = torch.stack(gates).detach().cpu()
        except:
            break
        softmax_gates = torch.softmax(gates / 0.01, dim=1).tolist()
        sns.heatmap(softmax_gates, cmap='viridis', annot=True)
        plt.title(f'Gate Values for {task_name} step {step}')
        plt.xlabel('Semantic (DM), Syntax, Positional (Chain), MLP')
        plt.ylabel('Layers')

        # Save the heatmap
        save_path = os.path.join(data_dir, plot_name)
        plt.savefig(save_path)
        plt.clf()
        plt.close()
        
        
        if prev_gates != None:
            jsd.append(jensenshannon(softmax_gates, prev_gates, axis=1))
            _smr = []
            for i in range(len(softmax_gates)):
                _smr.append(spearmanr(softmax_gates[i], prev_gates[i]).statistic)
            smr.append(_smr)
        prev_gates = softmax_gates

    avg_jsd = [x.mean() for x in jsd]
    steps = list(range(10, len(avg_jsd) * 10 + 1, 10))
    fig, ax = plt.subplots()
    ax.plot(steps, avg_jsd)
    ax.set(xlabel='Steps', ylabel='JS-Divergence', title='Average \u0394JS-Divergence for Gate Values')
    fig.savefig(os.path.join(data_dir, 'avg_jsd.png'))
    plt.clf()
    plt.close()

    # avg_smr = [np.mean(x) for x in smr]
    # fig, ax = plt.subplots()
    # ax.plot(avg_smr)
    # ax.set(xlabel='steps', ylabel='smr',
    #     title='smr vs time')
    # fig.savefig(os.path.join(data_dir, 'avg_smr.png'))
    # plt.clf()
    # plt.close()
    limit = [np.array(jsd).min(), np.array(jsd).max() * 1.03]
    n_layers = len(jsd[0])
    fig, axs = plt.subplots(n_layers)
    plt.subplots_adjust(hspace=0)

    fig.suptitle('JS-Divergence of Gate Values Across Layers')
    fig.set_figheight(14)
    fig.set_figwidth(6)

    for layer in range(n_layers):
        jsd_layer = [x[layer] for x in jsd]
        steps = list(range(10, len(avg_jsd) * 10 + 1, 10))

        X_Y_Spline = make_interp_spline(steps, jsd_layer)
        X_ = np.linspace(min(steps), max(steps), 5000)
        Y_ = X_Y_Spline(X_)

        index = n_layers - layer - 1
        axs[index].set_ylim(limit)
        axs[index].plot(X_, Y_)
        axs[index].set_yticks([])
        axs[index].set_ylabel(f'Layer {layer + 1}', rotation=90)
        if layer > 0:
            ax.set_xticks([])
        else:
            axs[index].set_yticks([])
            axs[index].set(xlabel='Steps')

    fig.savefig(os.path.join(data_dir, f'gate_jsd_layer.png'))
    plt.clf()
    plt.close()

        # smr_layer = [x[layer] for x in smr]
        # fig, ax = plt.subplots()
        # ax.plot(smr_layer)
        # ax.set(xlabel='steps', ylabel='smr', title='smr vs time')
        # fig.savefig(os.path.join(data_dir, f'avg_smr_layer{layer + 1}.png'))
        # plt.clf()
        # plt.close()

# data = np.zeros((3, 12))
# for task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']:
#     task_dir = os.path.join('output', 'glue', 'roberta', f"moe_gnn_gumbel_{task}_prune_gates_attn")

#     for file in ['gates_probs.500.pt', 'gates_probs.pt']:
#         file_path = os.path.join(task_dir, file)
#         if os.path.exists(file_path):
#             probs = torch.load(file_path)
#             for layer, p in enumerate(probs):
#                 index = np.argmax(p)
#                 if index < 3:
#                     data[index, layer] += 1

# X = np.arange(12)
# fig = plt.figure()
# ax = fig.add_axes([0.1,0.1,0.9,0.9])
# ax.bar(X + 0.00, data[0], color = 'b', width = 0.25, label='sem')
# ax.bar(X + 0.25, data[1], color = 'g', width = 0.25, label='syn')
# ax.bar(X + 0.50, data[2], color = 'r', width = 0.25, label='pos')

# ax.set_xticks(np.arange(12) + 0.25, [str(x) for x in np.arange(1, 13)])
# ax.set_xlabel('Layer')
# ax.legend(labels=['sem', 'syn', 'pos'])
# # ax.set_yticks(np.arange(0, max(data), 2))
# # ax.set_xlabel(np.arange(1, 13))
# # plt.ylabel('Layers')
# plt.axis('on')
# plt.savefig('test.png')
    