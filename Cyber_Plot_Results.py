import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe

No_of_Dataset = 1


def plot_Total_comp_time():
    Graph_Time = np.load('Time.npy', allow_pickle=True)
    lable = ['5', '10', '15', '20', '25']
    colors = ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']
    Algorithm = ['DES', 'AES', 'Homomorphic Encryption', 'ECC', 'HECC']
    X = np.arange(len(lable))
    bar_width = 0.15
    x = np.arange(Graph_Time.shape[0])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    # Plot bars for each category
    for i in range(len(Algorithm)):
        bars = plt.bar(x + i * bar_width, Graph_Time[:, i], width=bar_width, label=Algorithm[i], color=colors[i])

    # Customizations
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    # Custom Legend with Dot Markers, positioned at the top
    dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=14) for color
                   in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=14,
               frameon=False, ncol=3)
    plt.tight_layout()
    plt.xticks(X + 0.30, ['5', '10', '15', '20', '25'], fontname="Arial", fontsize=14, fontweight='bold',
               color='#35530a')
    plt.ylabel('Total Computational time (S)', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.xlabel('Block Size', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname='Arial', fontsize=14, fontweight='bold', color='#35530a')
    path = "./Results/Total_Comput_time_enc.png"
    plt.savefig(path)
    plt.show()


def plot_key_sencitivity():
    sencitivity = np.load('key.npy', allow_pickle=True)
    colors = ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']
    Algorithm = ['DES', 'AES', 'Homomorphic Encryption', 'ECC', 'HECC']
    X = np.arange(5)
    bar_width = 0.15
    x = np.arange(sencitivity.shape[0])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    fig.canvas.manager.set_window_title('Algorithm Comparison of Block Size')
    # Plot bars for each category
    for i in range(len(Algorithm)):
        bars = plt.bar(x + i * bar_width, sencitivity[:, i], width=bar_width, label=Algorithm[i], color=colors[i])

    # Customizations
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # Custom Legend with Dot Markers, positioned at the top
    dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=14) for color
                   in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=14,
               frameon=False, ncol=3)
    plt.tight_layout()
    plt.xticks(X + 0.30, ['1', '2', '3', '4', '5'], fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    plt.xlabel('Cases', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('Key Sensitivity Analysis', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    path = "./Results/Key_sen_enc.png"
    plt.savefig(path)
    plt.show()


def plot_CPA_KPA():
    Eval = np.load('CPA_attack.npy', allow_pickle=True)
    colors = ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']
    Algorithm = ['DES', 'AES', 'Homomorphic Encryption', 'ECC', 'HECC']
    X = np.arange(5)
    bar_width = 0.15
    x = np.arange(Eval.shape[0])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    fig.canvas.manager.set_window_title('Algorithm Comparison of Block Size')
    # Plot bars for each category
    for i in range(len(Algorithm)):
        bars = plt.bar(x + i * bar_width, Eval[:, i], width=bar_width, label=Algorithm[i], color=colors[i])

    # Customizations
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    # Custom Legend with Dot Markers, positioned at the top
    dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=14) for color
                   in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=12,
               frameon=False, ncol=3)
    plt.tight_layout()
    labels = ['1', '2', '3', '4', '5']
    plt.xticks(x + 0.30, labels, fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    plt.xlabel('Case', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('CPA Attack', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    path = "./Results/CPA_Algorithm.png"
    plt.savefig(path)
    plt.show()

    colors = ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']
    Algorithm = ['DES', 'AES', 'Homomorphic Encryption', 'ECC', 'HECC']
    Eval = np.load('KPA_attack.npy', allow_pickle=True)
    bar_width = 0.15
    x = np.arange(Eval.shape[0])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    # Plot bars for each category
    for i in range(len(Algorithm)):
        bars = plt.bar(x + i * bar_width, Eval[:, i], width=bar_width, label=Algorithm[i], color=colors[i])

    # Customizations
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(True)
    # Custom Legend with Dot Markers, positioned at the top
    dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=14) for color
                   in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=14,
               frameon=False, ncol=3)
    plt.tight_layout()
    labels = ['1', '2', '3', '4', '5']
    plt.xticks(x + 0.30, labels, fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    plt.xlabel('Case', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('KPA Attack', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    path = "./Results/KPA_Algorithm.png"
    plt.savefig(path)
    plt.show()


def plot_Memory_size():
    Graph = np.load('Memory Size.npy', allow_pickle=True)
    lable = ['5', '10', '15', '20', '25']
    colors = ['yellowgreen', 'gold', 'mediumpurple', 'sandybrown', 'k']
    Algorithm = ['DES', 'AES', 'Homomorphic Encryption', 'ECC', 'HECC']
    X = np.arange(len(lable))
    bar_width = 0.15
    x = np.arange(Graph.shape[0])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    fig.canvas.manager.set_window_title('Algorithm Comparison of Block Size')
    # Plot bars for each category
    for i in range(len(Algorithm)):
        bars = plt.bar(x + i * bar_width, Graph[:, i], width=bar_width, label=Algorithm[i], color=colors[i])

    # Customizations
    # Remove axes outline
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # Custom Legend with Dot Markers, positioned at the top
    dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=14) for color
                   in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10), fontsize=14,
               frameon=False, ncol=3)
    plt.tight_layout()
    plt.xticks(X + 0.30, ['5', '10', '15', '20', '25'], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
    plt.xlabel('Block Size', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.ylabel('Memory Size (kB)', fontname="Arial", fontsize=14, fontweight='bold', color='k')
    plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
    path = "./Results/Mororysize_Algorithm.png"
    plt.savefig(path)
    plt.show()


def Plot_encryption():
    plot_CPA_KPA()
    plot_key_sencitivity()
    plot_Total_comp_time()
    plot_Memory_size()


if __name__ == '__main__':
    Plot_encryption()
