import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from matplotlib.font_manager import FontProperties

warnings.filterwarnings('ignore')


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CSA-ADHyNet', 'FA-ADHyNet', 'NGO-ADHyNet', 'SOA-ADHyNet', 'RNU-SOA-ADHyNet']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[0, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(Fitness.shape[2])
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='CSA-ADHyNet')
    plt.plot(length, Conv_Graph[1, :], color='#7D26CD', linewidth=3, marker='*', markerfacecolor='#00FFFF',
             markersize=12, label='FA-ADHyNet')
    plt.plot(length, Conv_Graph[2, :], color='#FF00FF', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='NGO-ADHyNet')
    plt.plot(length, Conv_Graph[3, :], color='#43CD80', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='SOA-ADHyNet')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='RNU-SOA-ADHyNet')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    cls = ['AD-GRU', 'Res-LSTM', 'GRU', 'DHyNet', 'RNU-SOA-ADHyNet']
    Actual = np.load('Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure(facecolor='#F9F9F9')
    fig.canvas.manager.set_window_title('ROC Curve')
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.set_facecolor("#F9F9F9")
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score_1.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Table():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Algorithm = ['Kfold', 'CSA-ADHyNet', 'FA-ADHyNet', 'NGO-ADHyNet', 'SOA-ADHyNet', 'RNU-SOA-ADHyNet']
    Classifier = ['Kfold', 'AD-GRU', 'Res-LSTM', 'GRU', 'DHyNet', 'RNU-SOA-ADHyNet']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 2, 4, 6, 9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Kfold)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Kfold)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plots_Results():
    eval = np.load('Epoch_Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18]
    bar_width = 0.15
    colors = ['#8f2d56', '#b388eb', '#219ebc', '#f77f00', '#8ac926']
    Classifier = ['AD-GRU', 'Res-LSTM', 'GRU', 'DHyNet', 'RNU-SOA-ADHyNet']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Method Comparison of Dataset')
            X = np.arange(3)
            bars1 = plt.bar(X + 0.00, Graph[:, 5], color='#8f2d56', width=0.15, label=Classifier[0])
            ax.bar_label(container=bars1, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 5]],
                         rotation=0, fontweight='bold', padding=5)

            bars2 = plt.bar(X + 0.15, Graph[:, 6], color='#b388eb', width=0.15, label=Classifier[1])
            ax.bar_label(container=bars2, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 6]],
                         rotation=0, fontweight='bold', padding=5)

            bars3 = plt.bar(X + 0.30, Graph[:, 7], color='#219ebc', width=0.15, label=Classifier[2])
            ax.bar_label(container=bars3, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 7]],
                         rotation=0, fontweight='bold', padding=5)
            bars4 = plt.bar(X + 0.45, Graph[:, 8], color='#f77f00', width=0.15, label=Classifier[3])
            ax.bar_label(container=bars4, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 8]],
                         rotation=0, fontweight='bold', padding=5)
            bars5 = plt.bar(X + 0.60, Graph[:, 4], color='#8ac926', width=0.15, label=Classifier[4])
            ax.bar_label(container=bars5, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 4]],
                         rotation=0, fontweight='bold', padding=5)

            # Customizations
            plt.xticks(X + 0.30, ['20', '40', '60'], fontname="Arial", fontsize=14,
                       fontweight='bold', color='k')
            plt.xlabel('No. of Epochs', fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=14, fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)

            # Custom Legend with Dot Markers, positioned at the top
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=12,
                       frameon=False, ncol=len(Classifier))
            plt.tight_layout()

            path = "./Results/%s_mod_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Line_PlotTesults():
    eval = np.load('Epoch_Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18]
    Algorithm = ['CSA-ADHyNet', 'FA-ADHyNet', 'NGO-ADHyNet', 'SOA-ADHyNet', 'RNU-SOA-ADHyNet']
    colors = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
    bar_width = 0.15

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Algorithm Comparison of Dataset')
            X = np.arange(3)
            bars1 = plt.bar(X + 0.00, Graph[:, 0], color='#6a994e', width=0.15, label=Algorithm[0])
            ax.bar_label(container=bars1, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 0]],
                         rotation=0, fontweight='bold', padding=5)

            bars2 = plt.bar(X + 0.15, Graph[:, 1], color='#00a8e8', width=0.15, label=Algorithm[1])
            ax.bar_label(container=bars2, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 1]],
                         rotation=0, fontweight='bold', padding=5)

            bars3 = plt.bar(X + 0.30, Graph[:, 2], color='violet', width=0.15, label=Algorithm[2])
            ax.bar_label(container=bars3, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 2]],
                         rotation=0, fontweight='bold', padding=5)
            bars4 = plt.bar(X + 0.45, Graph[:, 3], color='crimson', width=0.15, label=Algorithm[3])
            ax.bar_label(container=bars4, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 3]],
                         rotation=0, fontweight='bold', padding=5)
            bars5 = plt.bar(X + 0.60, Graph[:, 4], color='k', width=0.15, label=Algorithm[4])
            ax.bar_label(container=bars5, size=10, label_type='edge', labels=[f'{x:.1f}' for x in Graph[:, 4]],
                         rotation=0, fontweight='bold', padding=5)

            # Customizations
            plt.xticks(X + 0.30, ['20', '40', '60'], fontsize=14, fontname="Arial",
                       fontweight='bold', color='k')
            plt.xlabel('No. of Epochs', fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.ylabel(Terms[Graph_Terms[j]], fontsize=14, fontname="Arial", fontweight='bold', color='k')
            plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='#35530a')
            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(True)

            # Custom Legend with Dot Markers, positioned at the top
            colors = ['#6a994e', '#00a8e8', 'violet', 'crimson', 'k']
            dot_markers = [plt.Line2D([2], [2], marker='o', color='w', markerfacecolor=color, markersize=12) for color
                           in colors]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.12), fontsize=12,
                       frameon=False, ncol=3)
            plt.tight_layout()
            path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Confusion():
    Actual = np.load('Actual_1.npy', allow_pickle=True)
    Predict = np.load('Predict_1.npy', allow_pickle=True)
    class_2 = ['Apply', 'Cherry', 'Citrus', 'Corn', 'Gauva', 'Grape', 'Mango', 'Peach', 'Pepper', 'Potato', 'Sapota',
               'Straw\nberry', 'Tomato']
    fig, ax = plt.subplots(figsize=(10, 8))
    confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_2)
    cm_display.plot(ax=ax)
    path = "./Results/Confusion.png"
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # Rotate the labels
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    # plotConvResults()
    # Line_PlotTesults()
    Plots_Results()
    # Plot_ROC_Curve()
    # Table()
