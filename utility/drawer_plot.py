from sklearn.metrics import roc_curve, auc
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
import os
from csv import DictWriter
import tensorflow as tf


def plot_roc_curve(fpr, tpr, name_model, result_folder=""):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(result_folder + name_model + '_roc_curve.png')
    plt.show()
    plt.clf()


''' calculate the auc value for labels and scores'''


def roc(labels, scores, name_model, result_folder="", draw_plot=True):
    """Compute ROC curve and ROC area for each class"""
    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    # print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshod that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    if draw_plot:
        plot_roc_curve(fpr, tpr, name_model, result_folder)

    return roc_auc, optimal_threshold


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          result_folder=""
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(result_folder + title + '_cm.png')
    plt.show()
    plt.clf()


def plot_epoch_result(iters, loss, name, model_name, colour, result_folder=""):
    plt.plot(iters, loss, colour, label=name)
    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Iters')
    plt.ylabel(name)
    plt.legend()
    plt.savefig(result_folder + model_name + '_' + name + '_iters_result.png')
    plt.show()
    plt.clf()


def plot_anomaly_score(score_ano, labels, name, model_name, result_folder=""):
    df = pd.DataFrame(
        {'predicts': score_ano,
         'label': labels
         })

    df_normal = df[df.label == 0]
    sns.distplot(df_normal['predicts'], kde=False, label='normal')

    df_defect = df[df.label == 1]
    sns.distplot(df_defect['predicts'], kde=False, label='defect')

    #     plt.plot(epochs, disc_loss, 'b', label='Discriminator loss')
    plt.title(name)
    plt.xlabel('Anomaly Scores')
    plt.ylabel('Number of samples')
    plt.legend(prop={'size': 12})
    plt.savefig(result_folder + model_name + '_' + name + '_anomaly_scores_dist.png')
    plt.show()
    plt.clf()


def write_result(array_lines, name, result_folder=""):
    with open(f'{result_folder}{name}_result.txt', 'w+') as f:
        f.write('\n'.join(array_lines))
    f.close()


def write_result_dict(dict_lines, name, result_folder=""):
    with open(f'{result_folder}{name}_result_dict.txt', 'w+') as f:
        for key, value in dict_lines.items():
            f.write('%s : %s\n' % (key, value))
    f.close()


def write_settings(dict_input, name, result_folder=""):
    with open(f"{result_folder}{name}_settings.txt", 'w+') as f:
        for key, value in dict_input.items():
            f.write('%s : %s\n' % (key, value))
    f.close()


def write_main_result(dict_input, fields_name, name, result_folder="", name_file="main_result.csv"):
    result = os.path.dirname(result_folder)
    res_folder, tail = os.path.split(result)
    # print(res_folder, tail)

    name_file_folder = f'{res_folder}/{name_file}'

    file_exists = os.path.exists(name_file_folder)

    with open(name_file_folder, 'a+') as f_object:
        # Pass the file object and a list
        # of column names to DictWriter()
        # You will get a object of DictWriter
        dictwriter_object = DictWriter(f_object, fieldnames=fields_name)

        if file_exists == False:
            dictwriter_object.writeheader()

        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict_input)

    # Close the file object
    f_object.close()


def write_data_analytics(name, dict_list, result_folder=""):
    filename = f"{result_folder}{name}.csv"

    with open(filename, "w+") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict_list.keys())
        writer.writerows(zip(*dict_list.values()))
    outfile.close()

    print('saving is complete')

    # write file that predict falsely
    df = pd.read_csv(filename)
    df = df[df['preds'] != df['true']]
    df.to_csv(f"{result_folder}{name}_false_predictly.csv", encoding='utf-8', index=False)

    return filename


def show_plot_pie(values, labels, title, name, result_folder=""):
    # define Seaborn color palette to use
    colors = sns.color_palette('Set3')

    plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=False, colors=colors)
    plt.title(title)
    plt.axis('equal')
    plt.savefig(result_folder + name + '_plot_pie.png')
    plt.show()
    plt.clf()


def show_hist_plot(df, x, hue, name, result_folder=""):
    ax = sns.histplot(df, x=x, hue=hue, element="step", palette="Set3", legend=True)
    # ax.legend()
    ax.figure.savefig(result_folder + name + '_hist_plot.png')
    # plt.clf()


def show_bar_plot(df, x, y, hue, name, result_folder=""):
    fig = sns.barplot(data=df, x=x, y=y, hue=hue, palette="Set3")
    # fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.figure.savefig(result_folder + name + '_bar_plot.png')
    plt.clf()


def get_tnr_tpr_custom(labels, scores, tnr_min=0.9):
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # get the best threshold with fpr <=0.1
    df = pd.DataFrame(columns=['threshold', 'TNR', 'TPR'])

    for th in thresholds:

        modifed_scores_ano_final = (scores > th).astype(float)

        modifed_test_cm = tf.math.confusion_matrix(
            labels=labels,
            predictions=modifed_scores_ano_final
        ).numpy()

        M_T_TP = modifed_test_cm[1][1]
        M_T_FP = modifed_test_cm[0][1]
        M_T_FN = modifed_test_cm[1][0]
        M_T_TN = modifed_test_cm[0][0]

        TNR = (M_T_TN / (M_T_FP + M_T_TN))
        TPR = (M_T_TP / (M_T_TP + M_T_FN))

        # print(data)
        if TNR >= tnr_min:
            data = {
                "threshold": float(th),
                "TNR": float(TNR),
                "TPR": float(TPR),
            }
            df = pd.concat([df, pd.DataFrame.from_records([data])])

    test = df.sort_values('TPR', ascending=False)
    print(test.head(3))
    best_treshold = test['threshold'].iloc[0]
    best_TNR = test['TNR'].iloc[0]
    best_TPR = test['TPR'].iloc[0]

    return best_treshold, best_TNR, best_TPR
