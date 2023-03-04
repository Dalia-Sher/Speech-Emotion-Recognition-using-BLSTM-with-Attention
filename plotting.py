
# Import relevant libraries
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.utils import plot_model
from time import gmtime, strftime


# Function to create plotting
def plotting_function(args, model, model_dir, history, y_test_arr, preds, dic, model_type, dataset):

    # Getting the time for the plots
    d = strftime("%d", gmtime())
    m = strftime("%m", gmtime())
    H = strftime("%H", gmtime())
    M = strftime("%M", gmtime())

    model_fname = '%s_model_%s_%s_%s_%s.png' % (model_type, d, m, H, M)
    plot_model(model, to_file=model_dir + model_fname, show_shapes=True)

    if args.model_load == 'NO':
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'%s Model - Loss vs Epochs' % (model_type))
        # plt.title(f'%s Model - Loss vs Epochs %s/%s %s:%s' % (model_type, d, m, H, M))
        plt.legend()
        plt.savefig(f'/home/dsi/shermad1/PycharmProjects/SER_git/results/%s_Loss_5conv.png' % model_type)
        plt.show()

        plt.figure()
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='validation')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title(f'%s Model - Accuracy vs Epochs' % (model_type))
        # plt.title(f'%s Model - Accuracy vs Epochs %s/%s %s:%s' % (model_type, d, m, H, M))
        plt.legend()
        plt.savefig(f'/home/dsi/shermad1/PycharmProjects/SER_git/results/%s_Accuracy_160123.png' % model_type)
        plt.show()

    print('Classification_Report')
    print(classification_report(y_test_arr, preds, target_names=list(dic.values())))

    cm = confusion_matrix(y_true=y_test_arr, y_pred=preds)
    cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()

    sns.heatmap(cmn*100, cmap='Blues', annot=True, fmt='.2f', xticklabels=dic.values(), yticklabels=dic.values(), annot_kws={'size': 18})
    ax.xaxis.set_label_position("bottom")
    plt.setp(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
    plt.tight_layout()
    plt.title(f'%s Model - Confusion Matrix' % (model_type))
    # plt.title(f'%s Model - Confusion Matrix %s/%s %s:%s' % (model_type, d, m, H, M))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'/home/dsi/shermad1/PycharmProjects/SER_git/results/%s_Confusion_Matrix_%s_160123.jpg' % (model_type, dataset))
    plt.show()

def plotting_function_kmeans(args, model, model_dir, history, y_test_arr, preds, dic, model_type, dataset):

    # Getting the time for the plots
    print('Classification_Report')
    print(classification_report(y_test_arr, preds, target_names=list(dic.values())))

    cm = confusion_matrix(y_true=y_test_arr, y_pred=preds)
    cmn = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()

    sns.heatmap(cmn*100, cmap='Blues', annot=True, fmt='.2f', xticklabels=dic.values(), yticklabels=dic.values(), annot_kws={'size': 18})
    ax.xaxis.set_label_position("bottom")
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=18)
    plt.setp(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
    plt.tight_layout()
    plt.title(f'%s Model - Confusion Matrix' % (model_type))
    # plt.title(f'%s Model - Confusion Matrix %s/%s %s:%s' % (model_type, d, m, H, M))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'/home/dsi/shermad1/PycharmProjects/SER_git/results/%s_Confusion_Matrix_%s_kmeans.png' % (model_type, dataset))
    plt.show()

def compare_results(y_test_arr, preds, d):
    with open('data_new/IEMOCAP_df_new.pickle', 'rb') as f:
        IEMOCAP_df = pickle.load(f)

    IEMOCAP_df_test = IEMOCAP_df[IEMOCAP_df['split'] == 'Test']
    IEMOCAP_df_test = IEMOCAP_df_test.reset_index(drop=True)

    test_true_label = (pd.Series(y_test_arr)).map(d)
    test_true_label = list(test_true_label)

    test_pred_label = (pd.Series(preds)).map(d)
    test_pred_label = list(test_pred_label)

    IEMOCAP_df_new = pd.concat([IEMOCAP_df_test, pd.DataFrame(test_true_label, columns=['actual']),
                                           pd.DataFrame(test_pred_label, columns=['predicted'])], axis=1)

    print(IEMOCAP_df_new[['name', 'actual', 'predicted']][0:21])
