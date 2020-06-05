# -*- coding:utf-8 -*-

from numpy import mean
from numpy import std
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np

# summarize scores


def summarize_results(scores):
    print('scores : ', scores)
    m, s = mean(scores), std(scores)
    print('Accuracy : %.3f%% (+/-%.3f)' % (m, s))


def visualize(trainX, trainy, testX, testy, history, model):

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_acc'], 'b',
             label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--',
             label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    # plt.savefig("cnn_acc_and_loss.png")
    plt.show()

    # Print confusion matrix for training data
    y_pred_train = model.predict(trainX)
    # Take the class with the highest probability from the train predictions
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    #print(classification_report(trainy, max_y_pred_train))

    # confusion matrix
    LABELS = ["ID:2", "ID:3", "ID:4", "ID:6", "ID:9", "ID:12"]
    y_pred_test = model.predict(testX)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(testy, axis=1)

    matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix,
                cmap='Oranges',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.savefig("cnn_result.png")
    plt.show()

    #print(classification_report(max_y_test, max_y_pred_test))


def confusion_matrix(true_labels, predict_labels):

    # confusion matrix
    LABELS = ["ID:2", "ID:3", "ID:4", "ID:6", "ID:9", "ID:12"]
    # y_pred_test = model.predict(testX)
    # Take the class with the highest probability from the test predictions
    max_y_pred_test = true_labels
    max_y_test = predict_labels

    matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix,
                cmap='Oranges',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.savefig(save_fig_name+".png")
    plt.show()
