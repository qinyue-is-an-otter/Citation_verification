from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

# Threshold + tpr/fpr analysis
def analysis_threshold_cos(df, metric):
    y_true = df['Label']
    y_scores = df[metric]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label="Related")
    max_index = np.argmax(tpr-fpr)
    best_threshold = thresholds[max_index]
    plt.xlabel('Thresholds')
    plt.ylabel('True Positive Rate - False Positive Rate')
    plt.title(f'TPR-FPR curve for {metric}\nBest threshold = {best_threshold}')
    plt.plot(thresholds, tpr-fpr)
    # plt.plot(fpr, tpr) # Traditional ROC curve
    plt.show()
    return best_threshold

def get_best_thresholds(df, metric, kth_best):
    y_true = df['Label']
    y_scores = df[metric]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label="Related")
    best_indexs = np.argpartition(tpr-fpr, -kth_best)[-kth_best:] # Get positions of Kth best thresholds
    best_thresholds = thresholds[best_indexs]
    print(f'best thresholds: {best_thresholds}\nbest indexs = {best_indexs}')
    return best_thresholds

def analyze(prediction_file, metric):
    df = pd.read_csv(prediction_file, sep="\t", encoding="utf-8")
    df_unre = df[df["Label"] == "Unrelated"]
    df_rela = df[df["Label"] == "Related"]
    df = pd.concat([df_rela, df_unre])
    analysis_threshold_cos(df, metric)
    get_best_thresholds(df, metric, 5)

# analyze("Datasets/Output/Reference_output.tsv", "Cosine_similarity_sbert") # good_metrics_cleaned Reference_output
# --------------------------------------------------------------------------------------------------------------------------------
# This function plots the number of characters in citation contexts
def plot_sentence_char_len(file, col_name):
    df = pd.read_csv(file, sep="\t", encoding="utf-8")
    x = [len(phrase) for phrase in df[col_name].to_list()]
    df["num_chars"] = x
    plt.ylabel('Number of Citation Contexts')
    plt.xlabel('Number of Characters')
    plt.xlim(1, 600)
    # hue_order = ["Unrelated", "Related"]
    sb_plot = sb.histplot(data=df, x="num_chars", kde=True)
    fig = sb_plot.get_figure()
    fig.savefig(f"CC_window_size.png")
    plt.show()

# plot_sentence_char_len("Datasets/Output/Reference_output.tsv", "Citation_context")