from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
# from tqdm import tqdm
import pandas as pd

def best_threshold_cos(df, metric):
    y_true = df['Label']
    y_scores = df[metric]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label="Related")
    max_index = np.argmax(tpr-fpr)
    best_threshold = thresholds[max_index]
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Roc curve for {metric}\nBest threshold = {best_threshold}')
    plt.plot(fpr, tpr)
    #print(max_index)
    return best_threshold

# Threshold + tpr/fpr analysis
def analysis_threshold_cos(df, metric):
    y_true = df['Label']
    y_scores = df[metric]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label="Related")
    max_index = np.argmax(tpr-fpr)
    print(f"Max performance: {(tpr-fpr)[max_index]}")
    best_threshold = thresholds[max_index]
    plt.xlabel('Thresholds')
    plt.ylabel('True Positive Rate - False Positive Rate')
    plt.title(f'Roc curve for {metric}\nBest threshold = {best_threshold}')
    plt.plot(thresholds, tpr-fpr)
    plt.show()
    #print(max_index)
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
    #print(df_rela.head())
    df = pd.concat([df_rela, df_unre])
    # analysis_threshold_cos(df, "Rouge_score")
    # analysis_threshold_cos(df, "Cosine_similarity_roberta")
    analysis_threshold_cos(df, metric)
    get_best_thresholds(df, metric, 5)

analyze("Datasets/Output/reference_output.tsv", "Cosine_similarity_sbert") # good_metrics_cleaned Reference_output

# This is for fixing the threshold range on augmented set
# 0.14 - 0.21 first run 2000 total
# 0.16 - 0.21 seconde run
# 0.15 - 0.23
# 0.13 - 0.19
# 0.12 - 0.22

# Results on ref set for sbert:
# 0.16  f1 score is: 0.8464730290456431, the precision is: 0.7391304347826086, the recall is: 0.9902912621359223, the accuracy is 0.9304511278195489
# 0.17  f1 score is: 0.8571428571428571, the precision is: 0.7555555555555555, the recall is: 0.9902912621359223, the accuracy is 0.9360902255639098
# 0.18  f1 score is: 0.8820960698689957, the precision is: 0.8015873015873016, the recall is: 0.9805825242718447, the accuracy is 0.9492481203007519
# 0.19  f1 score is: 0.8820960698689957, the precision is: 0.8015873015873016, the recall is: 0.9805825242718447, the accuracy is 0.9492481203007519
# 0.20  f1 score is: 0.9009009009009008, the precision is: 0.8403361344537815, the recall is: 0.970873786407767, the accuracy is 0.9586466165413534
# 0.21  f1 score is: 0.9049773755656108, the precision is: 0.847457627118644, the recall is: 0.970873786407767, the accuracy is 0.9605263157894737
# 0.22  f1 score is: 0.903225806451613, the precision is: 0.8596491228070176, the recall is: 0.9514563106796117, the accuracy is 0.9605263157894737

# Best threshold for sbert on ref set: with threshold 0.218
# For sbert, the f1 score is: 0.903225806451613, the precision is: 0.8596491228070176, the recall is: 0.9514563106796117, the accuracy is 0.9605263157894737

"""
best thresholds: [0.19349852 0.228132   0.19322911 0.21843776 0.34044802]
0.218 is the best
best thresholds: [0.22157696 0.13264132 0.228132   0.34044802 0.17081366]
best indexs = [5 8 4 2 6]

best thresholds: [0.19349852 0.13264132 0.34044802 0.21843776 0.17081366]
best indexs = [ 7 10  4  6  8]
python3 Scripts/metrics.py --input_csv "" --output_file "Datasets/Output/Qwen0.6__aug_output.tsv" --metric_evaluation "metric_eval_only" --methods "sbert" --hist_plot "nope"
"""