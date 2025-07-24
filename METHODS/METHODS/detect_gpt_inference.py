import os
import pandas as pd
from sklearn.metrics import *
from tqdm.auto import tqdm

tqdm.pandas(desc = 'Inferencing..')
def get_performace(test_dataset, threshold, prediction_save_path, prediction_exist = False):

  if not prediction_exist:
    llr_test = test_dataset['llr'].values
    y_pred = (llr_test > threshold).astype(int)
    test_dataset['predict_label'] = y_pred

    # save the predictions
    # predictions_path = f"./prediction/model_{}_data_{}"
    test_dataset.to_csv(f'{prediction_save_path}', index=False)
    print(f'Prediction Saved to {prediction_save_path}')

  test_dataset = pd.read_csv(f'{prediction_save_path}')
  y_test, y_pred = test_dataset['label'].values, test_dataset['predict_label'].values

  print('Results: Human Is 1, AI is 0')
  print('Accuracy:', round(accuracy_score(y_test, y_pred),2))
  print('MCC Score:', round(matthews_corrcoef(y_test, y_pred),2))
  print('F1 Score:', round(f1_score(y_test, y_pred, average = 'macro'),2))
  print('Human Precision:', round(precision_score(y_test, y_pred, pos_label=1),2))
  print('AI Precision:', round(precision_score(y_test, y_pred, pos_label=0),2))
  print('Human Recall:', round(recall_score(y_test, y_pred, pos_label=1),2))
  print('AI Recall:', round(recall_score(y_test, y_pred, pos_label=0),2))


threshold = -1
# prediction_save_path = f'../HindiSumm/data_with_detectgpt_scores/train_predictions.csv'
prediction_save_path = f'../xquad/data_with_detectgpt_scores/Data_gemini_detectgpt.csv'
train_dataset = pd.read_csv(prediction_save_path)
prediction_exist = False
get_performace(train_dataset, threshold, prediction_save_path, prediction_exist = False)

from sklearn.metrics import classification_report   
true_labels = train_dataset['label'].values
predicted_labels = train_dataset['predict_label'].values
print(classification_report(true_labels, predicted_labels))

# plot a graph to see the distribution of the llr
import seaborn as sns
import matplotlib.pyplot as plt
# seperately for each class
sns.histplot(train_dataset[train_dataset['label'] == 1]['llr'], color = 'blue', label = 'Human')
sns.histplot(train_dataset[train_dataset['label'] == 0]['llr'], color = 'red', label = 'AI')
plt.legend()
plt.show()
plt.savefig('../HindiSumm/data_with_detectgpt_scores/llr_distribution.png')
