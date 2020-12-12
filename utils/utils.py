import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

def nan_info(df, hide_no_nans=True):
  nans_count = df.isnull().sum()
  nans_percent = 100*nans_count/df.shape[0]
  nan_df = pd.DataFrame({'count': nans_count, 'percent': nans_percent}).sort_values('percent', axis=0, ascending=False)
  if hide_no_nans:
      return nan_df[nan_df['count']!=0]
  return nan_df

def show_corr_mx(df, size=(20,12)):
  sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2) 
  fig=plt.gcf()
  fig.set_size_inches(*size)

def show_scores(y, y_pred):
  conf_mx = confusion_matrix(y, y_pred)
  print('scores\n')
  print('precision', precision_score(y, y_pred))
  print('recall   ', recall_score(y, y_pred))
  print('f1       ', f1_score(y, y_pred))
  print('accuracy ', np.sum(y == y_pred)/y.shape[0])

  plt.figure()

  sns.heatmap(conf_mx, annot=True, fmt='3.0f')

def plot_precision_recall_vs_threshold_curve(precisions, recalls, thresholds):
  fig, [ax1, ax2] = plt.subplots(2, 1, sharex="col", sharey="row", figsize=(8,10))
  ax2.plot(thresholds, precisions[:-1], 'b--', label='precision')
  ax2.plot(thresholds, recalls[:-1], 'g-', label='recall')
  ax2.set_xlabel('Threshold')
  ax2.set_title("Precision and recall versus the decision threshold")
  ax2.legend(loc='best')
  ax2.grid()
  
  
  ax1.plot(precisions, recalls)
  ax1.set_title("Precision versus recall")
  ax1.set_ylabel('Precision')
  ax1.set_xlabel('Recall')
  ax1.grid()

def plot_learning_curves(model, X, y, error, step_size=30):
  '''
  error / score function should be appropriate to the model and the problem
  - regression - RMSE
  - classification - sklearn.metrics.accuracy_score or sklearn.metrics.f1_score

  If score function,  an utility function (the higher the better) is used the graph is mirrored horizontally.
  '''
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  train_index, test_index = list(split.split(X, y))[0]
  X_train, X_val, y_train, y_val = X[train_index], X[test_index], y[train_index], y[test_index]
  train_errors, val_errors = [], []
  for m in range(step_size, len(X_train), step_size):
      model.fit(X_train[:m], y_train[:m])
      y_train_predict = model.predict(X_train[:m])
      y_val_predict = model.predict(X_val)
      
      train_errors.append(error(y_train[:m], y_train_predict))
      val_errors.append(error(y_val, y_val_predict))
  plt.figure()    
  plt.plot(range(step_size, len(X_train), step_size), train_errors, "r-+", linewidth=2, label="train")
  plt.plot(range(step_size, len(X_train), step_size), val_errors, "b-", linewidth=3, label="val")
  plt.legend()
  plt.xlabel('Training Set Size')
  plt.ylabel('Error')
  plt.title('Learning Curves')
  plt.grid()

def show_feature_importances(clf, features):
  plt.figure()
  imp_df = pd.DataFrame({'Features': features, 'Importance': clf.feature_importances_}, index=features).sort_values(by='Importance')
  imp_df.plot(kind='barh')