# Fabric notebook source


# MARKDOWN ********************

# Copyright (c) Microsoft Corporation. All rights reserved. 
# 
# Licensed under the MIT License.
# 
# # AutoML with FLAML Library
# 
# 
# ## 1. Introduction
# 
# FLAML is a Python library (https://github.com/microsoft/FLAML) designed to automatically produce accurate machine learning models 
# with low computational cost. It is fast and economical. The simple and lightweight design makes it easy to use and extend, such as adding new learners. FLAML can 
# - serve as an economical AutoML engine,
# - be used as a fast hyperparameter tuning tool, or 
# - be embedded in self-tuning software that requires low latency & resource in repetitive
#    tuning tasks.
# 
# In this notebook, we use a binary classification task to showcase the task-oriented AutoML in FLAML library.
# 
# FLAML requires `Python>=3.7`. To run this notebook example, please install flaml with the `notebook` option:
# ```bash
# pip install flaml[notebook]
# ```

# CELL ********************

%pip install flaml[notebook]==1.1.1

# MARKDOWN ********************

# ## 2. Classification Example
# ### Load data and preprocess
# 
# Download [Airlines dataset](https://www.openml.org/d/1169) from OpenML. The task is to predict whether a given flight will be delayed, given the information of the scheduled departure.

# CELL ********************

from flaml.data import load_openml_dataset

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir='./')

# CELL ********************

X_train.head()

# MARKDOWN ********************

# ### Run FLAML
# In the FLAML automl run configuration, users can specify the task type, time budget, error metric, learner list, whether to subsample, resampling strategy type, and so on. All these arguments have default values which will be used if users do not provide them. For example, the default classifiers are `['lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree', 'lrl1']`. 

# CELL ********************

''' import AutoML class from flaml package '''
from flaml import AutoML

automl = AutoML()

# CELL ********************

settings = {
    "time_budget": 600,  # total running time in seconds
    "metric": 'accuracy', 
                        # check the documentation for options of metrics (https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#optimization-metric)
    "task": 'classification',  # task type
    "log_file_name": 'airlines_experiment.log',  # flaml log file
    "seed": 7654321,    # random seed
}


# CELL ********************

'''The main flaml automl API'''
automl.fit(X_train=X_train, y_train=y_train, **settings)

# MARKDOWN ********************

# ### Best model and metric

# CELL ********************

'''retrieve best config and best learner'''
print('Best ML leaner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

# CELL ********************

automl.model.estimator

# CELL ********************

'''pickle and save the automl object'''
import pickle

with open('automl.pkl', 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
'''load pickled automl object'''
with open('automl.pkl', 'rb') as f:
    automl = pickle.load(f)

# CELL ********************

'''compute predictions of testing dataset''' 
y_pred = automl.predict(X_test)
print('Predicted labels', y_pred)
print('True labels', y_test)
y_pred_proba = automl.predict_proba(X_test)[:,1]

# CELL ********************

''' compute different metric values on testing dataset'''
from flaml.ml import sklearn_metric_loss_score

print('accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred, y_test))
print('roc_auc', '=', 1 - sklearn_metric_loss_score('roc_auc', y_pred_proba, y_test))
print('log_loss', '=', sklearn_metric_loss_score('log_loss', y_pred_proba, y_test))

# MARKDOWN ********************

# See Section 4 for an accuracy comparison with default LightGBM and XGBoost.
# 
# ### Log history

# CELL ********************

from flaml.data import get_output_from_log

time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
    get_output_from_log(filename=settings['log_file_name'], time_budget=240)
for config in config_history:
    print(config)

# CELL ********************

import matplotlib.pyplot as plt
import numpy as np

plt.title('Learning Curve')
plt.xlabel('Wall Clock Time (s)')
plt.ylabel('Validation Accuracy')
plt.scatter(time_history, 1 - np.array(valid_loss_history))
plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
plt.show()

# MARKDOWN ********************

# ## 3. Comparison with alternatives


# MARKDOWN ********************

# ### Default LightGBM

# CELL ********************

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()

# CELL ********************

lgbm.fit(X_train, y_train)

# CELL ********************

y_pred_lgbm = lgbm.predict(X_test)

# MARKDOWN ********************

# ### Default XGBoost

# CELL ********************

from xgboost import XGBClassifier

xgb = XGBClassifier()
cat_columns = X_train.select_dtypes(include=['category']).columns
X = X_train.copy()
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
y_train = y_train.astype('int')
y_test_xgb = y_test.astype('int')

# CELL ********************

xgb.fit(X, y_train)

# CELL ********************

X = X_test.copy()
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
y_pred_xgb = xgb.predict(X)


# CELL ********************

print('default xgboost accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred_xgb, y_test_xgb))
print('default lgbm accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred_lgbm, y_test))
print('flaml (10 min) accuracy', '=', 1 - sklearn_metric_loss_score('accuracy', y_pred, y_test))
