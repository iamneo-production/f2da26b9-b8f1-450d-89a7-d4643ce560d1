#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn import linear_model, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from datetime import datetime, timedelta
from pylab import rcParams
import os
import importlib
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')


# # DataPreparation

# In[2]:


# -------------------- PREPARE DATA -------------------- #

class DataLoader():
    def __init__(self, datasets_original, dataset_names):
        self.num_datasets = len(datasets_original)
        self.datasets_original = datasets_original
        self.dataset_names = dataset_names
        self.datases_revised = None
        self.drawdowns = None
        self.crashes = None
    
    def get_data_revised(self, crash_thresholds):
        datasets = []
        for d in self.datasets_original:
            data_original = pd.read_csv(d, index_col = 'Date')
            data_original.index = pd.to_datetime(data_original.index, format='%Y/%m/%d')
            data_norm = data_original['Close'] / data_original['Close'][-1]
            data_ch = data_original['Close'].pct_change()
            window = 10
            data_vol = data_original['Close'].pct_change().rolling(window).std()
            data = pd.concat([data_original['Close'], data_norm, data_ch, data_vol], axis=1).dropna()
            data.columns = ['price', 'norm', 'ch', 'vol']
            datasets.append(data)
        self.drawdowns = []
        self.crashes = []
        for df, ct in zip(datasets, crash_thresholds):
            pmin_pmax = (df['price'].diff(-1) > 0).astype(int).diff()
            pmax = pmin_pmax[pmin_pmax == 1]
            pmin = pmin_pmax[pmin_pmax == -1]
            # make sure drawdowns start with pmax, end with pmin:
            if pmin.index[0] < pmax.index[0]:
                pmin = pmin.drop(pmin.index[0])
            if pmin.index[-1] < pmax.index[-1]:
                pmax = pmax.drop(pmax.index[-1])
            D = (np.array(df['price'][pmin.index]) - np.array(df['price'][pmax.index]))/ np.array(df['price'][pmax.index])
            d = {'Date':pmax.index, 'drawdown':D, 'd_start': pmax.index,'d_end': pmin.index}    
            df_d = pd.DataFrame(d).set_index('Date')
            df_d.index = pd.to_datetime(df_d.index, format='%Y/%m/%d')
            df_d = df_d.reindex(df.index).fillna(0)
            df_d = df_d.sort_values(by='drawdown')
            df_d['rank'] = list(range(1,df_d.shape[0]+1))
            self.drawdowns.append(df_d)
            df_d = df_d.sort_values(by='Date')
            df_c = df_d[df_d['drawdown'] < ct]
            df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'rank']
            self.crashes.append(df_c)
        self.datasets_revised = []  
        for i in range(len(datasets)):
            self.datasets_revised.append(pd.concat([datasets[i],self.drawdowns[i]], axis=1))
        return self.datasets_revised, self.crashes

    def get_dfs_xy(self, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)     
        dfs_x, dfs_y = [], []
        for df, c in zip(self.datasets_revised, self.crashes):
            df['ch'] = df['ch'] / abs(df['ch']).mean()
            df['vol'] = df['vol'] / abs(df['vol']).mean()
            xy = {}
            for date in df.index[252:-126]: # <--subtract 126 days in the end
                xy[date] = list([df['ch'][(date-timedelta(5)):date].mean()])
                xy[date].append(df['ch'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['ch'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['ch'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['ch'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['ch'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['ch'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['ch'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date].append(df['vol'][(date-timedelta(5)):date].mean())
                xy[date].append(df['vol'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['vol'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['vol'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['vol'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['vol'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['vol'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['vol'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c for c in c['crash_st']]) for month in months]
            df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
            df_x = df_xy.iloc[:, :-len(months)]
            df_y = df_xy.iloc[:, -len(months):]
            dfs_x.append(df_x)
            dfs_y.append(df_y)
        return dfs_x, dfs_y
    
    def get_dfs_xy_predict(self, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)     
        dfs_x, dfs_y = [], []
        for df, c in zip(self.datasets_revised, self.crashes):
            df['ch'] = df['ch'] / abs(df['ch']).mean()
            df['vol'] = df['vol'] / abs(df['vol']).mean()
            xy = {}
            for date in df.index: # <--subtract 126 days in the end
                xy[date] = list([df['ch'][(date-timedelta(5)):date].mean()])
                xy[date].append(df['ch'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['ch'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['ch'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['ch'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['ch'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['ch'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['ch'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date].append(df['vol'][(date-timedelta(5)):date].mean())
                xy[date].append(df['vol'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['vol'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['vol'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['vol'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['vol'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['vol'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['vol'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c for c in c['crash_st']]) for month in months]
            df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
            df_x = df_xy.iloc[:, :-len(months)]
            df_y = df_xy.iloc[:, -len(months):]
            dfs_x.append(df_x)
            dfs_y.append(df_y)
        return dfs_x, dfs_y

    def get_train_test(self, dfs_x, dfs_y, datasets, test_data):
        for i, name in enumerate(datasets):
            if name == test_data:
                index = i
        dfs_x_copy = list(dfs_x)
        dfs_y_copy = list(dfs_y)
        np_x_test = None
        np_y_test = None
        if test_data:
            df_x_test = dfs_x_copy.pop(index)
            df_y_test = dfs_y_copy.pop(index)
            np_x_test = np.array(df_x_test)
            np_y_test = np.array(df_y_test)
        np_x_train = np.concatenate(([np.array(x) for x in dfs_x_copy]))
        np_y_train = np.concatenate(([np.array(y) for y in dfs_y_copy]))
        return np_x_train, np_y_train, np_x_test, np_y_test

    def split_results(self, df_combined, dfs_xy, dataset_names, test_data, y_pred_t_bin, y_pred_tr_bin, y_train, y_test):
        df_combined = [dfc.reindex(dfs.index) for dfc, dfs in zip(df_combined, dfs_xy)]
        dfs_predict = []
        n = 0
        for df, name in zip(df_combined, dataset_names):
            if name == test_data:
                df['y'] = y_test
                df['y_pred'] = y_pred_t_bin
                dfs_predict.append(df)
            else:
                df['y'] = y_train[n:n+df.shape[0]]
                df['y_pred'] = y_pred_tr_bin[n:n+df.shape[0]]
                dfs_predict.append(df)
                n += df.shape[0]
        return dfs_predict


# # DataEvaluation

# In[3]:


# -------------------- EVALUATE DATA -------------------- #
class EvaluateResults():
    def __init__(self, y_train_all, y_val_all, y_pred_tr_all, y_pred_val_all, model_name, test_data):
        self.y_train_all = y_train_all
        self.y_val_all = y_val_all
        self.y_pred_val_all = y_pred_val_all
        self.y_pred_tr_all = y_pred_tr_all
        self.model_name = model_name
        self.test_data = test_data
    
    def find_threshold(self, beta, threshold_min, threshold_max, resolution=20):
        precision_tr_all, recall_tr_all, accuracy_tr_all = [], [], []
        precision_t_all, recall_t_all, accuracy_t_all = [], [], [] 
        fbeta_tr_all, fbeta_t_all = [], []
        thresholds = list(np.linspace(threshold_min, threshold_max, resolution))
        for threshold in thresholds:
            precision_tr, recall_tr, accuracy_tr = [], [], []
            precision_val, recall_val, accuracy_val = [], [], []
            y_pred_val_bin_all, y_pred_tr_bin_all = [], []
            score_fbeta_tr, score_fbeta_t = [], []
            for y_train, y_val, y_pred_tr, y_pred_val in zip(self.y_train_all, 
                                                            self.y_val_all, \
                                                            self.y_pred_tr_all,\
                                                            self.y_pred_val_all):
                y_pred_tr_bin = y_pred_tr > threshold
                y_pred_tr_bin = y_pred_tr_bin.astype(int)
                y_pred_tr_bin_all.append(y_pred_tr_bin)
                precision_tr.append(metrics.precision_score(y_train, y_pred_tr_bin))
                recall_tr.append(metrics.recall_score(y_train, y_pred_tr_bin))
                accuracy_tr.append(metrics.accuracy_score(y_train, y_pred_tr_bin))
                score_fbeta_tr.append(metrics.fbeta_score(y_train, y_pred_tr_bin, beta=beta))
                y_pred_val_bin = y_pred_val > threshold
                y_pred_val_bin = y_pred_val_bin.astype(int)
                y_pred_val_bin_all.append(y_pred_val_bin)
                precision_val.append(metrics.precision_score(y_val, y_pred_val_bin))
                recall_val.append(metrics.recall_score(y_val, y_pred_val_bin))
                accuracy_val.append(metrics.accuracy_score(y_val, y_pred_val_bin))
                score_fbeta_t.append(metrics.fbeta_score(y_val, y_pred_val_bin, beta=beta))
            precision_tr_all.append(np.mean(precision_tr)) 
            precision_t_all.append(np.mean(precision_val)) 
            recall_tr_all.append(np.mean(recall_tr)) 
            recall_t_all.append(np.mean(recall_val))
            accuracy_tr_all.append(np.mean(accuracy_tr)) 
            accuracy_t_all.append(np.mean(accuracy_val))
            fbeta_tr_all.append(np.mean(score_fbeta_tr))
            fbeta_t_all.append(np.mean(score_fbeta_t))
        plt.subplot(1,3,1)
        plt.plot(thresholds, precision_tr_all, color='blue')
        plt.plot(thresholds, precision_t_all, color='red')
        plt.title('Precision by threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.legend(['training set', 'validation set'])
        plt.grid()
        plt.subplot(1,3,2)
        plt.plot(thresholds, recall_tr_all, color='blue')
        plt.plot(thresholds, recall_t_all, color='red')
        plt.title('Recall by threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.legend(['training set', 'validation set'])
        plt.grid()
        plt.subplot(1,3,3)
        plt.plot(thresholds, fbeta_tr_all, color='blue')
        plt.plot(thresholds, fbeta_t_all, color='red')
        plt.title('F-beta score by threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F-beta score')
        plt.legend(['training set', 'validation set'])
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    def training_results(self, threshold, training_set_names, beta=2):
        precision_tr, recall_tr, accuracy_tr, score_fbeta_tr = [], [], [], []
        precision_val, recall_val, accuracy_val, score_fbeta_val = [], [], [], []
        y_pred_tr_bin_all, y_pred_val_bin_all = [], []
        for y_train, y_val, y_pred_tr, y_pred_val in zip(self.y_train_all, self.y_val_all,                                                         self.y_pred_tr_all, self.y_pred_val_all):
            if threshold:
                y_pred_tr_bin = y_pred_tr > threshold
                y_pred_tr_bin = y_pred_tr_bin.astype(int)
            else:
                y_pred_tr_bin = y_pred_tr.astype(int)
            y_pred_tr_bin_all.append(y_pred_tr_bin)
            precision_tr.append(metrics.precision_score(y_train, y_pred_tr_bin))
            recall_tr.append(metrics.recall_score(y_train, y_pred_tr_bin))
            accuracy_tr.append(metrics.accuracy_score(y_train, y_pred_tr_bin))
            score_fbeta_tr.append(metrics.fbeta_score(y_train, y_pred_tr_bin, beta=beta))
            if threshold:
                y_pred_val_bin = y_pred_val > threshold
                y_pred_val_bin = y_pred_val_bin.astype(int)
            else:
                y_pred_val_bin = y_pred_val.astype(int)
            y_pred_val_bin_all.append(y_pred_val_bin)
            precision_val.append(metrics.precision_score(y_val, y_pred_val_bin))
            recall_val.append(metrics.recall_score(y_val, y_pred_val_bin))
            accuracy_val.append(metrics.accuracy_score(y_val, y_pred_val_bin))
            score_fbeta_val.append(metrics.fbeta_score(y_val, y_pred_val_bin, beta=beta))
        
        y_tr_pos = [np.mean(y) for y in self.y_train_all]
        y_tr_pred_pos = [np.mean(y_pred) for y_pred in y_pred_tr_bin_all]
        y_val_pos = [np.mean(y) for y in self.y_val_all]
        y_val_pred_pos = [np.mean(y_pred) for y_pred in y_pred_val_bin_all]
        d = {'positive actual train': np.round(y_tr_pos, 2), 'positive pred train': np.round(y_tr_pred_pos, 2), 'precision train': np.round(precision_tr,2),              'recall train': np.round(recall_tr,2),              'accuracy_train': np.round(accuracy_tr,2),              'score_fbeta train': np.round(score_fbeta_tr,2),              'positive actual val': np.round(y_val_pos, 2),              'positive pred val': np.round(y_val_pred_pos, 2),              'precision val': np.round(precision_val, 2),              'recall val': np.round(recall_val, 2),              'accuracy val': np.round(accuracy_val,2),              'score fbeta val': np.round(score_fbeta_val,2)}
        results_all = pd.DataFrame.from_dict(d, orient='index')
        results_all.columns = training_set_names
        print('Results for each train/val split:')
        print(results_all)
        print('\n')
        
        # calculate precision, recall, accuracy for comparable random model
        sum_tr = sum([len(tr) for tr in self.y_train_all])
        pos_tr = sum([sum(tr) for tr in self.y_train_all])
        sum_val = sum([len(t) for t in self.y_val_all])
        pos_val = sum([sum(t) for t in self.y_val_all])
        sum_tr_pred = sum([len(tr) for tr in y_pred_tr_bin_all])
        pos_tr_pred = sum([sum(tr) for tr in y_pred_tr_bin_all])
        sum_val_pred = sum([len(t) for t in y_pred_val_bin_all])
        pos_val_pred = sum([sum(t) for t in y_pred_val_bin_all])

        y_train_pos_actual = pos_tr / sum_tr
        y_train_pos_pred = pos_tr_pred / sum_tr_pred
        rnd_TP = y_train_pos_pred * y_train_pos_actual
        rnd_FP = y_train_pos_pred * (1 - y_train_pos_actual)
        rnd_TN = (1 - y_train_pos_pred) * (1 - y_train_pos_actual)
        rnd_FN = (1 - y_train_pos_pred) * y_train_pos_actual
        rnd_pr_tr = rnd_TP / (rnd_TP+rnd_FP)
        rnd_re_tr = rnd_TP / (rnd_TP+rnd_FN)
        rnd_ac_tr = rnd_TP + rnd_TN
        y_val_pos_actual = pos_val / sum_val
        y_val_pos_pred = pos_val_pred / sum_val_pred
        rnd_TP = y_val_pos_pred * y_val_pos_actual
        rnd_FP = y_val_pos_pred * (1 - y_val_pos_actual)
        rnd_TN = (1 - y_val_pos_pred) * (1 - y_val_pos_actual)
        rnd_FN = (1 - y_val_pos_pred) * y_val_pos_actual
        rnd_pr_val = rnd_TP / (rnd_TP+rnd_FP)
        rnd_re_val = rnd_TP / (rnd_TP+rnd_FN)
        rnd_ac_val = rnd_TP + rnd_TN
        rnd_fbeta_tr = (1 + beta ** 2) * (rnd_pr_tr * rnd_re_tr) / ((beta ** 2 * rnd_pr_tr) + rnd_re_tr)
        rnd_fbeta_val = (1 + beta ** 2) * (rnd_pr_val * rnd_re_val) /((beta ** 2 * rnd_pr_val) + rnd_re_val)
        
        print('Results - average over all train/val splits:')
        print('Positive train cases actual:            '+ str(round(y_train_pos_actual, 2)))
        print('Positive train cases predicted:         '+ str(round(y_train_pos_pred, 2)))
        print('Avg precision train (model/random):     '+ str(round(np.mean(precision_tr), 2))              + ' / ' + str(round(rnd_pr_tr, 2)))
        print('Avg recall train (model/random):        '+ str(round(np.mean(recall_tr), 2))              + ' / ' + str(round(rnd_re_tr, 2)))
        print('Avg accuracy train (model/random):      '+ str(round(np.mean(accuracy_tr), 2))              + ' / ' + str(round(rnd_ac_tr, 2)))
        print('Score train fbeta:                      '+ str(round(np.mean(score_fbeta_tr), 2))              + ' / ' + str(round(rnd_fbeta_tr, 2)))
        print('Positive validation cases actual:       '+ str(round(y_val_pos_actual, 2)))
        print('Positive validation cases predicted:    '+ str(round(y_val_pos_pred, 2)))
        print('Avg precision validation (model/random):'+ str(round(np.mean(precision_val), 2))              + ' / ' + str(round(rnd_pr_val, 2)))
        print('Avg recall validation (model/random):   '+ str(round(np.mean(recall_val), 2))              + ' / ' + str(round(rnd_re_val, 2)))
        print('Avg accuracy validation (model/random): '+ str(round(np.mean(accuracy_val), 2))              + ' / ' + str(round(rnd_ac_val, 2)))
        print('Score validation fbeta:                 '+ str(round(np.mean(score_fbeta_val), 2))              + ' / ' + str(round(rnd_fbeta_val, 2)))

    def test_results(self, y_test, y_pred_t, threshold, beta=2):
        if threshold:
            y_pred_t_bin = y_pred_t > threshold
            y_pred_t_bin = y_pred_t_bin.astype(int)
        else:
            y_pred_t_bin = y_pred_t.astype(int)
        precision_t = metrics.precision_score(y_test, y_pred_t_bin)
        recall_t = metrics.recall_score(y_test, y_pred_t_bin)
        accuracy_t = metrics.accuracy_score(y_test, y_pred_t_bin)
        score_fbeta_t = metrics.fbeta_score(y_test, y_pred_t_bin, beta=beta)
        y_t_pos = np.mean(y_test)
        y_t_pos_actual = sum(y_test) / len(y_test)
        y_t_pos_pred = np.mean(y_pred_t_bin)
        rnd_TP = y_t_pos_pred * y_t_pos
        rnd_FP = y_t_pos_pred * (1 - y_t_pos)
        rnd_TN = (1 - y_t_pos_pred) * (1 - y_t_pos)
        rnd_FN = (1 - y_t_pos_pred) * y_t_pos
        rnd_pr_t = rnd_TP / (rnd_TP+rnd_FP)
        rnd_re_t = rnd_TP / (rnd_TP+rnd_FN)
        rnd_ac_t = rnd_TP + rnd_TN
        rnd_fbeta = (1 + beta ** 2) * (rnd_pr_t * rnd_re_t) / ((beta ** 2 * rnd_pr_t) + rnd_re_t)
        print('Test results (test set: S&P 500):')
        print('Positive test cases actual:         '+ str(round(y_t_pos_actual, 2)))
        print('Positive test cases predicted:      '+ str(round(y_t_pos_pred, 2)))
        print('Precision test (model/random):      '+ str(round(np.mean(precision_t), 2))              + ' / ' + str(round(rnd_pr_t, 2)))
        print('Recall test (model/random):         '+ str(round(np.mean(recall_t), 2))              + ' / ' + str(round(rnd_re_t, 2)))
        print('Accuracy test (model/random):       '+ str(round(np.mean(accuracy_t), 2))              + ' / ' + str(round(rnd_ac_t, 2)))
        print('Score test fbeta:                   '+ str(round(np.mean(score_fbeta_t), 2))              + ' / ' + str(round(rnd_fbeta, 2)))
        return y_pred_t_bin
        
    def plot_test_results(self, df, c, t_start, t_end):
        t_start = [datetime.strptime(t, '%Y-%m-%d') for t in t_start]
        t_end = [datetime.strptime(t, '%Y-%m-%d') for t in t_end]
        for t1, t2 in zip(t_start, t_end):
            gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1])
            plt.subplot(gs[0])
            y_start = list(df[t1:t2][df.loc[t1:t2, 'y'].diff(-1) < 0].index)
            y_end = list(df[t1:t2][df.loc[t1:t2, 'y'].diff(-1) > 0].index)
            crash_st = list(filter(lambda x: x > t1 and x < t2, c['crash_st']))
            crash_end = list(filter(lambda x: x > t1 and x < t2, c['crash_end']))
            [plt.axvspan(x1, x2, alpha=0.2, color='red') for x1, x2 in zip(y_start, y_end)]
            [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crash_st, crash_end)]
            df_norm = df['price'][t1:t2] / df['price'][t1:t2].max()
            plt.plot(df_norm[t1:t2], color='blue') 
            plt.title(self.model_name + ' Testcase: ' + self.test_data + ' ' + str(t1.year) + '-'                       + str(t2.year))
            plt.legend(['price', 'downturn / crash'])
            plt.xticks([])
            plt.grid()     
            plt.subplot(gs[1])
            plt.plot(df.loc[t1:t2, 'vol'])
            [plt.axvspan(x1, x2, alpha=0.2, color='red') for x1, x2 in zip(y_start, y_end)]
            [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crash_st, crash_end)]
            plt.legend(['Volatility'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.grid()
            plt.xticks([])
            plt.subplot(gs[2])
            plt.plot(df['y'][t1:t2], color='black')
            plt.plot(df['y_pred'][t1:t2].rolling(10).mean(), color='darkred', linewidth=0.8)
            [plt.axvspan(x1, x2, alpha=0.2, color='red') for x1, x2 in zip(y_start, y_end)]
            [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crash_st, crash_end)]
            plt.legend(['crash within 6m', 'crash predictor'])
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.grid()
            plt.show()  


# # DataCollection

# In[43]:


# -------------------- Data preparation -------------------- #
#os.chdir('C:/Users/PRIYANKA/OneDrive/Desktop/Projects/Stockcrashprediction/Project/data')
datasets_original = ['^BSESN.csv', '^GSPC.csv', '^N225.csv', 'SSE.csv','^HSI.csv', '^SSMI.csv', '^BVSP.csv']
dataset_names = [ 'BSESN', 'S&P 500', 'N225', 'SSE', 'HSI','SMI', 'BVSP']

# specify drawdown thresholds for crashes (determined in exploration.ipynb):
# crashes according to Jacobsson:
crash_thresholds = [-0.193619, -0.0936, -0.185957	, -0.170386, -0.196736,  -0.203478	, -0.197904]
# crashes according to Sornette:
# crash_thresholds = [-0.1053, -0.1495, -0.1706, -0.2334, -0.1563, -0.1492, -0.2264]
months = [1, 3, 6, 12, 18, 24]   # <-- predict if crash n months ahead (use: 1, 3 or 6)
data = DataLoader(datasets_original, dataset_names)
datasets_revised, crashes = data.get_data_revised(crash_thresholds)
dfs_x, dfs_y = data.get_dfs_xy(months=months)


# # DataTraining

# In[44]:


# -------------------- Train Linear Regression -------------------- #
model_name = 'Linear Regression'
test_data = 'BSESN'
month_prediction = 3
index_test = [i for i, name in enumerate(dataset_names) if name == test_data][0]
index_month = [i for i, m in enumerate(months) if m == month_prediction][0]
training_set_names = list(dataset_names)
training_set_names.pop(index_test)
dfs_x_training = list(dfs_x)
dfs_x_training.pop(index_test)
dfs_y_training = list(dfs_y)
dfs_y_training.pop(index_test)
y_train_all, y_val_all = [], []
y_pred_train_all, y_pred_val_all = [], []
for val_data in training_set_names:
    x_train, y_train, x_val, y_val = data.get_train_test(dfs_x_training, dfs_y_training,             training_set_names, test_data=val_data)
    y_train, y_val = y_train[:, index_month].astype(int), y_val[:, index_month].astype(int)
    y_train_all.append(y_train)
    y_val_all.append(y_val)
    print('Train ' + str(model_name) + ' - validation data: ' + str(val_data))
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_train_all.append(y_pred_train)
    y_pred_val = model.predict(x_val)
    y_pred_val_all.append(y_pred_val)


# # Threshold

# In[45]:


# -------------------- Find best threshold -------------------- #
beta = 2
rcParams['figure.figsize'] = 14, 4
eval_ = EvaluateResults(y_train_all, y_val_all, y_pred_train_all, y_pred_val_all, model_name,            test_data)
eval_.find_threshold(beta=beta, threshold_min=0.01, threshold_max=0.15, resolution=40)


# # Results

# In[46]:


# -------------------- Evaluate results -------------------- #
threshold = 0.075
beta = 2
print(model_name)
print('\n')
print('Predict crash in:            ' + str(month_prediction) + ' months')
print('Threshold for positives:     ' + str(threshold))
print('Number of features:          ' + str(dfs_x[0].shape[1]))
print('Number of rows for training: ' + str(len(y_pred_train_all[0]) + len(y_pred_val_all[0])))
print('\n')
eval_.training_results(threshold, training_set_names, beta=beta)


# # Testing 

# In[47]:


# -------------------- Test model -------------------- #
x_train, y_train, x_test, y_test = data.get_train_test(dfs_x, dfs_y, dataset_names, test_data)
y_train, y_test = y_train[:, index_month].astype(int), y_test[:, index_month].astype(int)
lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
y_pred_test_bin = eval_.test_results(y_test, y_pred_test, threshold, beta=beta)


# # Ploting

# In[48]:


# -------------------- Plot test results -------------------- #
df = datasets_revised[index_test].reindex(dfs_x[index_test].index)
df['y'] = y_test
df['y_pred'] = y_pred_test_bin
c = crashes[index_test]
t_start = ['1995-01-01', '2004-01-01','2010-01-01']
t_end = ['2003-01-01', '2010-01-01', '2016-01-01']
rcParams['figure.figsize'] = 10, 6
eval_.plot_test_results(df, c, t_start, t_end)


# # Indian Prediction

# In[75]:



x_train, y_train, _, _ = data.get_train_test(dfs_x, dfs_y, dataset_names, test_data=None)
thresholds = [0.07, 0.075, 0.085, 0.08, 0.075, 0.07] # <-- determined by eval model for each prediction
dataset_original = ['CSVForDate_2000-2022.csv']
dataset_name = ['CSVForDate_13_22.csv']
crash_threshold = [-0.193619]
data_new = DataLoader(dataset_original, dataset_name)
dataset_revised, crashes = data_new.get_data_revised(crash_threshold)
dfs_x_new, dfs_y_new = data_new.get_dfs_xy_predict(months=months)
x_new, _, _, _ = data_new.get_train_test(dfs_x_new, dfs_y_new, dataset_name, test_data=None)


for index_month in range(len(months)):
    y_train_ = y_train[:, index_month].astype(int)
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train, y_train_)
    y_pred_new = model.predict(x_new)
    y_pred_new_bin = y_pred_new > thresholds[index_month]
    y_pred_new_bin = y_pred_new_bin.astype(int)
    current_pred = np.dot(np.linspace(0,1,42)/sum(np.linspace(0,1,42)), y_pred_new_bin[-42:])
    k = str(model_name) + ' prediction of a crash within ' + str(months[index_month])           + ' months: ' + str(np.round(current_pred, 2))
    print(k)
    #lst.append(k)


# #Web Interface

# In[129]:


get_ipython().system('pip install -q streamlit')


# In[138]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nimport pandas as pd\nfrom PIL import Image\nst.title("Stock Crash Prediction in Year 2022")\nimg = Image.open("crash.png")\nst.subheader("*Previous Crash*")\nst.image(img)\nst.header("Expected Crash")\n\nst.subheader("Linear Regression prediction of a crash")\n\ndf = pd.DataFrame({\'Within_months\':[1, 3, 6, 12, 18, 24], \'Confidence Score\':[0.0, 0.0, 0.0, 0.0, 0.56, 0.97]})\n\nstyler = df.style.hide_index().format(subset=[\'Confidence Score\'], decimal=\'.\', precision=3).bar(subset=[\'Confidence Score\'], align="mid")\n\nst.write(styler.to_html(), unsafe_allow_html=True)\n')


# In[139]:


get_ipython().system('streamlit run app.py & npx localtunnel --port 8501')

