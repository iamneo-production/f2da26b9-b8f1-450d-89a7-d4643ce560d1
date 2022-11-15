#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


a = pd.read_csv('Debt_GDP.csv', on_bad_lines='skip')
b = pd.read_csv('GDP.csv', on_bad_lines='skip')
c = pd.read_csv('Inflation.csv', on_bad_lines='skip')
d = pd.read_csv('Production.csv', on_bad_lines='skip')


# In[ ]:


df = pd.concat([a, b])


# In[ ]:


import seaborn as sns


# In[ ]:


df = pd.concat([df, c])
df = pd.concat([df, d])
df.head(20)


# In[ ]:


cor = df.corr()

ax = sns.heatmap(cor, annot=True, cmap='rocket_r')


# In[ ]:


print(a.columns)
print(b.columns)
print(c.columns)
print(d.columns)

