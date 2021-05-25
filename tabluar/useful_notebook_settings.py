%load_ext blackcellmagic
# %%black


# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

print(f'value: {1.234 :,.2f}')

import pandas as pd
import numpy as np
pd.options.display.float_format = "{:,.2f}".format

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 400)

import warnings
warnings.filterwarnings('ignore')

load = context.catalog.load
load_params = context.params

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import matplotlib.pyplot as plt
%matplotlib inline
def nrows(n):
    pd.options.display.max_rows = n
def show(data, n=5):
    nrows(n)
    return data.limit(n).toPandas()
def plots(figsize=(8, 5)):
    return plt.subplots(figsize=figsize)

import sys
sys.path.append("../src/")

%load_ext blackcellmagic
%load_ext autoreload
%autoreload 2