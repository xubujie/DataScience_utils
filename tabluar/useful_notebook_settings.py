%load_ext blackcellmagic
# %%black
import pandas as pd
import numpy as np
pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 400)
import warnings
warnings.filterwarnings('ignore')

load = context.catalog.load
load_params = context.params

import matplotlib.pyplot as plt
%matplotlib inline
import sys

%load_ext blackcellmagic
%load_ext autoreload
%autoreload 2
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.repl.eagerEval.enabled",True)