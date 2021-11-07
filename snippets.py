
#### kedro 
import pandas as pd
import numpy as np
pd.options.display.float_format = "{:,.1f}".format

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 400)

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import DataFrame, Row, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col as C
import seaborn as sns
import matplotlib.pyplot as plt
def pshow(df, n=5):
    return df.toPandas().limit(5)
load_params = context.params
%matplotlib inline
%load_ext autoreload
%autoreload 2

### kedro snippets