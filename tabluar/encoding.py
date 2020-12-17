class KFoldTargetEncoderTrain():
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def transform(self,X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = False, random_state=2019)
        col_mean_name = self.colnames + '_TE'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    
                   np.corrcoef(X[self.targetName].values,
                               encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X
class KFoldTargetEncoderTest():
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
    def transform(self,X):
        mean =  self.train[[self.colNames,
                self.encodedName]].groupby(
                                self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        X[self.encodedName] = X[self.colNames]
#         X = X.replace({self.encodedName: dd})
        X[self.encodedName] = X[self.colNames].map(dd).fillna(0.5)
        return X


def numerical_encoding(df, feat, order = None):
    df[feat] = df[feat].astype('category').cat.as_ordered()
    df[feat] = df[feat].cat.codes+1
    return feat
def frequency_encoding(df,col):
    d = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(d)/d.max()
    return [[n],d]
def apply_frequency_encoding(df,col,mp,xx=1.0):
    cv = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(mp),n] = xx*np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    return [[n],mp]
def target_encoding(df, col, tar='HasDetections'):
    d = {}
    v = df[col].unique()
    for x in v:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col] == x].mean()
        d[x] = m
    n = col + '_TE'
    df[n] = df[col].map(d)
    return [[n], d]
# TARGET ENCODING first ct columns by freq
def target_encoding_partial(df,col,ct,tar='HasDetections',xx=0.5):
    d = {}
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    for x in nm:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col]==x].mean()
        d[x] = m
    n = col+"_TE"
    df[n] = df[col].map(d).fillna(xx)
    return [[n],d]
# TARGET ENCODING from dictionary
def target_encoding_test(df,col,mp,xx=0.5):
    n = col+"_TE"
    df[n] = df[col].map(mp).fillna(xx)
    return [[n],0]
# FREQUENCY ENCODING first ct columns by freq
def frequency_encoding_partial(df,col,ct):
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(nm),n] = np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    d = {}
    for x in nm: d[x] = cv[x]
    return [[n],d]
# FREQUENCY ENCODING from dictionary
def frequency_encoding_test(df,col,mp,xx=1.0):
    cv = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(mp),n] = xx*np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    return [[n],mp]


def nan_check(x):
    if isinstance(x,float):
        if math.isnan(x):
            return True
    return False