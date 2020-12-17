## aggregation features
from tqdm import tqdm 

def agg(df,agg_cols):
    for c in tqdm(agg_cols):
        print (c)
        print (c['agg'])
        new_feature = '{}_{}_{}'.format('_'.join(c['groupby']), c['agg'], c['target'])
        
        if c['agg'] == 'mode':
            df[new_feature] = df.groupby(c['groupby'])[c['target']].apply(pd.Series.mode).reset_index(drop=True)           
        elif c['agg'] == 'diff':
            df[new_feature] = df.groupby(c['groupby'])[c['target']].transform(lambda x: x.diff())
        elif c['agg'] == 'cumcount':
            df[new_feature] = df.groupby(c['groupby']).cumcount()
        elif c['agg'] == 'shift':
            df[new_feature] = df.groupby(c['groupby'])[c['target']].shift()            
        else:    
            df[new_feature] = df.groupby(c['groupby'])[c['target']].transform(c['agg'])

    return df

agg_cols = [    
    
# ---------------
#     position
# ---------------    
    
    {'groupby': ['position'], 'target':'education', 'agg':'mean'},       
    
    {'groupby': ['position'], 'target':'age', 'agg':'max'}, 
    {'groupby': ['position'], 'target':'age', 'agg':'min'},     
    {'groupby': ['position'], 'target':'age', 'agg':'mean'}, 
    {'groupby': ['position'], 'target':'age', 'agg':'std'}, 
    {'groupby': ['position'], 'target':'age', 'agg':'nunique'},     
]