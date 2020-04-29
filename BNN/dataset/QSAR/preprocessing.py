import pandas as pd 

dfs = pd.read_csv('qsar_aquatic_toxicity.csv',delimiter=";",header=None)

dfs.to_csv('data.txt',header=None,index=None, sep=' ', mode='a')

print(dfs)