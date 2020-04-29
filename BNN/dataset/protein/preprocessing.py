import pandas as pd 

dfs = pd.read_csv('CASP.csv')

dfs=dfs[["F1","F2","F3","F4","F5","F6","F7","F8","F9","RMSD"]]

dfs.to_csv('data.txt', header=None, index=None, sep=' ', mode='a')
