import pandas as pd 

dfs = pd.read_excel('Folds5x2_pp.xlsx', sheet_name='Sheet1')

dfs.to_csv('data.txt', header=None, index=None, sep=' ', mode='a')
