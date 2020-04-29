import pandas as pd 
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
dfs = pd.read_csv('forestfires.csv')

dfs['month'] = le.fit_transform(dfs['month'])
dfs['day'] = le.fit_transform(dfs['day'])

dfs.to_csv('data.txt', header=None, index=None, sep=' ', mode='a')

print(dfs)