import numpy as  np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

class data_split:

	def __init__(self,dataset_name,path,seed,test_size=0.1):

		self.seed=seed
		# np.random.seed(1)
		self.dataset_name=dataset_name
		self.path=path
		self.test_size=test_size

	def split(self):

		data=np.loadtxt(self.path+self.dataset_name+'/data.txt')

		X=data[:,:-1]
		Y=data[:,-1]

		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.seed)

		return X_train, X_test, y_train, y_test

	def shape(self):

		X_train,X_test,y_train,y_test=self.split()
		input_size=np.shape(X_train)[1]

		output_size=1

		return input_size,output_size

	def normalize(self):

		X_train,X_test,y_train,y_test=self.split()

		X=StandardScaler()
		X_train=X.fit_transform(X_train)
		X_test=X.transform(X_test)

		Y=StandardScaler()
		y_train=Y.fit_transform(y_train.reshape(-1,1))
		y_test = y_test.reshape(-1,1)

		y_mean=Y.mean_
		y_std=Y.scale_

		return torch.from_numpy(X_train), torch.from_numpy(X_test), torch.from_numpy(y_train), torch.from_numpy(y_test), torch.from_numpy(y_mean), torch.from_numpy(y_std)


