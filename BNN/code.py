import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.optim as optim
from data import data_split
import time
import sys
import argparse

class Model:
	def __init__(self,input_size, hidden_size, output_size, K, alpha, v_prior, init_scale, offset,seed):
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.output_size=output_size
		self.q = {'w1': {}, 'b1':{}, 'w2':{}, 'b2':{}}
		self.init_scale=init_scale
		self.offset=offset
		self.seed=seed
		self.K=K
		self.v_prior=v_prior
		self.alpha=alpha
		torch.manual_seed(0)

	def initialise_q(self):

		dtype = torch.FloatTensor

		self.q['w1']['mean'] = Variable(self.init_scale*torch.randn(self.input_size, self.hidden_size).type(dtype),requires_grad=True)
		self.q['w1']['sd'] = Variable(torch.exp(self.offset+self.init_scale*torch.randn(self.input_size, self.hidden_size).type(dtype)),requires_grad=True)

		self.q['b1']['mean'] = Variable(self.init_scale*torch.randn(1, self.hidden_size).type(dtype),requires_grad=True)
		self.q['b1']['sd'] = Variable(torch.exp(self.offset+self.init_scale*torch.randn(1, self.hidden_size).type(dtype)),requires_grad=True)

		self.q['w2']['mean'] = Variable(self.init_scale*torch.randn(self.hidden_size,self.output_size).type(dtype),requires_grad=True)
		self.q['w2']['sd'] = Variable(torch.exp(self.offset+self.init_scale*torch.randn(self.hidden_size,self.output_size).type(dtype)),requires_grad=True)

		self.q['b2']['mean'] = Variable(self.init_scale*torch.randn(1,self.output_size).type(dtype),requires_grad=True)
		self.q['b2']['sd'] = Variable(torch.exp(self.offset+self.init_scale*torch.randn(1,self.output_size).type(dtype)),requires_grad=True)

		self.v_noise = Variable(torch.tensor(1).type(dtype),requires_grad=True)


	def sample(self):
	    sampled_q = {}
	    for key in self.q.keys(): # for each of w1, b1, w2, b2        
	        dim = [self.K] + list(self.q[key]['mean'].shape) # Define dimensions of sampled tensor: (K, w_rows, w_columns)
	        sampled_q[key] = torch.randn(dim, dtype = float) * self.q[key]['sd'] + self.q[key]['mean'] # Take samples
	    return sampled_q # size (K, input, output)

	def predict(self,X, sampled_q):
	    fc1_out = F.relu(torch.add(torch.matmul(X, sampled_q['w1']), sampled_q['b1'])) # (K, input, hidden)
	    y_predict = torch.add(torch.matmul(fc1_out, sampled_q['w2']), sampled_q['b2']) # (K, hidden, output)
	    return y_predict # size (K, hidden, output)


	def flatten(self,sampled_q):
	    ''' Flatten sampled_q to dim(K, L); mean of q to dim(1, L); sd of q to dim(1, L) '''
	    for k in sampled_q.keys():
	        if k == 'w1':
	            sampled_q_flat = torch.flatten(sampled_q[k], start_dim = 1)
	            q_mean_flat = torch.flatten(self.q[k]['mean'])
	            q_sd_flat = torch.flatten(self.q[k]['sd'])
	        else:
	            sampled_q_flat = torch.cat((sampled_q_flat, torch.flatten(sampled_q[k], start_dim = 1)), dim = 1)
	            q_mean_flat = torch.cat((q_mean_flat, torch.flatten(self.q[k]['mean'])))
	            q_sd_flat = torch.cat((q_sd_flat, torch.flatten(self.q[k]['sd'])))
	    return q_mean_flat.view((1, -1)), q_sd_flat.view((1, -1)), sampled_q_flat

	def adjust_parameters_q(self):
	    q_adjusted = {}
	    for k in self.q.keys():
	        q_adjusted[k] = {}
	        q_adjusted[k]['sd'] = torch.sqrt(self.q[k]['sd']**2 * self.v_prior / (self.v_prior + self.q[k]['sd']**2))
	        q_adjusted[k]['mean'] = self.q[k]['mean'] * (self.v_prior / (self.v_prior + self.q[k]['sd']**2))
	    return q_adjusted

	def get_error_and_ll(self, X, y,location, scale):
		v_noise = self.v_noise * scale**2
		sampled_q = self.sample()
		y_predict = self.predict(X, sampled_q) * scale + location
		error = torch.sqrt(torch.mean((y - torch.mean(y_predict, 0))**2)).item()
		log_factor = -0.5 * torch.log(2 * math.pi *(v_noise)) - 0.5 * (y - y_predict)**2 / v_noise
		ll = torch.mean(torch.logsumexp(log_factor - torch.log(torch.tensor([self.K], dtype = float)), 0)).item()
		''' Note ll differs from log_likelihood. 
		    ll takes the average log-likelihood across N (data points); log_likelihood takes the sum '''
		return error, ll

	def compute_loss(self, y, y_predict, sampled_q, N):
	    ''' 
	    K = number of times BNN is sampled
	    N = total sample size
	    L = total number of weights (includin g bias terms)
	    F = number of features

	    Dimensions:
	    y_predict           (K x N x output_size)               * assume 1 output
	    y                   (N x output_size)                   * assume 1 output
	    X                   (N x F)
	    v_prior             scalar, defaults to 1
	    v_noise             scalar, defaults to 1
	    sampled_q           dictionary:
	        sampled_q['w1']     (K x F x hidden_layer_size)
	        sampled_q['b1']     (K x 1 x hidden_layer_size)
	        sampled_q['w2']     (K x hidden_layer_size x 1)
	        sampled_q['b2']     (K x 1 x 1)
	    q                   dictionary of dictionaries:
	        q['w1']['mean']     (F x hidden_layer_size)             
	        q['b1']['mean']     (1 x hidden_layer_size)             
	        q['w2']['mean']     (hidden_layers x output_layer_size) * assume 1 output
	        q['b2']['mean']     (1 x output_layer_size)             * assume 1 output

	    '''
	    # Flatten q and sampled q
	    q_mean_flat, q_sd_flat, sampled_q_flat = self.flatten(sampled_q) # shapes (1, L), (1, L), (K, L), where L = total num of weights

	    # Compute common components of loss
	    avg_log_likelihood = torch.mean((-0.5 * torch.log(2 * math.pi * self.v_noise) - 0.5 * (y - y_predict)**2 / self.v_noise), 1) # K x 1
	    log_q = torch.sum((-0.5 * torch.log(2 * math.pi * (q_sd_flat**2)) - 0.5 * ((sampled_q_flat - q_mean_flat) / q_sd_flat)**2), 1, keepdim = True) # K x 1
	    log_p0 = torch.sum((-0.5 * np.log(2 * math.pi * self.v_prior) - 0.5 * (sampled_q_flat **2 / self.v_prior)), 1, keepdim = True) # K x 1
	    log_F = log_p0 + N * avg_log_likelihood - log_q

	    # Compute loss for different alpha values
	    if self.alpha == 1:
	        L_vi = torch.mean(log_F)
	        loss = -L_vi

	    elif self.alpha == -math.inf:
	        L_vr = torch.max(log_F)
	        loss = -L_vr

	    elif self.alpha == math.inf:
	        L_vr = torch.min(log_F)
	        loss = -L_vr

	    else:
	        L_vr = (torch.logsumexp((1-self.alpha)*log_F, dim = 0) - torch.log(torch.tensor(self.K, dtype = float))) / (1-self.alpha)
	        loss = -(L_vr)


	    return loss

	def fit_q(self,data, batch_size, epochs, learning_rate):

		X_train, X_test, y_train, y_test, y_mean, y_std = data.normalize()

		iterations=int(X_train.shape[0]/batch_size)

		optimizer = optim.Adam([self.q['w1']['mean'], self.q['w1']['sd'], self.q['b1']['mean'], self.q['b1']['sd'], self.q['w2']['mean'], self.q['w2']['sd'], self.q['b2']['mean'], self.q['b2']['sd'],self.v_noise], lr=learning_rate)

		print("     Epoch     |     Error     | Log-likelihood|     Loss   ")
                          
		for epoch in range(epochs):

			# Shuffle data
			idx = list(range(X_train.shape[0]))
			np.random.shuffle(idx) # np seed handling required
			X_train = X_train[idx]
			y_train = y_train[idx]

			if epoch%100==0:

				# Print error, log-likelihood, and loss
				error, ll = self.get_error_and_ll(X_train, y_train,location = 0, scale = 1)

				# Print loss
				sampled_q = self.sample()
				y_predict = self.predict(X_train, sampled_q)
				loss= self.compute_loss(y_train, y_predict, sampled_q, X_train.shape[0])
				print("{0:15}|{1:15}|{2:15}|{3:15}".format(epoch, round(error, 5), round(ll, 5), round(loss, 5)))


			# Iterate over all batches
			for i in range(iterations):

				X_train_batch=X_train[i*batch_size:(i+1)*batch_size]
				y_train_batch=y_train[i*batch_size:(i+1)*batch_size]

				# Take K instances of weights from q
				sampled_q = self.sample()

				# Predict output, and compute loss
				y_predict = self.predict(X_train_batch, sampled_q)
				loss= self.compute_loss(y_train_batch, y_predict,sampled_q, X_train.shape[0])

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		error, ll = self.get_error_and_ll( X_test, y_test, y_mean, y_std)

		return error,ll

def main(dataset,alpha,learning_rate = 0.001, v_prior = 1.0,batch_size = 32,epochs = 500,K = 100,hidden_size = 100,offset=-10,init_scale=0.1,seed=0):

	data=data_split(dataset,'/home/rohan/Desktop/Projects/AML_project/VRbound/BayesianNN/data/',seed,0.1)

	input_size,output_size=data.shape()

	if dataset == ('protein' or 'year'):
		hidden_size = 100
		K = 10
	else:
		hidden_size = 50
		K = 100

	model=Model(input_size, hidden_size, output_size, K, alpha, v_prior, init_scale, offset,seed)
	model.initialise_q()
	start_time = time.time()
	output=model.fit_q(data, batch_size, epochs, learning_rate)
	running_time = time.time() - start_time

	params=[dataset,alpha,seed]
	path=""
	for i in params:
		path+=str(i)+"_"
	save_file='./Results/'+dataset+'/'+path+'.npy'
	np.save(save_file,output)

	print('Dataset: '+dataset+' Alpha: '+str(alpha)+' seed: '+str(seed)+' RMSE: '+str(output[0])+' NLL: '+str(-output[1])+' running_time: '+str(running_time))


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Running the program')
	parser.add_argument('--dataset',action="store",type=str,default='boston',dest='dataset')
	parser.add_argument('--alpha',action="store",type=float,default=1,dest='alpha')
	parser.add_argument('--seed',action="store",type=int,default=42,dest='seed')

	parser.add_argument('--lr',action="store",type=float,default=0.001,dest='lr')
	parser.add_argument('--v_prior',action="store",type=float,default=1.0,dest='v_prior')
	parser.add_argument('--batch_size',action="store",type=int,default=32,dest='batch_size')
	parser.add_argument('--epochs',action="store",type=int,default=500 ,dest='epochs')
	parser.add_argument('--K',action="store",type=int,default=100,dest='K')
	parser.add_argument('--hidden_size',action="store",type=int, default=100 ,dest='hidden_size')
	parser.add_argument('--offset',action="store",type=float,default=-10,dest='offset')
	parser.add_argument('--init_scale',action="store",type=float, default=0.1 ,dest='init_scale')


	results = parser.parse_args()
	dataset=results.dataset
	alpha=results.alpha
	seed=results.seed
	learning_rate =results.lr
	v_prior =results.v_prior
	batch_size =results.batch_size
	epochs =results.epochs
	K =results.K
	hidden_size =results.hidden_size
	offset=results.offset
	init_scale=results.init_scale

	main(dataset, alpha,learning_rate, v_prior,batch_size,epochs,K,hidden_size,offset,init_scale,seed)
