import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import math

class Agent:
	   
	def __init__(self,input_dim, output_dim,lr, gamma):
		self.input_dim = input_dim 					#No. of state features
		self.output_dim = output_dim				#No. of actions ie., dimensionality of the action space
		self.actions = range(output_dim)  			#List of possible actions
		self.lr = lr  								#Initial learning rate, reduces by 0.99 for every episode of training beyond 100 episodes. 
		self.gamma = gamma							#Discount factor
		self.epsilon = 0.5							#Initial value of epsilon, lower values indicate exploitation over exploration.
		self.memory = {'s':[],'a':[],'r':[],'s1':[],'done':[]}	#Experience replay buffer 
		self.batchsize = 16   						#The paper uses 32. However I found 16 to be marginally faster.
		self.memory_size = 2000						#Maximum number of experiences stored in the replay buffer
		self.primary_model = self._make_model()				#The primary "online" model which is trained to estimate Q value 
		self.target_model = self._make_model()      #The target model which is used to estimate the target Q value
		self.tau = 0.1                              #soft update parameter used to "blend" the weights of the target model and the primary model.                            
		lrate = LearningRateScheduler(self.step_decay)	#Keras callback to decay learning rate 
		self.callback_list = [lrate]
		self.step=0									#Indicates number of training episodes, used in step_decay() to activate decay of learning rate.
	
	def step_decay(self,epoch, lr):
		"""
		This function is used to decay the learning rate of the primary model after it is trained for 100 epsiodes.
		The learning rate reduces by 0.99 for every subsequent iteration.
		Inputs:
		lr - current learning rate of the primary model
		epochs - always equal to 1 as we train the primary model for 1 epoch in every training episode.
		Returns:
		lrate - modified learning rate for the primary model
		"""
		if self.step<110:
			lrate=lr
			return lrate
		drop = 0.99
		epochs_drop = 2.0
		lrate = lr * (drop)
		return lrate	

	def _make_model(self):
		"""
		A simple model with two dense layers of size 64 and an output layer of size equal to dimensionality of action space.
		The state features are given as input and the output is an estimate of Q values for that state and all possible actions.   
		Returns:
		model - the Q value estimator.
		"""
		model = Sequential()
		model.add( Dense(64, input_shape = (self.input_dim,), activation='relu') )
		model.add( Dense(64, activation='relu') )
		model.add( Dense( self.output_dim, activation = 'linear' ))
		model.compile(loss='mse',optimizer=Adam(lr = self.lr))
 
		return model
				  
				  
	def remember(self, exp):
		"""
		Used to store (s,a,r,s) tuples in the replay buffer.
		If the the size of the buffer exceeds the maximum possible size, 
		the first tuple is popped out to make space for the input experience.
		Inputs:
		exp - A dictionary containing one (state, action, reward , next state , done ) experience.
		done indicates if the next state is the terminal state.
		"""
		if len(self.memory['s'])>self.memory_size:
			for key in self.memory.keys():
				self.memory[key].pop(0)
		for key,value in exp.items():
			self.memory[key].append(value)

				  

	def act(self, state):
		""" 
		Using the epsilon greedy algorithm. 
		Given a state, the agent picks a random action with probability epsilon,
		and picks the action which gives the highest Q value with probability 1 - epsilon. 
		"""
		
		if np.random.rand() <= self.epsilon:
			action = np.random.choice(self.actions)			#Choose random action with probability epsilon
		else:
			Qs = self.primary_model.predict(state)[0]   	#Keras returns a tensor (2d array), so take first element
			action = np.argmax(Qs)
				  
		return action
				  
			  
	
	def replay(self):
		""" 
		Experience replay is used and a batch of states from the replay buffer 
		is sampled for training the primary network. 
		The target Q value is found using the target network, with action selection being done by the primary model.
		This reduces maximisation bias.
			
		Q_target(state,action) = reward + gamma*Q_target(next_state,argmax(Q_primary(action)), for non-terminal next_state
							   = reward                                             	  , for terminal next_state
		The primary model is then trained for 1 epoch.
		"""
		if len(self.memory['s']) < self.batchsize:
			ids = np.arange(len(self.memory['s']))								#if not enough experiences in replay buffer use all of them
		else:
			ids=np.random.randint(len(self.memory['s']),size=self.batchsize)	#grab random batch
		states = np.asarray([self.memory['s'][i] for i in ids])					#Access the corresponding (s,a,r,s,done) tuple
		actions = np.asarray([self.memory['a'][i] for i in ids])
		rewards = np.asarray([self.memory['r'][i] for i in ids])
		next_states = np.asarray([self.memory['s1'][i] for i in ids])
		dones = np.asarray([self.memory['done'][i] for i in ids])

		
		#Action selection, using the primary model to predict the action which gives maximum Q value given the state .		
		Q_target = self.primary_model.predict(states)
		q_actions_max = np.argmax(self.primary_model.predict(next_states),axis=1)	
		#Target estimation, using the target network to estimate target Q values.
		q_next_state = self.target_model.predict(next_states)
		q_next_state = q_next_state[np.arange(dones.shape[0]),q_actions_max]  
		q_values_req = np.where(dones,rewards,rewards+self.gamma*q_next_state)
		
		#Only the Q values corresponding to the maximum actions are changed.
		Q_target[np.arange(dones.shape[0]),actions] = q_values_req
		#Training the primary model for one epoch.	
		self.primary_model.fit(states,Q_target,verbose=False, epochs=1,callbacks=self.callback_list)
			
		
	def soft_update_target_network(self):
		"""  
		An alternative to hard updating (copying primary network, once every K iterations) the target network.
		Here the target network is updated every timestep,according to 
			 
		theta_target = (1-tau)*theta_target + tau*theta_primary
			 
		where,
			tau = parameter (the smaller, the "softer" the update)
			theta_target = parameters from the target network
			theta_primary = parameters from the primary network
  
		"""
		
		pars_behavior = self.primary_model.get_weights()       
		pars_target = self.target_model.get_weights()  			
		
		ctr = 0
		for par_behavior,par_target in zip(pars_behavior,pars_target):
			par_target = par_target*(1-self.tau) + par_behavior*self.tau
			pars_target[ctr] = par_target
			ctr += 1

		self.target_model.set_weights(pars_target)