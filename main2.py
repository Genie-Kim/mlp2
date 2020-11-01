# train a generative adversarial network on a one-dimensional function
from numpy import hstack
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from matplotlib import pyplot
import tensorflow as tf
import math
import numpy as np
import datetime
from tensorboardX import SummaryWriter
from keras.utils.vis_utils import plot_model
import os
from itertools import product
from keras import optimizers


# define the standalone discriminator model
def define_discriminator(act,num_neuron_D,n_inputs=2):
	model = Sequential()
	# to use LeakyRelu
	if act.split(' ')[0] == 'leakyrelu':
		model.add(Dense(num_neuron_D, kernel_initializer='he_uniform', input_dim=n_inputs))
		model.add(LeakyReLU(alpha = float(act.split(' ')[1])))
	else:
		model.add(Dense(num_neuron_D, activation=act, kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(act, latent_dim, num_neuron_G,n_outputs=2):
	model = Sequential()
	# to use LeakyRelu
	if act.split(' ')[0] == 'leakyrelu':
		model.add(Dense(num_neuron_G, kernel_initializer='he_uniform', input_dim=latent_dim))
		model.add(LeakyReLU(alpha = float(act.split(' ')[1])))
	else:
		model.add(Dense(num_neuron_G, activation=act, kernel_initializer='he_uniform', input_dim=latent_dim))

	model.add(Dense(n_outputs, activation='linear'))
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the discriminator
	model.add(discriminator)
	# compile model
	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
	model.compile(loss='binary_crossentropy', optimizer=adam)
	# model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# generate n real samples with class labels
def generate_real_samples(n):
	# generate inputs in [-pi, pi]
	X1 = 2*math.pi*rand(n) - math.pi
	# generate outputs X^2
	X2 = np.sin(X1)
	# stack arrays
	X1 = X1.reshape(n, 1)
	X2 = X2.reshape(n, 1)
	X = hstack((X1, X2))
	# generate class labels
	y = ones((n, 1))
	return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = zeros((n, 1))
	return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim,writer,log_dir,setting,best_list,val_sample_num=100):
	bestRMSE, _, best_new_metric, _ = best_list
	# prepare real samples
	x_real, y_real = generate_real_samples(val_sample_num)
	# evaluate discriminator on real examples
	_, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(generator, latent_dim, val_sample_num)
	# evaluate discriminator on fake examples
	_, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
	# calculate RMSE From Fake sample and sin function
	RMSE = calculate_rmse_ofsample(x_fake)
	# because RMSE is not good at selecting better generator, I made auxilery metric
	auxilery_metric = (acc_real-0.5)**2 + (acc_fake-0.5)**2
	new_metric = auxilery_metric + RMSE
	# summarize discriminator performance
	print(epoch, ' acc_real:',acc_real, ' acc_fake:',acc_fake, ' now bestRMSE:',bestRMSE,' now RMSE:',RMSE, 'best new metric:',best_new_metric,'new_metric:',new_metric)
	# figure title
	title = str(epoch) + ' epoch_real(R) fake(B)_' + setting + '_RMSE:' + str(RMSE) + '_NewM:' + str(new_metric)

	# use new metric for evaluate generator
	if best_new_metric > new_metric: # best new_metric이 갱신되면 저장.
		draw_scatter(x_real, x_fake, title+' new', log_dir)
	# use RMSE for evaluate generator
	elif bestRMSE > RMSE or (epoch+1) % 200==0 or best_new_metric > new_metric:  # best RMSE가 갱신되거나 특정 epoch에 도달하면 figure 저장.
		draw_scatter(x_real,x_fake,title,log_dir)


	if bestRMSE > RMSE:# best RMSE 갱신
		best_list[0] = RMSE
		best_list[1] = epoch
	if best_new_metric > new_metric:# best metric 갱신
		best_list[2] = new_metric
		best_list[3] = epoch

	writer.add_scalars('discriminator validation acc',{'real':acc_real,'fake':acc_fake},epoch)
	writer.add_scalar('generator validation RMSE', RMSE, epoch)
	writer.add_scalar('generator validation new_metric', new_metric, epoch)

	return best_list

def generate_filename_fromsetting(default_setting):
	act = activations[default_setting['act']]
	ldim = latent_dims[default_setting['ldim']]
	num_neuron_D = num_of_neurons[default_setting['num_neuron_D']]
	num_neuron_G = num_of_neurons[default_setting['num_neuron_G']]

	return '_'.join(['activ='+act, 'latentdim='+str(ldim), 'NumNeuD='+str(num_neuron_D),'NumNeuG='+str(num_neuron_G)])


def calculate_rmse_ofsample(fake_sample):
	return np.sqrt(((fake_sample[:,1] - np.sin(fake_sample[:,0]))**2).mean())



# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim,writer,log_dir,setting, n_epochs=10000, n_batch=256, n_eval=10):
	# determine half the size of one batch, for updating the discriminator
	half_batch = int(n_batch / 2)
	best_list = [1e10,0,1e10,0] # initial [best rmse, best rmse epoch, best new metric, best new metric epoch]
	# manually enumerate epochs
	for i in range(n_epochs):
		# prepare real samples
		x_real, y_real = generate_real_samples(half_batch)
		# prepare fake examples
		x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator
		D_R_loss = d_model.train_on_batch(x_real, y_real,return_dict=True)
		D_F_loss = d_model.train_on_batch(x_fake, y_fake,return_dict=True)
		# prepare points in latent space as input for the generator
		x_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		gan_loss = gan_model.train_on_batch(x_gan, y_gan,return_dict=True)
		# loss가 이미 average 되어 나온다. 따라서 /2를 해줘야 함.
		writer.add_scalars('D G train loss',{'discriminator':(D_R_loss['loss']+D_F_loss['loss'])/2,'generator': gan_loss['loss']}, i)
		writer.add_scalars('discriminator_train_loss',{'real':D_R_loss['loss'], 'fake': D_F_loss['loss']},i)
		writer.add_scalars('discriminator_train_acc',{'real': D_R_loss['accuracy'], 'fake':D_F_loss['accuracy']}, i)

		# evaluate the model every n_eval epochs
		if (i+1) % n_eval == 0:
			best_list = summarize_performance(i, g_model, d_model, latent_dim,writer,log_dir,setting,best_list)
	return best_list

def draw_scatter(x_real,x_fake,title,log_dir):
	# scatter plot real and fake data points
	pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
	pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
	pyplot.title(title, fontdict={'fontsize': 7})
	pyplot.savefig(os.path.join(log_dir, title + '.png'), bbox_inches='tight')
	# pyplot.show()
	pyplot.close()


if __name__ == '__main__':

	# base learning rate = 0.001(equal learning rate of discriminator & generator, beta = (0.9,0.999)
	activations = ['relu', 'sigmoid', 'tanh', 'leakyrelu 0.1','leakyrelu 0.3']  # leakyrelu_alpha format
	latent_dims = [2, 5, 10, 15, 20]
	num_of_neurons = [5, 15, 25, 35, 45]

	exp_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	os.makedirs(exp_dir)
	with open(os.path.join(exp_dir,'RMSE info.txt'), "w") as f:
		f.write('setting,best RMSE,best RMSE epoch, best new metric, best new metric epoch' + '\n')
	# default_setting = {'act':0, 'ldim' : 1, 'num_neuron_D':2, 'num_neuron_G':1}

	# 두개를 교차해서 search 할때는
	templist = [x for x in range(5)]
	item = [templist,templist]
	for index1, index2 in list(product(*item)):
		default_setting = {'act': 2, 'ldim': index1, 'num_neuron_D': 4, 'num_neuron_G': index2}

	# # custom search
	# templist = [(2,2),(4,3),(4,1),(4,2)]
	#
	# for index1, (index2,index3) in list(product(*[[0,1,2,3,4],templist])):
	# 	default_setting = {'act': 2, 'ldim': 3, 'num_neuron_D': 4, 'num_neuron_G': 1}

	# # 한 개의 parameter만 searching.
	# for index in range(5):
	# 	default_setting = {'act': 2, 'ldim': index, 'num_neuron_D': 2, 'num_neuron_G': 2}

		setting = generate_filename_fromsetting(default_setting)
		print('The Setting : ' + setting)

		# 세팅마다 폴더 만들고 그 안에 이미지 및 tensorboard결과 저장.
		log_dir = os.path.join(exp_dir,setting)
		os.mkdir(log_dir)
		writer = SummaryWriter(log_dir)
		with tf.device('/device:GPU:0'):
			# size of the latent space
			latent_dim = latent_dims[default_setting['ldim']]
			# number of neuron of discriminator's hidden layer
			num_neuron_D = num_of_neurons[default_setting['num_neuron_D']]
			# number of neuron of generator's hidden layer
			num_neuron_G = num_of_neurons[default_setting['num_neuron_G']]
			# activation function of hidden layer of both generator & discriminator
			act = activations[default_setting['act']]
			# create the discriminator
			discriminator = define_discriminator(act,num_neuron_D)
			# create the generator
			generator = define_generator(act, latent_dim,num_neuron_G)
			# create the gan
			gan_model = define_gan(generator, discriminator)

			# summarize the model
			discriminator.summary()
			generator.summary()
			gan_model.summary()

			# plot the model
			plot_model(generator, to_file=os.path.join(log_dir,setting + ' generator_plot.png'), show_shapes=True, show_layer_names=True)
			plot_model(discriminator, to_file=os.path.join(log_dir,setting + ' discriminator_plot.png'), show_shapes=True, show_layer_names=True)
			plot_model(gan_model, to_file=os.path.join(log_dir,setting + ' GAN_plot.png'), show_shapes=True, show_layer_names=True)
			# train model
			best_list = train(generator, discriminator, gan_model, latent_dim, writer,log_dir,setting,n_batch=256,n_epochs=7000)

			with open(os.path.join(exp_dir,'RMSE info.txt'), "a") as f:
				f.write(setting + ',' + ','.join([str(x) for x in best_list]) + '\n') # write bestRMSE per setting, csv format.

