# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:07:33 2018

@author: Administrator
"""
from keras.layers import Conv2DTranspose, Reshape, Conv2D, LeakyReLU, Flatten, Activation, Input
from keras.layers import Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mse
import keras.backend as K

def sampling(args):
    """Implements reparameterization trick by sampling
    from a gaussian with zero mean and std=1.
    Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    Returns:
        sampled latent vector (tensor)
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_networks(latent_dim = 20,dim = 32):
    #------------------------------
    # Build encoder
    #------------------------------
    img = Input(shape=(270, 180, 3))
    x = Conv2D(8, kernel_size=7, padding="same")(img)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(8, kernel_size=5, strides = 2, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(16, kernel_size=5, strides = 3, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(32, kernel_size=5, strides = 3, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(64, kernel_size=7, strides = 5, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    x = Dense(dim)(x)
    x=LeakyReLU(alpha=0.1)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = Model(img,[z_mean, z_log_var, z])
    encoder.summary()
    #------------------------------
    # Build decoder
    #------------------------------
    tensor = Input(shape=(latent_dim, ))
    x = Dense(6*64)(tensor)
    x=LeakyReLU(alpha=0.1)(x)
    x = Reshape((3,2,64))(x)
    x = Conv2DTranspose(64, kernel_size=7, strides = 5, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    
    x = Conv2DTranspose(32, kernel_size=5, strides = 3, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    
    x = Conv2DTranspose(16, kernel_size=5, strides = 3, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    
    x = Conv2DTranspose(8, kernel_size=5, strides = 2, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(8, kernel_size=5, padding="same")(x)
    x=LeakyReLU(alpha=0.1)(x)
    x = Conv2D(3, kernel_size=7, padding="same")(x)
    x = Activation('tanh')(x)
    decoder = Model(tensor,x)
    decoder.summary()
    #------------------------------
    # Build CVAE
    #------------------------------
    outputs = decoder(encoder(img)[2])
    CAE = Model(img, outputs, name='cvae')
    
    reconstruction_loss = mse(K.flatten(img), K.flatten(outputs))
    reconstruction_loss *= 270 * 180
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.abs(K.sum(kl_loss, axis=-1))
    kl_loss *= 0.5
    cvae_loss = K.mean(reconstruction_loss + kl_loss)
    CAE.add_loss(cvae_loss)
    optimizer = Adam(lr = 0.001)
    CAE.compile(optimizer=optimizer)
    CAE.summary()
    return encoder, decoder, CAE
def save_network(net,filename):
    net.save(filename)
def load_network(filename):
    from keras.models import load_model
    return load_model(filename)
