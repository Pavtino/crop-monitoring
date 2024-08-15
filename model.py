# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:17:32 2024

@author: Martin Ange Mbalkam
this file containt:
    CNN class model
    LSTM class model
    GAN class model
"""
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Lambda, Input, Dense
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.layers import LSTM,Dense,Embedding, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
'''
CNN class Model
'''
class CNNModel():    
    
    def __init__(self,X,y,epochs):
        super().__init__()
        model = Sequential()

        model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten()) #to convert 3D feature map to 1D
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X, y, batch_size=32, epochs=epochs, validation_split=0.3)
        #save model

       # model.evaluate(X_test,y_test)
"""
LSTM Class Model
"""    
class LSTMModel():
     
     def __init__(self,X,Y,embedding_dim,hidden_dim,look_back,epochs=20):
        super().__init__()
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=epochs, batch_size=1, verbose=2)
        #save model  
        
        
        
"""
GAN Class Model
It generate image and nitrogen
"""
class GANModel():
    
    def __init__(self,data,image_size):
        super().__init__()
        self.image_size=image_size
        
        
    def build_vae(self,intermediate_dim=512, latent_dim=2):
        """Build VAE
          :param intermediate_dim: size of hidden layers of the
          encoder/decoder
          :param latent_dim: latent space size
          :returns tuple: the encoder, the decoder, and the full vae
        """
        # encoder first
        inputs = Input(shape=(self.image_size,), name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        # latent mean and variance
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        # reparameterization trick for random sampling
        # Note the use of the Lambda layer
        # At runtime, it will call the sampling function
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        # full encoder encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.image_size, activation='sigmoid')(x)
        # full decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        # VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae')
        # Loss function
        # we start with the reconstruction loss
        reconstruction_loss = binary_crossentropy(inputs, outputs) * self.image_size
        # next is the KL divergence

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # we combine them in a total loss
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        return encoder, decoder, vae
    
    
    def sampling(args: tuple):
        """
        Reparameterization trick by sampling z from unit Gaussian
        :param args: (tensor, tensor) mean and log of variance of
        q(z|x)
        :returns tensor: sampled latent vector z
        """
        # unpack the input tuple
        z_mean, z_log_var = args
        # mini-batch size
        mb_size = K.shape(z_mean)[0]
        # latent space size
        dim = K.int_shape(z_mean)[1]
        # random normal vector with mean=0 and std=1.0
        epsilon = K.random_normal(shape=(mb_size, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def generateImage(self,x_train,x_test):
        encoder, decoder, vae =self.build_vae()
        vae.compile(optimizer='adam')
        vae.summary()
        vae.fit(x_train,epochs=50,batch_size=128,validation_data=(x_test, None))
        
        
#LSTM-CNN
class LSTMCNNModel():
      def __init__(self,data,X_train,y_train,X_test,y_test,epochs=20):
            super().__init__()
            model = Sequential()
            model.add(Embedding(20000,32, input_length=100))
            model.add(Conv2D(256, 3, activation='relu', input_shape=(178, 1), padding='same'))
            model.add(MaxPooling2D(2))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, 3, activation='relu', padding='same'))
            model.add(MaxPooling2D(2))
            model.add(Dropout(0.2))
            model.add(LSTM(64, return_sequences=True))
            model.add(LSTM(32))
            model.add(Flatten())
            model.add(Dense(250, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=epochs, batch_size=128, verbose=2)
            result = model.evaluate(X_test,y_test)
            self.model=model
