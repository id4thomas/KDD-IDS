import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import sin,cos,sqrt

#Simple AE Model
DATA_DIM = 114
LATENT_DIM = 32
MODEL_NAME="ae"

class AE():
    def __init__(self):
        #Learning Rate
        self.lr=1e-3
        #Model
        self.encoder,self.decoder=self.make_model()
        #Optimizer
        self.ae_op=tf.keras.optimizers.Adam(lr=self.lr)
        #Trainables
        self.trainables=self.encoder.trainable_weights+self.decoder.trainable_weights

    def make_model(self):
        #119 -> 128 -> 64 -> 32 -> 64 -> 128 -> 119
        #Encoder
        e_in=tf.keras.layers.Input(shape=(DATA_DIM,))
        e1=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(e_in)
        e2=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(e1)
        z=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e2)
        encoder=keras.models.Model(inputs=e_in,outputs=z)
        encoder.summary()
        
        #Decoder
        d_in=tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1=tf.keras.layers.Dense(64, activation='relu',kernel_initializer='he_uniform')(d_in)
        d2=tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_uniform')(d1)
        out=tf.keras.layers.Dense(DATA_DIM, activation='sigmoid',kernel_initializer='he_uniform')(d2)
        decoder=keras.models.Model(inputs=d_in,outputs=out)
        decoder.summary()

        return encoder,decoder  

    def recon(self,x):
        encoded=self.encoder(x)
        pred=self.decoder(encoded)
        return pred

    def calc_loss(self,x):
        out=self.recon(x)

        #Reconstruction Loss
        #Categorical Cross Entropy
        # recon_loss=tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, out))*DATA_DIM
        #MSE
        recon_loss=tf.reduce_mean(tf.square(x-out))
        return recon_loss

    def update(self,grad):
        self.ae_op.apply_gradients(zip(grad,self.trainables))

    def save_model(self,save_path,epoch,batch):
        self.encoder.save(save_path+'/encoder{}_{}.h5'.format(epoch,batch))
        self.decoder.save(save_path+'/decoder{}_{}.h5'.format(epoch,batch))
