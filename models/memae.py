import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import sin,cos,sqrt

DATA_DIM = 114
LATENT_DIM = 3
N = 50

MODEL_NAME="memae"

ATK=1
SAFE=0

class MEMAE():
    def __init__(self):
        #Learning Rate
        self.lr=1e-4
        #Model
        self.initializer=tf.initializers.glorot_normal()

        self.encoder,self.decoder=self.make_model()
        self.mem=self.make_memory()

        #Optimizer
        self.op=tf.keras.optimizers.Adam(lr=self.lr)
        # self.op=tf.keras.optimizers.RMSprop(lr=self.lr)

        #Trainables
        self.trainables=self.encoder.trainable_weights+self.decoder.trainable_weights+[self.mem]


    def make_model(self):
        #119 -> 128 -> 64 -> 32 -> 64 -> 128 -> 119
        leak_rate=0.01
        #Encoder
        e_in=tf.keras.layers.Input(shape=(DATA_DIM,))
        e1=tf.keras.layers.Dense(120,kernel_initializer='he_uniform')(e_in)
        e1=tf.keras.layers.BatchNormalization()(e1)
        # e1=tf.keras.layers.ReLU()(e1)
        # e1=tf.keras.layers.LeakyReLU(alpha=leak_rate)(e1)
        e1=tf.keras.activations.tanh(e1)


        e2=tf.keras.layers.Dense(60,kernel_initializer='he_uniform')(e1)
        e2=tf.keras.layers.BatchNormalization()(e2)
        # e2=tf.keras.layers.ReLU()(e2)
        # e2=tf.keras.layers.LeakyReLU(alpha=leak_rate)(e2)
        e2=tf.keras.activations.tanh(e2)

        e3=tf.keras.layers.Dense(30,kernel_initializer='he_uniform')(e2)
        e3=tf.keras.layers.BatchNormalization()(e3)
        # e3=tf.keras.layers.ReLU()(e3)
        # e3=tf.keras.layers.LeakyReLU(alpha=leak_rate)(e3)
        e3=tf.keras.activations.tanh(e3)

        e4=tf.keras.layers.Dense(10,kernel_initializer='he_uniform')(e3)
        # e4=tf.keras.layers.ReLU()(e4)
        # e4=tf.keras.layers.LeakyReLU(alpha=0.1)(e4)
        e4=tf.keras.activations.tanh(e4)

        z=tf.keras.layers.Dense(LATENT_DIM,kernel_initializer='he_uniform')(e4)
        encoder=keras.models.Model(inputs=e_in,outputs=z)
        encoder.summary()

        #Decoder
        d_in=tf.keras.layers.Input(shape=(LATENT_DIM,))
        d1=tf.keras.layers.Dense(10,kernel_initializer='he_uniform')(d_in)
        d1=tf.keras.layers.BatchNormalization()(d1)
        # d1=tf.keras.layers.ReLU()(d1)
        # d1=tf.keras.layers.LeakyReLU(alpha=leak_rate)(d1)
        d1=tf.keras.activations.tanh(d1)

        d2=tf.keras.layers.Dense(30,kernel_initializer='he_uniform')(d1)
        d2=tf.keras.layers.BatchNormalization()(d2)
        # d2=tf.keras.layers.ReLU()(d2)
        # d2=tf.keras.layers.LeakyReLU(alpha=leak_rate)(d2)
        d2=tf.keras.activations.tanh(d2)

        d3=tf.keras.layers.Dense(60,kernel_initializer='he_uniform')(d2)
        d3=tf.keras.layers.BatchNormalization()(d3)
        # d3=tf.keras.layers.ReLU()(d3)
        # d3=tf.keras.layers.LeakyReLU(alpha=leak_rate)(d3)
        d3=tf.keras.activations.tanh(d3)

        d4=tf.keras.layers.Dense(120,kernel_initializer='he_uniform')(d3)
        # d4=tf.keras.layers.ReLU()(d4)
        # d4=tf.keras.layers.LeakyReLU(alpha=0.1)(d4)
        d4=tf.keras.activations.tanh(d4)

        # out=tf.keras.layers.Dense(DATA_DIM, activation='sigmoid',kernel_initializer='he_uniform')(d3)
        out=tf.keras.layers.Dense(DATA_DIM,kernel_initializer='he_uniform')(d4)
        decoder=keras.models.Model(inputs=d_in,outputs=out)
        decoder.summary()

        return encoder,decoder

    def make_memory(self):
        #Code From https://github.com/YeongHyeon/MemAE-TF2/blob/master/source/neuralnet.py
        w = tf.Variable(self.initializer([N,LATENT_DIM]), name="mem_w", trainable=True, dtype=tf.float32)
        return w

    def get_cossim(self,z):
        num=tf.linalg.matmul(z, tf.transpose(self.mem, perm=[1,0]))
        denom=tf.linalg.matmul(z**2, tf.transpose(self.mem, perm=[1,0])**2)
        w = (num + 1e-12) / (denom + 1e-12)
        return w

    def access_memory(self,z):
        cos_sim=self.get_cossim(z)
        attention=tf.nn.softmax(cos_sim)

        #Hard Shrinkage
        #Shrink Threshold
        lam=1/N
        addr_num = tf.keras.activations.relu(attention - lam) * attention
        addr_denum = tf.abs(attention - lam) + 1e-12
        memory_addr = addr_num / addr_denum

        renorm = tf.clip_by_value(memory_addr, 1e-12, 1-(1e-12))

        z_hat = tf.linalg.matmul(renorm, self.mem)

        return z_hat,renorm

    def step(self,x,train=False):
        z=self.encoder(x)
        z_hat,w_hat=self.access_memory(z)
        x_hat=self.decoder(z_hat)
        return x_hat,w_hat

    def update(self,grad):
        self.op.apply_gradients(zip(grad,self.trainables))

    def calc_loss(self,x):
        out,w_hat=self.step(x)
        recon_loss=tf.reduce_mean(tf.square(out-x))
        mem_etrp = tf.reduce_sum((-w_hat) * tf.math.log(w_hat + 1e-12))
        return recon_loss,mem_etrp

    def save_model(self,save_path,epoch,batch):
        self.encoder.save(save_path+'/encoder{}_{}.h5'.format(epoch,batch))
        self.decoder.save(save_path+'/decoder{}_{}.h5'.format(epoch,batch))
