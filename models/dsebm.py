import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import sin,cos,sqrt

DATA_DIM = 114

MODEL_NAME="dsebm"

ATK=1
SAFE=0

SEED=2020

class DSEBM():
    def __init__(self):
        self.initializer=tf.initializers.glorot_normal()
        self.b_prime=tf.Variable(self.initializer([DATA_DIM]), name="b_prime", trainable=True, dtype=tf.float32)
        self.make_model()

        self.lr=1e-4
        self.op=tf.keras.optimizers.Adam(lr=self.lr)

    def make_model(self):
        dim1=128
        dim2=512
        # self.x_in=tf.keras.layers.Input(shape=(DATA_DIM,))
        #FC Kernel
        self.kernel1=tf.Variable(self.initializer([DATA_DIM,dim1]), name="k1", trainable=True, dtype=tf.float32)
        self.kernel2=tf.Variable(self.initializer([dim1,dim2]), name="k2", trainable=True, dtype=tf.float32)

        #Bias
        self.b1=tf.Variable(self.initializer([dim1]), name="b1", trainable=True, dtype=tf.float32)
        self.b2=tf.Variable(self.initializer([dim2]), name="b2", trainable=True, dtype=tf.float32)
        #Bias Inverse
        self.b_in1=tf.Variable(self.initializer([DATA_DIM]), name="b_in1", trainable=True, dtype=tf.float32)
        self.b_in2=tf.Variable(self.initializer([dim1]), name="b_in2", trainable=True, dtype=tf.float32)
        # tf.keras.activations.softplus

        self.trainables=[self.kernel1,self.kernel2,self.b1,self.b2,self.b_in1,self.b_in2,self.b_prime]

    def recon(self,x):
        x = tf.nn.softplus(tf.matmul(x, self.kernel1) + self.b1)
        x = tf.nn.softplus(tf.matmul(x, self.kernel2) + self.b2)

        # Inverse shares Params
        x = tf.nn.softplus(tf.matmul(x, tf.transpose(self.kernel2)) + self.b_in2)
        x = tf.nn.softplus(tf.matmul(x, tf.transpose(self.kernel1)) + self.b_in1)
        return x

    def calc_loss(self,x):
        with tf.GradientTape(persistent=True) as t2:
            t2.watch(self.trainables)
            # x=self.net.x_in(x)
            x=tf.Variable(x, dtype=tf.float32)
            # loss=self.calc_loss(x)
            noise=tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1., dtype=tf.float32)
            #Add noise
            x_noise=x+noise

            net_out=self.recon(x)
            net_noise_out=self.recon(x_noise)

            energy = 0.5 * tf.reduce_sum(tf.square(x - self.b_prime)) - tf.reduce_sum(net_out)
            energy_noise = 0.5 * tf.reduce_sum(tf.square(x_noise - self.b_prime)) - tf.reduce_sum(net_noise_out)

        #Recon
        fx=x-t2.gradient(energy , x)
        # fx=tf.squeeze(fx, axis=0)
        fx_noise = x_noise - t2.gradient(energy_noise,x_noise)

        #Reconstruction Loss
        loss=tf.reduce_mean(tf.square(x - fx_noise))
        return loss

    def update(self,grads):
        self.op.apply_gradients(zip(grads,self.trainables))
