from models.ae import AE

import pandas as pd
import numpy as np
import tensorflow as tf

from utils.data_utils import *
from utils.plot_utils import *

from sklearn.metrics import average_precision_score

SEED=2020

class TrainAE:
    def __init__(self):
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        #Get Model
        self.net=AE()

    def train_batch(self,batch):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.net.trainables)
            recon_loss=self.net.calc_loss(batch)
        #Get Gradients
        ae_grads=t.gradient(recon_loss,self.net.trainables)

        #Apply Grads
        self.net.update(ae_grads)

        return recon_loss

    def train(self,data,num_epochs,batch_size):
        #Unpack Data
        x_train=data['x_train']
        y_train=data['y_train']

        x_val=data['x_val']
        y_val=data['y_val']


        train_losses=[]
        val_losses=[]

        batch_print=1000
        print("Start Training For {} Epochs".format(num_epochs))
        for ep in range(num_epochs):
            #Shuffle
            np.random.shuffle(x_train)
            batch_iters=int(x_train.shape[0]/batch_size)

            batch_loss=0
            print("\nEpoch {}".format(ep+1))
            for i in range(batch_iters):
                #run batch
                cur_idx=i*batch_size
                batch=x_train[cur_idx:cur_idx+batch_size]
                # idx = np.random.randint(0, x_train.shape[0], batch_size)
                # batch = x_train[idx]

                train_recon=self.train_batch(batch)
                batch_loss+=train_recon
                #self.net.apply_grads(grads)
                if (i+1)%batch_print==0:
                    print('Batch loss {} recon:{:.5f}'.format(i+1,train_recon))

            #train epoch loss
            ep_loss=batch_loss/batch_iters
            train_losses.append(ep_loss)

            #Val Loss
            val_recon=self.calc_loss(x_val)
            val_losses.append(val_recon)

            print('Epoch recon loss Train:{:.5f} Val:{:.5f}'.format(ep_loss,val_recon))


        recon_plot=plot_losses(train_losses,val_losses,'Recon Loss')
        # recon_plot.set_title('Recon Loss')
        recon_plot.savefig('./plot/ae_recon_loss.png')
        # plt.clf()

if __name__ == "__main__":
    kdd_path='../kdd_data'
    x_train,y_train=get_hdf5_data(kdd_path+'/processed/train.hdf5',labeled=True)
    x_train,x_val,y_train,y_val=split_data(x_train,y_train)
    x_train,y_train=filter_atk(x_train,y_train)

    x_test,y_test=get_hdf5_data(kdd_path+'/processed/test.hdf5',labeled=True)

    print("Train {} Val {} Test {}".format(x_train.shape[0],x_val.shape[0],x_test.shape[0]))

    num_ep=10
    batch_size=128
    trainer=TrainAE()
    data={
        'x_train':x_train,
        'y_train':y_train,
        'x_val':x_val,
        'y_val':y_val
    }

    trainer.train(data,num_ep,batch_size)

    #Evaluate Test Data
    test_recon= trainer.net.recon(x_test)
    test_dist=np.mean(np.square(x_test - test_recon),axis=1)
    print(average_precision_score(y_test, test_dist))
