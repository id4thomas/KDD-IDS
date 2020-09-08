import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

ATK=1
SAFE=0

def plot2d(data,label,idx=[0,1],atk_front=False):
    fig=plt.figure()
    plt2d=fig.add_subplot(1,1,1)
    if atk_front:
        #safe
        s = plt2d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]], marker='x', color='y')
        #atk
        a = plt2d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]], marker='o', color='b')
    else:
        #atk
        a = plt2d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]], marker='o', color='b')
        #safe
        s = plt2d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]], marker='x', color='y')

    plt2d.legend((s,a),('safe','attack'))
    return fig

def plot3d(data,label,idx=[0,1,2],atk_front=False):
    fig=plt.figure()
    plt3d=fig.add_subplot(1,1,1,projection='3d')

    if atk_front:
        #safe
        s=plt3d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]],data[label==SAFE,idx[2]], marker='x', color='y')
        #atk
        a=plt3d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]],data[label==ATK,idx[2]], marker='o', color='b')
    else:
        #safe
        s=plt3d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]],data[label==SAFE,idx[2]], marker='x', color='y')
        #atk
        a=plt3d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]],data[label==ATK,idx[2]], marker='o', color='b')
    plt3d.legend((s,a),('safe','attack'))
    return fig

def plot_losses(train_l,val_l,title):
    fig=plt.figure()
    l_plot=fig.add_subplot(1,1,1)
    l_plot.plot(range(len(train_l)),train_l,c='r',label='train')
    l_plot.plot(range(len(val_l)),val_l,c='b',label='val')
    l_plot.set_title(title)
    return fig
