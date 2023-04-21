from __future__ import print_function
from pathlib import Path
import shutil
import urllib.request

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import torch

from patchify import patchify

import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

class segmentation_task:
    
    """ 
    train_set:   list
                 list of pictures that needs to be labeled
                 
    patch_size:  int
                 width/height size of the patch
    

    """
    
    def __init__(self, train_set, patch_size):
        
        self.train_set = train_set
        self.patch_size = patch_size
        
        
        self.width, self.height = train_set[0].size
        self.patch_nwidth = int(self.width/self.patch_size*2-1)
        self.patch_nheight = int(self.height/self.patch_size*2-1)
        self.n_samples = len(self.train_set)
        
        dataset = self.img_to_dataset(train_set)
        df = pd.DataFrame(dataset)
        df.columns = df.columns.astype(str)
        testdata = bnpy.data.XData.from_dataframe(df[: self.n_samples*self.patch_nwidth*self.patch_nheight])
        
        K = 16          # n clusters
        gamma = 10  # DP concentration param
        sF = 0.1       # scale of expected covariance
                       ## Try emperical covariance

        full_trained_model, full_info_dict = bnpy.run(
            testdata, 
            'DPMixtureModel', 'ZeroMeanGauss', 'memoVB',
            #output_path='/tmp/faithful/demo_sparse_resp-K=3-lik=Gauss-ECovMat=5*eye/',
            output_path = '/Users/inii/CS_141/Final Project/segmentation_test/test_result',
            nLap=350, nTask=1, 
            nBatch=1, 
            #convergeThr=0.001,
            gamma0=gamma, 
            sF=sF, ECovMat='eye',
            K=K,
            moves='merge,shuffle',
            #nnzPerRowLP = 2,
            #initname='test',
     
            )
        
        LP_trained = full_trained_model.calc_local_params(testdata)
        testls = LP_trained['resp']
        
        frequency_df = self.patches_labels_by_frequency(testls)
        self.patches_labels_on_img(testls)
        
        
    def img_to_dataset(self, imgset):

        grayscale = torchvision.transforms.Grayscale(1)
        dataset = []
        L = len(imgset)
       
        for img in range(L):
            gray_sample = np.array(grayscale(imgset[img]).getdata()).reshape(self.width, self.height)
            patches = patchify(gray_sample, (self.patch_size, self.patch_size), int(self.patch_size/2))
            for i in range(self.patch_nwidth):
                for j in range(self.patch_nheight):
                    data = patches[i][j].flatten()
                    data = data - np.mean(data)
                    dataset.append(data)
        return dataset
        
    def generate_pixel_coordinates(self):
        x = []
        y = []
        for i in range(1,self.patch_nheight+1):
            x = np.append(x, np.asarray(range(1, self.patch_nwidth+1))*(self.patch_size/2))
            yls = self.height - i*(self.patch_size/2)
            y = np.append(y, np.ones(self.patch_nwidth)*yls)
        return x, y
   

    def patches_labels_on_img(self, resp):
    
        n = int(len(resp)/self.patch_nwidth/self.patch_nheight)
        ## Draw random sampples to present
        np.random.seed(12345)
        S = np.random.choice(10, size = (2, 3), replace = False)
    
        x, y = self.generate_pixel_coordinates()
        labels = resp.argmax(axis = 1)
    
        fig, axs = plt.subplots(4, 3, figsize = (10,10))
        #plt.subplots_adjust(hspace=0)
        fig.tight_layout()
    
    
        for i in range(2):
            for j in range(3):
                s = S[i, j]
                start = s*self.patch_nwidth*self.patch_nheight
                end = start + self.patch_nwidth*self.patch_nheight
                label_s = labels[start:end]
                pltdf = pd.DataFrame(dict(x=x, y=y, label=label_s))
                groups = pltdf.groupby('label')
                # Plot
                img = self.train_set[s]
                #fig, ax = plt.subplots()
                axs[i*2, j].margins(0.02)
                axs[i*2, j].imshow(img, extent=[0, self.width, 0, self.height])
                #axs[].subplots_adjust(hspace=0)
                axs[i*2, j].axis('off')
                for name, group in groups:
                    axs[i*2, j].plot(group.x, group.y, marker='s', linestyle='',alpha = 0.07, ms=self.patch_size, label=name)
            
                axs[i*2+1, j].margins(0.02)
                axs[i*2+1, j].imshow(img, extent=[0, self.width, 0, self.height])
                axs[i*2+1, j].axis('off')
            axs[i*2, j].legend()
    
        plt.show()
    
        #fig1, axs1 = plt.subplots(2, 3, figsize = (10,10))
        #for i in range(2):
        #    for j in range(3):
        #        s = S[i, j]
        #        img = train_set[s][0]
        #        axs1[i, j].margins(0.02)
        #        axs1[i, j].imshow(img, extent=[0, 256, 0, 256]) 
        #        axs1[i, j].axis('off')
        #plt.show()
        
    def patches_labels_by_frequency(self, resp):
    
        N = int(len(resp)/self.patch_nwidth/self.patch_nheight)
        K = len(resp[0])
    
        col_name = []
        for k in range(K):
            col_name.append('Cluster'+' '+str(k))

        df = pd.DataFrame(index = range(1, N+1),columns=col_name)
    
        for n in range(N):
            frequency = np.sum(resp[n*self.patch_nwidth*self.patch_nheight: (n+1)*self.patch_nwidth*self.patch_nheight], axis =
                               0)/self.patch_nwidth/self.patch_nheight
            df.iloc[n] = frequency
        
        return df