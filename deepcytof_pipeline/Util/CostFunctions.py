'''
Created on Jul 13, 2016
@author: Kelly P. Stanton
'''
import sys
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
IntType = 'int32'
FloatType = 'float32'

#calculate the squared distance between x and y
def squaredDistance(X,Y):
    # Keras 2.x uses 'axis' instead of 'dim'
    r = K.expand_dims(X, axis=1) 
    s = K.expand_dims(Y, axis=0)
    return K.sum(K.square(r - s), axis=-1)

class MMD:
    def __init__(self,
                 MMDLayer,
                 MMDTargetTrain,
                 MMDTargetValidation_split=0.1,
                 MMDTargetSampleSize=1000,
                 n_neighbors = 20,
                 scales = None,
                 weights = None):
        if scales == None:
            print("setting scales using KNN", flush=True)
            med = np.zeros(20)
            for ii in range(1,20):
                sample = MMDTargetTrain[np.random.randint(MMDTargetTrain.shape[0], size=MMDTargetSampleSize),:]
                nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
                distances,dummy = nbrs.kneighbors(sample)
                med[ii]=np.median(distances[:,1:n_neighbors])
            med = np.median(med)  
            scales = [med/2, med, med*2]
            print(scales, flush=True)
        
        scales = K.variable(value=np.asarray(scales))
        if weights == None:
            print("setting all scale weights to 1", flush=True)
            weights = np.ones(K.eval(K.shape(scales))[0])
        
        weights = K.variable(value=np.asarray(weights))
        self.MMDLayer =  MMDLayer
        
        MMDTargetTrain, MMDTargetValidation = train_test_split(MMDTargetTrain, test_size=MMDTargetValidation_split, random_state=42)
        self.MMDTargetTrain = K.variable(value=MMDTargetTrain)
        self.MMDTargetTrainSize = K.eval(K.shape(self.MMDTargetTrain)[0])
        self.MMDTargetValidation = K.variable(value=MMDTargetValidation)
        self.MMDTargetValidationSize = K.eval(K.shape(self.MMDTargetValidation)[0])
        self.MMDTargetSampleSize = MMDTargetSampleSize
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights
        
    def RaphyKernel(self,X,Y):
        sQdist = K.expand_dims(squaredDistance(X,Y),0) 
        # Ensure dimensions match for broadcasting
        scales_exp = K.expand_dims(K.expand_dims(self.scales,-1),-1)
        weights_exp = K.expand_dims(K.expand_dims(self.weights,-1),-1)
        return K.sum(weights_exp * K.exp(-sQdist / (K.pow(scales_exp, 2))), 0)
    
    def cost(self, source, target):
        xx = self.kernel(source, source)
        xy = self.kernel(source, target)
        yy = self.kernel(target, target)
        MMD_val = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
        # Add a small epsilon to ensure sqrt doesn't hit exactly 0
        return K.sqrt(K.maximum(MMD_val, 1e-10))

    def KerasCost(self, y_true, y_pred):
        # FIX: Changed 'low' to 'minval' and 'high' to 'maxval' for TF 1.15 compatibility
        # Also using K.random_uniform directly with the sample size
        
        sample_indices_train = tf.cast(
            tf.random.uniform(
                shape=[self.MMDTargetSampleSize],
                minval=0,
                maxval=self.MMDTargetTrainSize,
                dtype=tf.float32,
            ),
            IntType,
        )
        
        MMDTargetSampleTrain = K.gather(self.MMDTargetTrain, sample_indices_train)

        sample_indices_val = tf.cast(
            tf.random.uniform(
                shape=[self.MMDTargetSampleSize],
                minval=0,
                maxval=self.MMDTargetValidationSize,
                dtype=tf.float32,
            ),
            IntType,
        )
            
        MMDTargetSampleValidation = K.gather(self.MMDTargetValidation, sample_indices_val)

        MMDtargetSample = K.in_train_phase(MMDTargetSampleTrain, MMDTargetSampleValidation) 
        
        ret = self.cost(self.MMDLayer, MMDtargetSample)
        # Ensure gradients are tracked by involving y_pred/y_true
        return ret + 0*K.sum(y_pred) + 0*K.sum(y_true)
