from tensorflow.keras import callbacks as cb
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import initializers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Add, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers as opt
from tensorflow.keras.regularizers import l2
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
import os

import CostFunctions as cf
from Util import FileIO as io

class Sample:
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

def _log_epoch(label, epoch, logs):
    log_every = int(os.getenv("DEEPCYTOF_TRAIN_LOG_EVERY", "5"))
    if log_every <= 0:
        return
    if (epoch + 1) % log_every != 0:
        return
    loss = None if logs is None else logs.get("loss")
    if loss is None:
        print(f"[{label}] epoch {epoch + 1}", flush=True)
    else:
        print(f"[{label}] epoch {epoch + 1} loss={loss:.4f}", flush=True)

def step_decay(epoch):
    initial_lrate = 1e-5
    drop = 0.1
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def constructMMD(target):
    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2
    init = initializers.RandomNormal(mean=0.0, stddev=1e-4)
    space_dim = target.X.shape[1]
    
    calibInput = Input(shape=(space_dim,))
    
    # Block 1
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block1_a1) 
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block1_a2) 
    block1_output = Add()([block1_w2, calibInput])
    
    # Block 2
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block2_a1) 
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block2_a2) 
    block2_output = Add()([block2_w2, block1_output])
    
    # Block 3
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block3_a1) 
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block3_a2) 
    block3_output = Add()([block3_w2, block2_output])
    
    calibMMDNet = Model(inputs=calibInput, outputs=block3_output)
    return calibMMDNet, block3_output

def calibrate(target, source, sourceIndex, predLabel, path):
    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2
    init = initializers.RandomNormal(mean=0.0, stddev=1e-4)
    space_dim = target.X.shape[1]
    
    calibInput = Input(shape=(space_dim,))
    
    # Block 1
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block1_a1) 
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block1_a2) 
    block1_output = Add()([block1_w2, calibInput])

    # Block 2
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block2_a1) 
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block2_a2) 
    block2_output = Add()([block2_w2, block1_output])

    # Block 3
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block3_a1) 
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block3_a2) 
    block3_output = Add()([block3_w2, block2_output])
    
    calibMMDNet = Model(inputs=calibInput, outputs=block3_output)

    # Subsampling with Index Fix (.ravel())
    n_target = target.X.shape[0]
    p_target = np.random.permutation(n_target)
    mmd_subsample = float(os.getenv("DEEPCYTOF_MMD_SUBSAMPLE", "0.05"))
    n_target_take = min(n_target, max(1, int(mmd_subsample * n_target)))
    toTake_target = p_target[range(n_target_take)] 
    targetXMMD = target.X[toTake_target]
    targetYMMD = target.y[toTake_target]
    
    # Filtering rows where label is not 0 using .ravel() to ensure 1D mask
    mask_target = (targetYMMD != 0).ravel()
    targetXMMD = targetXMMD[mask_target]
    targetYMMD = targetYMMD[mask_target]
    targetYMMD = np.reshape(targetYMMD, (-1, 1))

    n_source = source.X.shape[0]
    p_source = np.random.permutation(n_source)
    n_source_take = min(n_source, max(1, int(mmd_subsample * n_source)))
    toTake_source = p_source[range(n_source_take)] 
    sourceXMMD = source.X[toTake_source]
    sourceYMMD = predLabel[toTake_source]
    
    # Filtering rows for source using .ravel()
    mask_source = (sourceYMMD != 0).ravel()
    sourceXMMD = sourceXMMD[mask_source]
    sourceYMMD = sourceYMMD[mask_source]
    sourceYMMD = np.reshape(sourceYMMD, (-1, 1))

    lrate = LearningRateScheduler(step_decay)
    optimizer = opt.RMSprop(learning_rate=0.0)
    
    calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true, y_pred: 
       cf.MMD(block3_output, targetXMMD, 
            MMDTargetValidation_split=0.1).KerasCost(y_true, y_pred))

    sourceLabels = np.zeros(sourceXMMD.shape[0])

    mmd_epochs = int(os.getenv("DEEPCYTOF_MMD_EPOCHS", "20"))
    mmd_batch_size = int(os.getenv("DEEPCYTOF_MMD_BATCH_SIZE", "20000"))
    mmd_verbose = int(os.getenv("DEEPCYTOF_MMD_VERBOSE", "0"))
    # Fit the calibration network
    calibMMDNet.fit(sourceXMMD, sourceLabels, epochs=mmd_epochs,
            batch_size=mmd_batch_size, validation_split=0.0, verbose=mmd_verbose,
            callbacks=[lrate, cb.LambdaCallback(
                on_epoch_end=lambda epoch, logs: _log_epoch("mmd", epoch, logs)
            )])
    
    plt.close('all')
    
    save_dir = os.path.join(io.DeepLearningRoot(), 'savemodels', path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    calibMMDNet.save_weights(os.path.join(save_dir, 'ResNet'+ str(sourceIndex)+'.h5'))
    
    calibrateSource = Sample(calibMMDNet.predict(source.X), source.y)
    calibMMDNet = None
    return calibrateSource

def loadModel(target, source, sourceIndex, predLabel, path):
    mmdNetLayerSizes = [25, 25]
    l2_penalty = 1e-2
    init = initializers.RandomNormal(mean=0.0, stddev=1e-4)
    space_dim = target.X.shape[1]
    
    calibInput = Input(shape=(space_dim,))
    
    # Block 1
    block1_bn1 = BatchNormalization()(calibInput)
    block1_a1 = Activation('relu')(block1_bn1)
    block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block1_a1) 
    block1_bn2 = BatchNormalization()(block1_w1)
    block1_a2 = Activation('relu')(block1_bn2)
    block1_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block1_a2) 
    block1_output = Add()([block1_w2, calibInput])

    # Block 2
    block2_bn1 = BatchNormalization()(block1_output)
    block2_a1 = Activation('relu')(block2_bn1)
    block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block2_a1) 
    block2_bn2 = BatchNormalization()(block2_w1)
    block2_a2 = Activation('relu')(block2_bn2)
    block2_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block2_a2) 
    block2_output = Add()([block2_w2, block1_output])

    # Block 3
    block3_bn1 = BatchNormalization()(block2_output)
    block3_a1 = Activation('relu')(block3_bn1)
    block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block3_a1) 
    block3_bn2 = BatchNormalization()(block3_w1)
    block3_a2 = Activation('relu')(block3_bn2)
    block3_w2 = Dense(space_dim, activation='linear',
                      kernel_regularizer=l2(l2_penalty), kernel_initializer=init)(block3_a2) 
    block3_output = Add()([block3_w2, block2_output])
    
    calibMMDNet = Model(inputs=calibInput, outputs=block3_output)
    calibMMDNet.load_weights(os.path.join(io.DeepLearningRoot(),
                                          'savemodels', path, 'ResNet'+ str(sourceIndex)+'.h5'))
    
    return calibMMDNet
