from tensorflow.keras import callbacks as cb
from tensorflow.keras.layers import Input, Dense, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os

from Util import DataHandler as dh
import os.path
from Util import FileIO as io

class Sample:
    X = None
    y = None
    def __init__(self, X, y = None):
        self.X = X
        self.y = y

class TrainProgress(cb.Callback):
    def __init__(self, label):
        super().__init__()
        self.label = label
        self.log_every = int(os.getenv("DEEPCYTOF_TRAIN_LOG_EVERY", "5"))

    def on_epoch_end(self, epoch, logs=None):
        if self.log_every <= 0:
            return
        if (epoch + 1) % self.log_every == 0 or (epoch + 1) == self.params.get("epochs", epoch + 1):
            loss = None if logs is None else logs.get("loss")
            if loss is None:
                print(f"[{self.label}] epoch {epoch + 1}/{self.params.get('epochs', epoch + 1)}", flush=True)
            else:
                print(f"[{self.label}] epoch {epoch + 1}/{self.params.get('epochs', epoch + 1)} loss={loss:.4f}", flush=True)
        
def trainDAE(target, dataPath, refSampleInd, trainIndex, relevantMarkers, mode,
             keepProb, denoise, loadModel, path):
    # --- SAFE PATCH START ---
    # Ensure trainIndex is a numpy array so .size works
    trainIndex = np.array(trainIndex)
    
    # Initialize as an empty 2D array with the correct number of columns
    # This keeps the 'concatenate' logic identical but prevents the crash
    sourceX = np.empty((0, target.X.shape[1])) 
    # --- SAFE PATCH END ---
    for i in np.arange(trainIndex.size-1):
        sourceIndex = np.delete(trainIndex, refSampleInd)[i]
        source = dh.loadDeepCyTOFData(dataPath, sourceIndex,
                                      relevantMarkers, mode)
        numZerosOK=1
        toKeepS = np.sum((source.X==0), axis = 1) <= numZerosOK
        if i == 0:
            sourceX = source.X[toKeepS]
        else:
            sourceX = np.concatenate([sourceX, source.X[toKeepS]], axis = 0)
        
    # preProcess source
    sourceX = np.log(1 + np.abs(sourceX))
    
    numZerosOK=1
    toKeepT = np.sum((target.X==0), axis = 1) <= numZerosOK
    
    inputDim = target.X.shape[1]
    
    ae_encodingDim = 25
    l2_penalty_ae = 1e-2
    
    if denoise:
        if loadModel:
            from tensorflow.keras.models import load_model
            autoencoder = load_model(os.path.join(dataPath,
                                                  'savemodels', path, 'denoisedAE.h5'))
        else:
            # train de-noising auto encoder and save it.
            trainTarget_ae = np.concatenate([sourceX, target.X[toKeepT]],
                                            axis=0)
            trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb,
                                                size = trainTarget_ae.shape)
        
            input_cell = Input(shape=(inputDim,))
            encoded = Dense(
                ae_encodingDim,
                activation="relu",
                kernel_regularizer=l2(l2_penalty_ae),
            )(input_cell)
            encoded1 = Dense(
                ae_encodingDim,
                activation="relu",
                kernel_regularizer=l2(l2_penalty_ae),
            )(encoded)
            decoded = Dense(
                inputDim,
                activation="linear",
                kernel_regularizer=l2(l2_penalty_ae),
            )(encoded1)
        
            autoencoder = Model(inputs=input_cell, outputs=decoded)
            autoencoder.compile(optimizer='rmsprop', loss='mse')
            dae_epochs = int(os.getenv("DEEPCYTOF_DAE_EPOCHS", "10"))
            dae_batch_size = int(os.getenv("DEEPCYTOF_DAE_BATCH_SIZE", "2048"))
            autoencoder.fit(trainData_ae, trainTarget_ae, epochs=dae_epochs,
                            batch_size=dae_batch_size, shuffle=True,
                            validation_split=0.0, verbose = 0,
                            callbacks=[TrainProgress("dae")])
            # --- SAFE SAVE START ---
            save_folder = os.path.join(dataPath, 'savemodels', path)
            
            # Create the folder if it doesn't exist
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f">>> Created directory: {save_folder}")
            
            # Save the file
            save_file_path = os.path.join(save_folder, 'denoisedAE.h5')
            autoencoder.save(save_file_path)
            print(f">>> Model saved successfully to {save_file_path}")
            # --- SAFE SAVE END ---
            del sourceX
            plt.close('all')
        
        return autoencoder

def predictDAE(target, autoencoder, denoise = False):
    if denoise:    
        # apply de-noising auto encoder to target.
        denoiseTarget = Sample(autoencoder.predict(target.X), target.y)
    else:
        denoiseTarget = Sample(target.X, target.y)
        
    return denoiseTarget
