import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV2, MobileNet, EfficientNetB0, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator

### Learning rate scheduler for training
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))
    
### Check for available GPUs
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
print(gpu_devices)

### Iterate across training paradigms
for loss in ["mse","mae"]:
    for data_arrangement in ['see']: #["sss", "see", "sse", "eee"]:
    #     for m in [MobileNet, MobileNetV2, MobileNetV3Small, EfficientNetB0, ResNet50]:
        for m in [MobileNetV2]:
            pm = m(include_top=False, input_shape=(224, 224, 3))

            now = datetime.now().strftime(r"%y%m%d_%H%M%S")

            fn = f"models/{now}_{m.__name__}_{loss}_GAv_Dx_D1_{data_arrangement}"

            # Initiate all callbacks
            csv = CSVLogger(fn+".csv")
            mc  = ModelCheckpoint(fn+"_epoch{epoch:03d}.keras", save_best_only=True) #, save_weights_only=True)
            lrs = LearningRateScheduler(scheduler)

            callbacks = [csv, mc, lrs]

            # Create a new deep neural network containing the backbone, e.g. MobileNetV2
            # and then the head --> GAP2D --> Dense(256) --> Dense(1)
            model = Sequential([pm, 
                                GlobalAveragePooling2D(), 
                                Dense(256, activation='relu'), 
                                Dropout(.2),
                                Dense(1, activation='sigmoid')])

            dg_train = DataGenerator("training_224x224_xseg_meta.csv", "training_224x224", "A", batch_size=32,
                                    data_arrangement=data_arrangement, workers=1, use_multiprocessing=False, max_queue_size=1)
            dg_val   = DataGenerator("training_224x224_xseg_meta.csv", "training_224x224", "B", batch_size=32,
                                    data_arrangement=data_arrangement, workers=1, use_multiprocessing=False, max_queue_size=1)

            adam = Adam() 

            # Define optimizer and loss function
            model.compile(adam, loss)

            # Plot data predictions in the naive setting (random weights)
            plt.figure()
            
            for i in range(3):
                X, y = dg_val[i]
                yp = model.predict(X)
                plt.plot(y, yp, 'b.')

            # Train model for 50 epochs using the training data
            model.fit(
                dg_train,
                validation_data=dg_val,
                epochs=50,
                callbacks=callbacks
            )
            
            # Plot data predictions after training (should be better)        
            for i in range(3):
                X, y = dg_val[i]
                yp = model.predict(X)
                plt.plot(y, yp, 'g.')
            
            # Ideal configuration --> y_pred == y
            plt.plot([0, 1], [0, 1], 'k-')
                
            # Save image for further evaluation
            plt.savefig(fn+".png", dpi=300)