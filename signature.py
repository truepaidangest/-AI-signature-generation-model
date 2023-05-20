####read and seperate the dataset for the model####
import pandas as pd
# Load data
X="file_name"
Y="forged"
train_df=pd.read_csv("/content/sign_data/train_data.csv",header=None,usecols=[1,2],names=[X,Y],dtype=str)
test_df=pd.read_csv("/content/sign_data/test_data.csv",header=None,usecols=[1,2],names=[X,Y],dtype=str)

# Train Test Split
from sklearn.model_selection import train_test_split
validation_df,test_df=train_test_split(test_df,test_size=0.5)
print(train_df.shape,validation_df.shape,test_df.shape)

display(train_df)
# See how many genuine/forged signitures in train data?
train_df.forged.value_counts()

# Define the data path, we will use it later.
path="/content/sign_data/"

# The mode of image
from PIL import Image
Image.open("/content/sign_data/train/001/001_01.PNG").mode



####generate new dataset by using ImageDataGenerator####
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Instantiate ImageDataGenerator class
train_datagen = ImageDataGenerator(rescale=1./255)

# Configure train data generator
train_gen=train_datagen.flow_from_dataframe( 
    train_df,                                   # Pandas dataframe containing the filepaths relative to directory (or absolute paths if directory is None) of the images in a string column.
    directory="/content/sign_data/train",       # string, path to the directory to read images from. If None, data in x_col column should be absolute paths.
    x_col=X,                          # string, column in dataframe that contains the filenames (or absolute paths if directory is None).
    y_col=Y,                             # string or list, column/s in dataframe that has the target data.
    target_size=(128, 256),                     # tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized. 
    color_mode="rgb",                           # one of "grayscale", "rgb", "rgba". Default: "rgb". Whether the images will be converted to have 1 or 3 color channels.
    class_mode="binary",                        # one of "binary", "categorical", "input", "multi_output", "raw", sparse" or None. Default: "categorical". If y_col is a binary class (0 or 1), you should use "binary". If y_col is numeric value array, you should use "raw".
    batch_size=32,   # size of the batches of data (default: 32). This batchsize is different from the one in model.fit(). It is just another batchsize served for converting original images to executable format for neural network.
    shuffle=True)                               # whether to shuffle the data (default: True). This is for eliminating bias in generating input data

# Configure validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen=val_datagen.flow_from_dataframe(
    validation_df,
    directory="/content/sign_data/train",
    x_col=X,
    y_col=Y,
    target_size=(128, 256),
    color_mode="rgb",
    class_mode="binary",
    ) #In the generation of validation and test dataset, its not necessary to define batchsize and shuffle. Because only training dataset is taken for training.

# Configure test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen=test_datagen.flow_from_dataframe(
    test_df,
    directory="/content/sign_data/test",
    x_col=X,
    y_col=Y,
    target_size=(128, 256),
    color_mode="rgb",
    class_mode="binary",
    )


####use dense neural networks for predict the forged signatures####
# Build sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense

model=Sequential()
model.add(Flatten())   # Flatten the channels into one single long array
model.add(Dense(2048 ,input_shape = (128*256*3,) ,activation = 'relu'))     
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss="binary_crossentropy",                   #computes the cross-entropy loss between true labels and predicted labels
              optimizer="adam",
              metrics=['acc'])

history=model.fit(train_gen,                                # Train data generator
      steps_per_epoch=train_gen.n//train_gen.batch_size,    #train_gen.n//train_gen.batch_size. In lab 2, we set "batchsize" in model.fit() to determine the step length for steps in one epoch. 
                  #Steps_per_epoch is the number of steps in one epoch. It can also determine the step length (batchszie) in each epoch. So, we can set steps_per_epoch instead to replace "batchsize".
                  #Either batchsize or steps_per_epoch is fine. Just make sure you set one of them.
      epochs=10,                                             
      validation_data=val_gen,                              # Validation data generator                      
      validation_steps=val_gen.n//val_gen.batch_size)       # val_gen.n//val_gen.batch_size, Need to sepcify this parameter.        "//" means get the integer part from division result.



####use CNNs to predict the forged signatures with high accuracy####
##Larger Convolution Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback 

# Build sequential model
model=Sequential()
model.add(Conv2D(128, 3,input_shape=(128,256,3),activation="relu"))
model.add(MaxPooling2D(2))
model.add(Conv2D(64, 3,activation="relu"))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dropout(0.5))                                                         # Add Dropout layer, randomly drops 50% neurons from propagation.
model.add(Dense(64,activation="relu", kernel_regularizer="l2"))
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['acc'])

# Configure EarlyStopping object
Es=EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)      # EarlyStopping with patience 2: if in 2 consecutive epochs, the validation loss didn't hit the record, training will stop.

# Fit the model and record the training history for plotting purpose
history=model.fit(train_gen,
      steps_per_epoch=train_gen.n//train_gen.batch_size, #train_gen.n//train_gen.batch_size. In lab 2, we set "batchsize" in model.fit() to determine the step length for steps in one epoch. 
                  #Steps_per_epoch is the number of steps in one epoch. It can also determine the step length (batchszie) in each epoch. So, we can set steps_per_epoch instead to replace "batchsize".
                  #Either batchsize or steps_per_epoch is fine. Just make sure you set one of them.
      epochs=10,
      validation_data=val_gen,
      validation_steps=val_gen.n//val_gen.batch_size, #val_gen.n//val_gen.batch_size
      callbacks=[Es])                                                           # Here is how we use EarlyStopping

model.summary()

# Plot history
import matplotlib.pyplot as plt
plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='g')
plt.show()

# Evaluate test data accuracy.
model.evaluate(test_gen)


##Convolution Model
# Final version
# Build sequential model
model=Sequential()
model.add(Conv2D(128, 3,input_shape=(128,256,3),activation="relu"))
model.add(MaxPooling2D(2))
#model.add(Conv2D(64, 3,activation="relu"))
#model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dropout(0.5))                                                         # Add Dropout layer, randomly drops 50% neurons from propagation.
model.add(Dense(64,activation="relu", kernel_regularizer="l2"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['acc'])

# Configure EarlyStopping object
Es=EarlyStopping(monitor='val_loss', patience=2,restore_best_weights=True)      # EarlyStopping with patience 2: if in 2 consecutive epochs, the validation loss didn't hit the record, training will stop.

# Fit the model and record the training history for plotting purpose
history=model.fit(train_gen,
      steps_per_epoch=train_gen.n//train_gen.batch_size, #train_gen.n//train_gen.batch_size
      epochs=10,
      validation_data=val_gen,
      validation_steps=val_gen.n//val_gen.batch_size, #val_gen.n//val_gen.batch_size
      callbacks=[Es])                                                           # Here is how we use EarlyStopping

# Plot history
import matplotlib.pyplot as plt
plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='g')
plt.show()

##Basic Convolution Model
# Intermediate version
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, Callback

model=Sequential()                                                      
model.add(Conv2D(128, 3,input_shape=(128, 256,3),activation="relu"))     # The first convolution layer: 16 filters with size 3x3.
#model.add(MaxPooling2D(2))
#model.add(Conv2D(64, 3,activation="relu"))
model.add(MaxPooling2D(2))                                              # Pooling: MaxPooling
model.add(Flatten())                                                    # Flatten the output of pooling layer i.e. create a long array
#model.add(Dropout(0.5))                                                    
model.add(Dense(64,activation="relu")) # No L2 regularizer here
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",                               # This is binary classification
              optimizer="adam",                                         # Adam optimizer
              metrics=['acc'])                                          # Monitoring the accuracy

# Fit the model and record the training history for plotting purpose
history=model.fit(train_gen,
      steps_per_epoch=train_gen.n//train_gen.batch_size, #train_gen.n//train_gen.batch_size
      epochs=10,
      validation_data=val_gen,
      validation_steps=val_gen.n//val_gen.batch_size) #val_gen.n//val_gen.batch_size

model.summary()

# Plot history
import matplotlib.pyplot as plt
plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='g')
plt.show()

model.evaluate(test_gen)

model.save('/content/modelsign.h5')