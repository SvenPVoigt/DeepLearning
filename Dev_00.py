import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model


(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

plt.figure(figsize=[10,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_data[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_labels[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_data[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[0]))
plt.show()

print('Training data shape : ', train_data.shape, train_labels.shape)

print('Testing data shape : ', test_data.shape, test_labels.shape)

# Find the unique numbers from the train labels
classes = np.unique(train_labels)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

# Create one hot categories
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

input_shape = (32,32,3)
model = createModel()
batch_size = 256
epochs = 5
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


# dimData = np.prod(train_images.shape[1:])
# train_data = train_images.reshape(train_images.shape[0], dimData)
# test_data = test_images.reshape(test_images.shape[0], dimData)
#
# # Change to float datatype
# train_data = train_data.astype('float32')
# test_data = test_data.astype('float32')
#
# # Scale the data to lie between 0 to 1
# train_data /= 255
# test_data /= 255
#
# # Change the labels from integer to categorical data
# train_labels_one_hot = to_categorical(train_labels)
# test_labels_one_hot = to_categorical(test_labels)
#
# # Display the change for category label using one-hot encoding
# print('Original label 0 : ', train_labels[0])
# print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])
#
# from keras.layers import Dropout
#
# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(dimData,)))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nClasses, activation='softmax'))
#
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
#                    validation_data=(test_data, test_labels_one_hot))
#
# [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
# print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

#Plot the Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

# Predict the most likely class
print(model.predict_classes(test_data[[0],:]))

# Predict the probabilities for each class
print(model.predict(test_data[[0],:]))
