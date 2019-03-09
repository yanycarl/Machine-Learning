from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import random

Epoch = 30
Number_Hidden = 128
Speed_Learning = 0.01
Opt = SGD(lr=Speed_Learning)
Proportion_Validation = 0.2
Size_Batch = 1024
Verbose = 1
Number_Classes = 10
Length_Unit = 784
np.random.seed(461)

(input_X_train, output_Y_train), (input_X_test, output_Y_test) = mnist.load_data()
print("Training data input shape: ", input_X_train.shape)
print("Training data output shape: ", output_Y_train.shape)
print("Test data input shape: ", input_X_test.shape)
print("Test data output shape: ", output_Y_test.shape)

image = input_X_train[10]
print("A sample image is as an array: ")
print(str(image))
# plt.imshow(image, cmap='gray')
# plt.show()

input_X_train = input_X_train.reshape(60000, Length_Unit)
input_X_test = input_X_test.reshape(10000, Length_Unit)
input_X_train = input_X_train.astype('float32')
input_X_test = input_X_test.astype('float32')

input_X_train /= 255
input_X_test /= 255
output_Y_train = np_utils.to_categorical(output_Y_train, Number_Classes)
output_Y_test = np_utils.to_categorical(output_Y_test, Number_Classes)

model = Sequential()

model.add(Dense(Number_Hidden, input_shape=(Length_Unit,)))
model.add(Activation('relu'))

model.add(Dense(Number_Hidden))
model.add(Activation('relu'))

model.add(Dense(Number_Hidden))
model.add(Activation('relu'))

model.add(Dense(Number_Hidden))
model.add(Activation('relu'))

model.add(Dense(Number_Hidden))
model.add(Activation('relu'))

# model.add(Dense(Number_Hidden))
# model.add(Activation('relu'))

# model.add(Dense(Number_Classes))
# model.add(Activation('relu'))

# model.add(Dense(Number_Classes))
# model.add(Activation('relu'))

# model.add(Dense(Number_Classes))
# model.add(Activation('relu'))

# model.add(Dense(Number_Classes))
# model.add(Activation('relu'))

model.add(Dense(Number_Classes))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Opt, metrics=['accuracy'])

fitted_model = model.fit(input_X_train, output_Y_train, batch_size=Size_Batch, epochs=Epoch, verbose=Verbose, validation_split=Proportion_Validation)

score = model.evaluate(input_X_test, output_Y_test, verbose=Verbose)
print("Test score/loss:", score[0])
print("Test accuracy:", score[1])

plt.plot(fitted_model.history['acc'])
plt.plot(fitted_model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

result = model.predict(input_X_test, batch_size=Size_Batch, verbose=Verbose)
new = []
for i in range(0, len(result)):
    if output_Y_test[i][0] == 1:
        new.append(result[i][0])
plt.boxplot(new)
plt.show()
