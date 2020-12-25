import emnist
import numpy as np
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import SGD
from settings import *

# tryLater:
    # different layers
    # different optimizers  (SGD vs 'adam')
# 4-5 epochs is fine for judging overall accuracy

def reformat(images, labels):
    """ 
    Reshape images into single channel and normalize.
    One-hot encode labels into 26 different categories for each character
    """
    images = images.reshape((images.shape[0], COMPRESSED_WIDTH, COMPRESSED_WIDTH, 1))
    images = images.astype('float32') / float(WHITE_COLOR)
    labels = to_categorical(labels)  # one-hot encoding
    return images, labels


def loadTrainData(category='letters'):
    """
    Load training data for EMNIST uppercase/lowercase 26 characters and
    format appropriately
    """
    trainImages, trainLabels = emnist.extract_training_samples(category)
    return reformat(trainImages, trainLabels)
    

def defineModel():
    """
    Define layered convolutional neural network for handwriting classification.
    """
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(COMPRESSED_WIDTH, COMPRESSED_WIDTH, 1), data_format='channels_last'))
    model.add(Convolution2D(32, (3, 3), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_LETTERS+1, activation='softmax'))

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def trainCNN(filepath='./saved_model', batchSize=32, numEpochs=10):
    """
    Train CNN based on training data.
    Save model under specified name
    """
    trainImages, trainLabels = loadTrainData()
    model = defineModel()
    model.fit(trainImages, trainLabels, batch_size=batchSize, epochs=numEpochs, verbose=1)
    save_model(model, filepath, save_format='h5')


if __name__ == '__main__':
    trainCNN()