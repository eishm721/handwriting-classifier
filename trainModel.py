import emnist
import numpy as np
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from settings import *
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator


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

    trainImages = trainImages[:1000]
    trainLabels = trainLabels[:1000]

    return reformat(trainImages, trainLabels)


def defineModel():
    """
    Define layered convolutional neural network for handwriting classification.
    
    784-[32C3-MP2-64C3-MP2]-512-27 w/ 50% dropout w/out bias
    """
    model = Sequential()

    # feature extraction
    model.add(Convolution2D(32, 3, use_bias=False, padding='same', activation='relu', input_shape=(COMPRESSED_WIDTH, COMPRESSED_WIDTH, 1), data_format='channels_last'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64, 3, use_bias=False, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    
    # classification
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_LETTERS+1, activation='softmax'))

    # compile model with 'adam' optimizer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
    trainCNN('./mini_dataset')