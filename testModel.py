import emnist
import numpy as np
import os
from keras.models import load_model
from trainModel import reformat
from settings import *


class HandwritingClassifier:
    """
    Classifier that uses pre-trained CNN to predict handwritten characters
    """
    def __init__(self, filepath='./model_v2'):
        assert os.path.exists(filepath), "Run trainModel.py to train CNN before testing"
        self.model = load_model(filepath)

    def loadTestData(self, category='letters'):
        """
        Load testing data for EMNIST uppercase/lowercase 26 characters and
        format appropriately
        """
        testImages, testLabels = emnist.extract_test_samples(category)
        return reformat(testImages, testLabels)

    def evaluatePerformance(self):
        """
        Evaluate performance of model on test data
        """
        x_test, y_test = self.loadTestData()
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0) 
        print("Loss: {:.3f}".format(loss))
        print("Accuracy: {:.3f}%".format(accuracy*100))

    def predict(self, image):
        """
        Given a 2D 28x28 numpy array containing a user-drawn input,
        uses CNN to predict what character the user has drawn.

        Returns array of all 26 characters with their associated probability ranked
        from highest to lowest
        """
        image = image.reshape(1, COMPRESSED_WIDTH, COMPRESSED_WIDTH, 1)
        image = image.astype('float32') / float(WHITE_COLOR)
        predictions = self.model.predict(image)[0]

        # rank letters with highest probability
        indexToLetter = lambda i: (chr(ord('a') + i - 1)).upper()
        letterProbs = [(indexToLetter(i), predictions[i]) for i in range(1, len(predictions))]
        letterProbs.sort(reverse=True, key=lambda x: x[1])
        return letterProbs

     
if __name__ == '__main__':
    c = HandwritingClassifier('./mini_dataset')
    c.evaluatePerformance()






