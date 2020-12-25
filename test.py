import emnist
from PIL import Image
import numpy as np

# THINGS TO DO
# predict characters & digits with diff datasets
# try training on diff datasets in emnist library

# DATA FORMAT for n samples
# images: n x 28 x 28
# labels: n


avaliable_datasets = emnist.list_datasets()

# train data
images, labels = emnist.extract_training_samples('letters')

# test data
images, labels = emnist.extract_test_samples('letters')



# idx = 2
# new_image = Image.fromarray(images[idx])
# new_image.show()
# print(chr(ord('a') + labels[idx] - 1))


