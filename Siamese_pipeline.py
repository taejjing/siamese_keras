import os
import pickle
import numpy as np

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self):
        self.data = {}
        self.categories = {}
        self.info = {}

    def load_data(self, path, data_subsets):
        # This code should be extracted to method... Done
        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))

            with open(file_path, "rb") as f:
                (X, c) = pickle.load(f)
                self.data[name] = X
                self.categories[name] = c
    
    def get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X = self.data[s]
        n_classes, n_examples, w, h = X.shape

        # randomly sample several classes to use in the batch
        categories = np.random.choice(n_classes, size=(batch_size, ), replace=False)
        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]
        # initialize vector for the targets, and make one half of it '1's so 2nd half of batch has same class
        targets = np.zeros((batch_size, ))
        targets[batch_size // 2] = 1

        for i in range(batch_size):
            category = categories[i]
            idx_1 = np.random.randint(0, n_examples)
            pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
            idx_2 = np.random.randint(0, n_examples)
            # pick images of same class for 1st half, different for 2nd
                





