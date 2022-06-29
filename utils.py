import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# spit data to train and test set
def data_split(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train)
    y_train = F.one_hot(y_train).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test)
    y_test = F.one_hot(y_test).type(torch.FloatTensor)
    return X_train, X_test, y_train, y_test

# load batch data during training process
def next_batch(inputs, targets, batchSize):
	# loop over the dataset
	for i in range(0, inputs.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (inputs[i:i + batchSize], targets[i:i + batchSize])