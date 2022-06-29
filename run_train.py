import data_loader
import neural_networks
from utils import data_split, next_batch

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from numpy import argmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_roc_curve, roc_curve, auc
import matplotlib.pyplot as plt

try:
    problem = sys.argv[1]
except:
    print("Enter the command as the folowing form:")
    print("$ python run_train.py [problem]")
    print("problem can be covid19, breast_cancer, parkinsons, human_stress, or mental_health")
    sys.exit()

if problem == 'covid19':
    # load data
    X, y = data_loader.COVID19_dataset('./Dataset/corona_tested_individuals_ver_006.english.csv')
    model = neural_networks.Covid19_neural_network()
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.01
    MOMENTUM=0.5
    opt = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    lossFunc = nn.MSELoss()
    save_trained_model = './Models/covid19_model.pth'
elif problem == 'breast_cancer':
    # load data
    X, y = data_loader.WDBC_dataset('')
elif problem == 'parkinsons':
    # load data
    X, y = data_loader.Parkinsons_dataset('')
elif problem == 'human_stress':
    # load data
    X, y = data_loader.SaYoPillow_dataset('')
elif problem == 'mental_health':
    # load data
    X, y = data_loader.NHANES_dataset('')
else:
    print("The problem is not supported!!!")
    sys.exit()

# spit data to train and test set
test_size=0.25
random_state=0
X_train, X_test, y_train, y_test = data_split(X, y, test_size, random_state)

# create a template to summarize training/testing progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

for epoch in range(0, EPOCHS):
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    samples = 0
    model.train()

    train_preds = []
    for (batchX, batchY) in next_batch(X_train, y_train, BATCH_SIZE):
        outputs = model(batchX)
        loss = lossFunc(outputs, batchY)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trainLoss += loss.item() * batchY.size(0)

        for output in outputs:
            if output[0] > output[1]:
                train_preds.append(0)
            else:
                train_preds.append(1)

        samples += batchY.size(0)
    
    y_true  = argmax(y_train, axis=1)
    trainAcc = accuracy_score(y_true, train_preds)

    trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples), trainAcc))

    testLoss = 0
    samples = 0
    model.eval()

    test_preds = []
    with torch.no_grad():
        for (batchX, batchY) in next_batch(X_test, y_test, BATCH_SIZE):
            outputs = model(batchX)
            loss = lossFunc(outputs, batchY)
            testLoss += loss.item() * batchY.size(0)
            for output in outputs:
                if output[0] > output[1]:
                    test_preds.append(0)
                else: 
                    test_preds.append(1)
            samples += batchY.size(0)
        
        y_true  = argmax(y_test, axis=1)
        testAcc = accuracy_score(y_true, test_preds)

        # display model progress on the current training batch
        trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
        print(trainTemplate.format(epoch + 1, (testLoss / samples), testAcc))

# Save trained model
torch.save(model, save_trained_model)