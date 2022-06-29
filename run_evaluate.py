import data_loader
import neural_networks
from utils import data_split, next_batch

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from numpy import argmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_roc_curve, roc_curve, auc
import matplotlib.pyplot as plt

try:
    problem = sys.argv[1]
except:
    print("Enter the command as the folowing form:")
    print("$ python run_evaluate.py [problem]")
    print("problem can be covid19, breast_cancer, parkinsons, human_stress, or mental_health")
    sys.exit()

# load data
if problem == 'covid19':
    X, y = data_loader.COVID19_dataset('./Dataset/corona_tested_individuals_ver_006.english.csv')
    model = neural_networks.Covid19_neural_network()
    BATCH_SIZE = 64
    lossFunc = nn.MSELoss()
    save_trained_model = './Models/covid19_model.pth'
elif problem == 'breast_cancer':
    X, y = data_loader.WDBC_dataset('')
elif problem == 'parkinsons':
    X, y = data_loader.Parkinsons_dataset('')
elif problem == 'human_stress':
    X, y = data_loader.SaYoPillow_dataset('')
elif problem == 'mental_health':
    X, y = data_loader.NHANES_dataset('')
else:
    print("The problem is not supported!!!")
    sys.exit()

# spit data to train and test set
test_size=0.25
random_state=0
X_train, X_test, y_train, y_test = data_split(X, y, test_size, random_state)

# load trained model
model = torch.load(save_trained_model)

# evaluate trained model
testLoss = 0
samples = 0
test_preds = []
with torch.no_grad():
    start_time = time.time()
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
    end_time = time.time()
    infer_time = end_time - start_time
    y_true = argmax(y_test, axis=1)
    testAcc = accuracy_score(y_true, test_preds)
    testF1 = f1_score(y_true, test_preds)
    testPrecision = precision_score(y_true, test_preds)
    testRecall = recall_score(y_true, test_preds)
    cm = confusion_matrix(y_true, test_preds)

print("===========Test model===========")    
print("+ Inference time = ", infer_time)
print("+ Accuracy = ", testAcc)
print("+ F1 score = ", testF1)
print("+ Precision = ", testPrecision)
print("+ Recall = ", testRecall)
print("+ Confusion matrix:\n", cm)

import matplotlib.pyplot as plt

with torch.no_grad():
    test_preds = model(X_test)

fpr, tpr, thresholds = roc_curve(y_test[:, 0], test_preds[:, 0])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
