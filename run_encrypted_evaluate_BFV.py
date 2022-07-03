from turtle import shape
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
import tenseal as ts

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
    # load trained model and convert to model with encrypted input
    save_trained_model = './Models/covid19_tenseal_model.pth'
    plain_model = neural_networks.Covid19_tenseal_neural_network()
    plain_model = torch.load(save_trained_model)
    secure_inference_model = neural_networks.Enc_Covid19_tenseal_neural_network(plain_model)
    ## IMPORTANT 
    ## Encryption Parameters
    # controls precision of the fractional part
    context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)

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

# encrypt input data
t_start = time.time()
enc_x_test = [ts.bfv_tensor(context, x.tolist()) for x in X_test]
t_end = time.time()
print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")

# evaluate trained model on encrypted data
t_start = time.time()

testLoss = 0
samples = 0
test_preds = []
for enc_x, y in zip(enc_x_test, y_test):
    enc_out = secure_inference_model(enc_x)
    out = enc_out.decrypt()
    out = torch.tensor(out)
    if out[0] > out[1]:
        test_preds.append(0)
    else: 
        test_preds.append(1)
    samples += 1

t_end = time.time()
infer_time = t_end - t_start
y_true = argmax(y_test, axis=1)
testAcc = accuracy_score(y_true, test_preds)
testF1 = f1_score(y_true, test_preds)
testPrecision = precision_score(y_true, test_preds)
testRecall = recall_score(y_true, test_preds)
cm = confusion_matrix(y_true, test_preds)

# import matplotlib.pyplot as plt

# with torch.no_grad():
#     test_preds = model(X_test)

# fpr, tpr, thresholds = roc_curve(y_test[:, 0], test_preds[:, 0])
# roc_auc = auc(fpr, tpr)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
