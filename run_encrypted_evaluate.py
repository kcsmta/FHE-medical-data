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
    plain_model = neural_networks.Covid19_neural_network()
    BATCH_SIZE = 64
    lossFunc = nn.MSELoss()
    save_trained_model = './Models/covid19_model.pth'
    # load trained model and convert to model with encrypted input
    plain_model = torch.load(save_trained_model)
    model = neural_networks.Encrypted_Covid19_neural_network(plain_model)
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

# parameters
poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
# create TenSEALContext
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
# scale of ciphertext to use
ctx_eval.global_scale = 2 ** 20
# this key is needed for doing dot-product operations
ctx_eval.generate_galois_keys()

# encrypt input data
t_start = time()
enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in X_test]
t_end = time()
print(f"Encryption of the test-set took {int(t_end - t_start)} seconds")

# evaluate trained model on encrypted data
t_start = time()

correct = 0
for enc_x, y in zip(enc_x_test, y_test):
    # encrypted evaluation
    enc_out = model(enc_x)
    # plain comparaison
    out = enc_out.decrypt()
    out = torch.tensor(out)
    out = torch.sigmoid(out)
    if torch.abs(out - y) < 0.5:
        correct += 1

t_end = time()
print(f"Evaluated test_set of {len(x_test)} entries in {int(t_end - t_start)} seconds")
print(f"Accuracy: {correct}/{len(x_test)} = {correct / len(x_test)}")
return correct / len(x_test)
    

encrypted_accuracy = encrypted_evaluation(eelr, enc_x_test, y_test)
diff_accuracy = plain_accuracy - encrypted_accuracy
print(f"Difference between plain and encrypted accuracies: {diff_accuracy}")
if diff_accuracy < 0:
    print("Oh! We got a better accuracy on the encrypted test-set! The noise was on our side...")

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
