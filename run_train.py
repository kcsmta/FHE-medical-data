import data_loader
import neural_networks
from utils import data_split, next_batch

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from concrete.ml.torch.compile import compile_torch_model

import numpy as np
from numpy import argmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_roc_curve, roc_curve, auc
import matplotlib.pyplot as plt

try:
    problem = sys.argv[1]
    framework = sys.argv[2]
except:
    print("Enter the command as the folowing form:")
    print("$ python run_train.py [problem] [framework]")
    print("problem can be covid19, breast_cancer, parkinsons, human_stress, or mental_health")
    print("framework can be seal or tenseal")
    sys.exit()

if problem == 'covid19':
    # load data
    X, y = data_loader.COVID19_dataset('./Dataset/corona_tested_individuals_ver_006.english.csv')
    if framework == 'tenseal':
        model = neural_networks.Covid19_tenseal_neural_network()
        BATCH_SIZE = 64
        EPOCHS = 10
        LR = 0.01
        MOMENTUM=0.5
        opt = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        lossFunc = nn.CrossEntropyLoss()
        save_trained_model = './Models/covid19_tenseal_model.pth'
    elif framework == 'concreteml':
        model = neural_networks.Covid19_concreteml_neural_network()
        BATCH_SIZE = 64
        EPOCHS = 10
        LR = 0.01
        MOMENTUM=0.5
        opt = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        lossFunc = nn.CrossEntropyLoss()
        save_trained_model = './Models/covid19_concreteml_model.pth'
    else:
        print("The framework is not supported!!!")
        sys.exit()
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
if framework == 'tenseal':
    torch.save(model, save_trained_model)

if framework == 'concreteml':

    try:
        print("Trying to quantize neural network...")
        quantized_compiled_module = compile_torch_model(
            model,
            X_train,
            n_bits=3,
        )
        print("The network is trained and FHE friendly.")
    except RuntimeError as e:
        if str(e).startswith("max_bit_width of some nodes is too high"):
            print("The network is not fully FHE friendly, retraining.")
        raise e

    # TODO: Save quantized trained model
    
    # TODO: split to run_encrypted_evaluation_TFHE.py
    
    # Convert data to a numpy array.
    X_train_numpy = X_train.numpy()
    X_test_numpy = X_test.numpy()
    y_train_numpy = y_train.numpy()
    y_test_numpy = y_test.numpy()
    q_X_test_numpy = quantized_compiled_module.quantize_input(X_test_numpy)
    quant_model_predictions = quantized_compiled_module(q_X_test_numpy)

    from tqdm import tqdm

    y_pred = model(X_test) # evaluate on plain data
    fhe_x_test = quantized_compiled_module.quantize_input(X_test_numpy)
    homomorphic_quant_predictions = []
    for x_q in tqdm(fhe_x_test):
        homomorphic_quant_predictions.append(
            quantized_compiled_module.forward_fhe.encrypt_run_decrypt(np.array([x_q]).astype(np.uint8))
        )
    homomorphic_predictions = quantized_compiled_module.dequantize_output(
        np.array(homomorphic_quant_predictions, dtype=np.float32).reshape(quant_model_predictions.shape)
    )

    acc_0 = 100 * (y_pred.argmax(1) == y_test).float().mean()
    acc_1 = 100 * (quant_model_predictions.argmax(1) == y_test_numpy).mean()
    acc_2 = 100 * (homomorphic_predictions.argmax(1) == y_test_numpy).mean()

    print(f"Test Accuracy: {acc_0:.2f}%")
    print(f"Test Accuracy Quantized Inference: {acc_1:.2f}%")
    print(f"Test Accuracy Homomorphic Inference: {acc_2:.2f}%")