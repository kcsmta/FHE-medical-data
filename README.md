# Fully homomorphic encryption for Privacy-preserving for Medical data

## Step 1: Install python 3.8
(Because CrypTen requires Python >=3.8, <3.10) Follow below steps:
```
$ sudo apt update
$ sudo apt install software-properties-common
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt install python3.8
$ sudo apt-get install python3.8-dev
```

## Step 2: Create virtual enviroment
Follow below steps:
```
$ python3.8 -m pip install pip
$ python3.8 -m pip install virtualenv
$ python3.8 -m virtualenv venv
$ source venv/bin/activate
```

## Step 3: Install pytorch
Follow the link: https://pytorch.org/get-started/locally/
```
$ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

## Step 4: Install Tenseal

```
$ pip install tenseal
```

## Step 5: Install Concrete-ML
### Step 5.1: Install graphviz
Install the GCC Compiler:
```
$ sudo apt update
$ sudo apt install build-essential
$ sudo apt-get install manpages-dev
```
```
$ sudo apt-get install graphviz graphviz-dev
$ pip install pygraphviz
```

### Step 5.2: Install Concrete-ML
```
$ pip install concrete-ml
```
