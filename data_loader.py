import pandas as pd
import numpy as np

def COVID19_dataset(filepath):
    dataFrame = pd.read_csv(filepath, low_memory=False)
    # romove all None and 
    dataFrame['cough'] = dataFrame['cough'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['cough'])
    dataFrame['fever'] = dataFrame['fever'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['fever'])
    dataFrame['sore_throat'] = dataFrame['sore_throat'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['sore_throat'])
    dataFrame['shortness_of_breath'] = dataFrame['shortness_of_breath'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['shortness_of_breath'])
    dataFrame['head_ache'] = dataFrame['head_ache'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['head_ache'])
    dataFrame['age_60_and_above'] = dataFrame['age_60_and_above'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['age_60_and_above'])
    dataFrame['gender'] = dataFrame['gender'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['gender'])
    dataFrame['test_indication'] = dataFrame['test_indication'].replace('None', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['test_indication'])
    dataFrame['corona_result'] = dataFrame['corona_result'].replace('other', np.nan)
    dataFrame = dataFrame.dropna(axis=0, subset=['corona_result'])

    # make data balanced
    dataFrame = dataFrame.groupby('corona_result')
    dataFrame = dataFrame.apply(lambda x: x.sample(dataFrame.size().min()).reset_index(drop=True))

    # replace Yes/No, Positive/Negative ... to 0, 1
    # for - cough, fever, sore_throat, shortness_of_breath, head_ache
    dataFrame = dataFrame.replace("1", 1)
    dataFrame = dataFrame.replace("0", 0)
    # for - age_60_and_above
    dataFrame = dataFrame.replace("Yes", 1)
    dataFrame = dataFrame.replace("No", 0)
    # for - gender
    dataFrame = dataFrame.replace("male", 1)
    dataFrame = dataFrame.replace("female", 0)
    # for - test_indication ['Other' 'Abroad' 'Contact with confirmed']
    dataFrame = dataFrame.replace("Contact with confirmed", 1)
    dataFrame = dataFrame.replace("Other", 0)
    dataFrame = dataFrame.replace("Abroad", 0)
    # for - corona_result
    dataFrame = dataFrame.replace("positive", 1)
    dataFrame = dataFrame.replace("negative", 0)


    features = dataFrame[['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache', 'age_60_and_above', 'gender', 'test_indication']]
    corona_results = dataFrame['corona_result']
    
    X = features.to_numpy()
    y = corona_results.to_numpy()

    return X, y


def WDBC_dataset(filepath):
    X = []
    y = []
    return X, y


def Parkinsons_dataset(filepath):
    X = []
    y = []
    return X, y


def SaYoPillow_dataset(filepath):
    X = []
    y = []
    return X, y


def NHANES_dataset(filepath):
    X = []
    y = []
    return X, y