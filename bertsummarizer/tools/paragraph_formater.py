import pickle
from typing import List
import string
import os
from bertsummarizer.tools import models

os.path.dirname(models.__file__)

with open(os.path.dirname(models.__file__) + '/mlp.pickle', 'rb') as f:
    mlp = pickle.load(f)

with open(os.path.dirname(models.__file__) + '/scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

def initialUpper(txt: str):
    if txt[0].isupper():
        return 1
    return 0
    
def numInBeginning(txt: str):
    if True in [x.isnumeric() for x in txt[:4]]:
        return 1
    return 0

def getPunctFeature(character: str):
    features_dummy = []
    for x in string.punctuation:
        features_dummy.append(1 if character == x else 0)
    return features_dummy

def extract_features(pair: List[str]):
    features = []
    for p in pair:
        sentence_len = len(p)
        features.append(sentence_len)
        if sentence_len > 0:
            features.append((sum([1 for x in p if x.isupper()]) / sentence_len) * 100)
            features.append(initialUpper(p))
            features.extend(getPunctFeature(p[0]))    
            features.extend(getPunctFeature(p[-1]))
            features.append(numInBeginning(p))
        else:
            features.append(0)
            features.append(0)
            features.extend(getPunctFeature(''))    
            features.extend(getPunctFeature(''))
            features.append(0)            
    return features    

def format(paragraphs):
    if paragraphs == []:
        return []        
    formated_paragraphs = []
    formated_paragraphs.append(paragraphs[0])
    moves = 0
    for p in paragraphs[1:]:
        features = extract_features([formated_paragraphs[-1], p])
        correct = mlp.predict(scaler.transform([features]))[0]
        if correct == 0:
            formated_paragraphs[-1] += ' ' + p
            moves += 1
        else:
            formated_paragraphs.append(p)
    if moves > 0:
        formated_paragraphs = format(formated_paragraphs)
    return formated_paragraphs    