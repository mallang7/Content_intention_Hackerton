import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import re
import json
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

import train

from train import clean_train_text
from train import LABEL
from train import stop_words
from train import model

test = pd.read_table('test.txt', header = None)
test_data = test[1]
test_label = test[0]

def label_predict(test_data):


  okt = Okt()
  clean_test = []

  for x in tqdm(test_data):

    if type(x) == str:
      clean_test.append(train.preprocessing(x, okt, remove_stopwords = True, stop_words=stop_words))
    else:
      clean_test.append([])  

  test_tokenizer = Tokenizer()
  test_tokenizer.fit_on_texts(clean_train_text)
  test_sequences = test_tokenizer.texts_to_sequences(clean_test)
  test_word_vocab = test_tokenizer.word_index
  MAX_SEQUENCE_LENGTH = 30
  
  test_inputs = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

  x_test = train.vectorize_sequences(test_inputs)


  predictions = model.predict(x_test)

  test_predict = []
  for i in range(len(test_data)):
    a = np.argmax(predictions[i])
    test_predict.append(a)

  PREDICT = []
  for i in range (len(test_data)):
    for j in range(len(LABEL)):
      if test_predict[i] == LABEL[j][0]:
        PREDICT.append(LABEL[j][1])


  with open('result.txt', 'w') as f:
    for line in PREDICT:
        f.write(line)
        f.write("\n")


label_predict(test_data)
