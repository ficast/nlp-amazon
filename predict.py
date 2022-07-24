from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

num_words = 1000
oov_token = '<UNK>'
pad_type = 'post'
trunc_type = 'post'
maxlen = 249


class Predict():
    def __init__(self):
        self.model = tf.keras.models.load_model('amazon_nlp/')

    def clean(self, text_list):
        texts = []
        for text in text_list:
            text = re.sub(r"@[A-Za-z0-9]+", " ", text)
            text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)
            text = re.sub(r"[^A-Za-z0-9]", " ", text)
            text = re.sub(r" +", " ", text)
            texts.append(text)
        return texts

    def tokenizer(self, text):
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        tokenizer.fit_on_texts(text)
        return tokenizer

    def predict(self, text_list):
        texts = self.clean(text_list)
        tokenizer = self.tokenizer(texts)
        sentences = tokenizer.texts_to_sequences(texts)
        predict_padded = pad_sequences(
            sentences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)
        for i, text in zip(self.model.predict(predict_padded), texts):
            print("Your comment is:")
            print('"""')
            print(text)
            print('"""')
            print("")
            print("The sentiment is most likely to be:")
            print("Positive" if i > 0.5 else "Negative")
            print("")
            print("With an accuracy of:")
            print(i[0])
            print("")


if __name__ == "__main__":
    try:
        text = sys.argv[1]
    except IndexError:
        print("No message passed")
        text = input("Input message here: ")
    predict = Predict()

    predict.predict([text])
    print("Done")
