import cv2
from matplotlib import lines
import numpy as np
import pandas as pd
import os
from pandas import read_csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import load_model
from img_input import main
from gtts import gTTS
import os


model = load_model("model_hand.h5")
letters, lines = main()

def predict():
    
    word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
                13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

    sentence = []
    for padded, i in zip(letters, lines):
        pred = model.predict(np.array([padded]))
        try:
            sentence[i].append(word_dict[np.argmax(pred[0])])
        except IndexError:
            sentence.append([word_dict[np.argmax(pred)]])

    final_output = ""
    for i in sentence:
        for j in i:
            final_output += j
        final_output += " "

    return final_output


def tts(final_output):
    s = gTTS(text=final_output, lang='en', slow=False)
    s.save("hand_reg_text.mp3")
    os.system("start hand_reg_text.mp3")


def img_input_show():
    window = np.concatenate(letters, axis=1)
    cv2.imshow("Processed input", window)
    cv2.waitKey(0)

def img_dataset_show():
    data = pd.read_csv(r"data.csv").astype('float32')
    X = data.drop('0', axis=1)
    y = data['0']
    x = np.reshape(X.values, (X.shape[0], 28, 28))

    window = np.concatenate(x[:20], axis=1)
    cv2.imshow("Dataset images", window)
    cv2.waitKey(0)