import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.datasets import GTZAN
from torchaudio.datasets.utils import download_url
from torch.utils.data import DataLoader
import torchaudio.transforms as tt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch
import torchaudio
import os
from youtube_dl import YoutubeDL
import IPython
from pydub import AudioSegment

audio_downloder = YoutubeDL({'format':'bestaudio'})

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

import essentia.standard as es
import mirdata
import numpy as np

import json

from collections import Counter
from sklearn import preprocessing

MODEL_NAME = 'genre_tzanetakis-musicnn-msd-1'
MODEL_JSON = f'{MODEL_NAME}.json'
MODEL_PB = f'{MODEL_NAME}.pb'

MUSICNN_SR = 16000 #We will fix sample rate at 16 kHz as it is required for the input of MusiCNN model.
def extract_mean_embedding(filename):
  """
  Extract mean-temporal embedding from audio contained in filename

  Args:
    filename (str): Name of the audio file

  Return:
    Mean embedding of the song
  """
  
  # Load audiofile with essentia monoloader to resample the audios to the necessary sample rate in MusiCNN model
  audio = es.MonoLoader(filename=filename, sampleRate=MUSICNN_SR)()

  # Extract the embedding
  musicnn_emb = es.TensorflowPredictMusiCNN(graphFilename=MODEL_PB, output='model/dense/BiasAdd')(audio)

  # Compute mean-embedding across the frames
  mean_emb = np.mean(musicnn_emb, axis=0)
  mean_emb = mean_emb[np.newaxis, :]  # Each song is a 1x200 row vector

  return mean_emb

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.4)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(0)
        # print("Input Shape",x.shape)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).float())
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out) 
        return out

with open('/content/embeddings.npy', 'rb') as f:
    embeddings = np.load(f)
with open('/content/labels.npy', 'rb') as f:
    labels = np.load(f)
with open('/content/labels_decoded.npy', 'rb') as f:
    labels_decoded = np.load(f)
with open('/content/track_ids.npy', 'rb') as f:
    track_ids = np.load(f)

genres = {genre_id: genre for genre_id, genre in zip(labels, labels_decoded)}

def get_genre(wav_file=None,youtube_link=None):
  if youtube_link:
    info = audio_downloder.extract_info(url=youtube_link, download=True)
    wav = AudioSegment.from_file(info['title']+'-'+info['display_id']+'.'+info['ext'])
    wav.export("temp.wav", format="wav")
    features = extract_mean_embedding("temp.wav")
    os.remove("temp.wav")
  else:
    features = extract_mean_embedding(wav_file)

  feature_tensor = torch.from_numpy(features)
  outputs = model(feature_tensor).squeeze(0)
  _, predicted = torch.max(outputs, 1)
  return (genres[predicted.item()])

app = Flask(__name__)
model = torch.load('rnn_epoch_100_R2',map_location=torch.device('cpu'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    music_url = str(request.form['url'])
    wav_file = str(request.form['wav_file'])

    output = get_genre(wav_file,music_url)
    # final_features = [np.array(int_features)]
    # # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Genre is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)