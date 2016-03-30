from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
import numpy as np
import random
import sys
from midi.utils import midiread, midiwrite
import os

import csv


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


MIDI_RANGE = (21, 109)
PIANO_RANGE = MIDI_RANGE[1] - MIDI_RANGE[0]
MEASURES = 2
TICKS_PER_MEASURE = 16
STEP = 9
EPOCHS = 1000
BATCH_SIZE = 32
TICKS_PER_INPUT = MEASURES*TICKS_PER_MEASURE
GEN_LENGTH = TICKS_PER_INPUT*8
DT = 0.3
    
if __name__ == '__main__':
  #names = ["very_easy", "easy", "medium", "hard", "harder", "fulldata"]
  names = ["fulldata"]

  # Number of iterations to do for each step of the curriculum training, harder has many more
  # it has >150 files
  iters_by_difficulty = {"very_easy":300, "easy":500, "medium":500, "hard": 500, "harder":2000, "fulldata":10000}

  history = LossHistory()

  with open('errors_iter_fulldata.csv', 'w') as csvfile:
      fieldnames = ['Epoch', 'Loss']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

      writer.writeheader()

  print 'building model...'
  model = Sequential()
  model.add(LSTM(512,  return_sequences=True, stateful=True,
                       batch_input_shape=(BATCH_SIZE, TICKS_PER_INPUT, PIANO_RANGE)))
  model.add(Dropout(0.2))
  model.add(LSTM(512,  return_sequences=False))
  model.add(Dense(PIANO_RANGE))
  model.add(Activation('sigmoid'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  
  # Use the following line to load saved weights
  model.load_weights('lstm_normalized_curriculum/output_harder/lstm_curric_harder_weights_500.h5')

  # Do curriculum training in order of difficulty
  for difficulty in names:
      #  Using C-normalized data
      if difficulty != 'fulldata':
          data_dir = "../biaxial-rnn-music-composition/C_Curriculum/" + difficulty + "/"
      else:
          data_dir = "../biaxial-rnn-music-composition/C_music/"

      save_dir = 'lstm_normalized_curriculum/output_{}/'.format(difficulty)

      ### read MIDI
      #data_dir = '../Data/Cirriculum/easy/'
      #data_dir = '../biaxial-rnn-music-composition/music/'

      files = os.listdir(data_dir)
      files = [data_dir + f for f in files if '.mid' in f or '.MID' in f]

      print files 

      dataset = []

      for f in files:
          try:
              dataset.append(midiread(f, MIDI_RANGE, DT).piano_roll)
              print "{} loaded".format(f)
          except IndexError:
              print "Skipping {}".format(f)
              pass

      print np.shape(dataset)

      X = []
      y = []

      for song in dataset:
          for i in range(0, len(song) - TICKS_PER_INPUT, STEP):
              X.append(song[i: i + TICKS_PER_INPUT])
              y.append(song[i + TICKS_PER_INPUT])

      max_samples = (len(X) // BATCH_SIZE) * BATCH_SIZE
      X = X[:max_samples]
      y = y[:max_samples]

      X = np.array(X)
      y = np.array(y)

      print np.shape(X)
      print np.shape(y)

      for iteration in range(1, iters_by_difficulty[difficulty]):
          print
          print 'Epoch {}, training...'.format(iteration)
          model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1, callbacks=[history])

          print "Loss: {}".format(history.losses)

          with open('errors_iter.csv', 'a') as csvfile:
              fieldnames = ['Epoch', 'Loss']
              writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
              writer.writerow({'Epoch': iteration, 'Loss': history.losses})


          if iteration % 500 == 0 or (iteration % 100 == 0 and iteration < 1000):  
              print "Generating sample..."
              # First we reset the state of the LSTM layer(s) so they can 
              # be dedicated to the new song
              model.reset_states()

              # To seed each generation, pick a random song, and use
              # it's first TICKS_PER_INPUT
              seed_song = random.randint(0, len(dataset) - 1)
              seed = np.array(dataset[seed_song][0:TICKS_PER_INPUT])

              seed_batch = np.zeros([BATCH_SIZE, TICKS_PER_INPUT, PIANO_RANGE])
              seed_batch[0][0] = seed[0]


              generated = []
              # for note in seed:
              #   generated.append(note)

              # Now add seperator note
              # generated.append(np.ones_like(seed[0]))
              generated.append(seed[0])

              for i in range(GEN_LENGTH):
                # np.newaxis is needed to make predict work for some reason...
                # pred_probs = model.predict(seed[np.newaxis, :], verbose=0)
                pred_probs = model.predict(seed_batch, verbose=0)

                next_note = np.random.binomial(n=1, p=pred_probs[0], size=pred_probs[0].shape)

                ### append to MIDI
                generated.append(next_note)
                # Update the input
                # seed = np.vstack((seed[1:], next_note))
                seed_batch[0][0] = next_note

              ### output MIDI
              sample_filename = save_dir + "lstm_curric_{}_composition_{}.mid".format(difficulty, iteration)
              print "Saving sample to {}".format(sample_filename)
              midiwrite(sample_filename, generated, MIDI_RANGE, DT)

              ### save model weights
              weights_filename = save_dir + "lstm_curric_{}_weights_{}.h5".format(difficulty, iteration)
              print "Saving weights to {}".format(weights_filename)
              model.save_weights(weights_filename, overwrite=True)

              # Clean states again for training
              model.reset_states()
