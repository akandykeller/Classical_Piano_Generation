# Classical_Piano_Generation

The goal of this project was to study the effect of variations in training techniques and model architectures on an LSTM's ability to generate classical piano music. We additionally tested the viability of applying traditional generative model evaluation techniques such as Indirect Sampling Likelihood to generated music. The encoding used for the music was chosen to be MIDI to allow the model to focus on learning note structure and composition rather than constructing notes from scratch. 

The model architechtures attempted included 1 & 2 layer LSTMs, Biaxial RNN, and RNN-RBM. The vanilla LSTMs were implemented using Keras. The Biaxial RNN was can be found here: https://github.com/hexahedria/biaxial-rnn-music-composition. The RNN-RBM was implemented in Theano, and modified from the example implementation.

The training techniques tested included curriculum learning, reversing the input training sequences, key normalization, and variations of input encodings. In the end, curriculum learning was shown to be surprisingly effective for all model architechtures, and combined with key normalization delivered qualitatively pleasant music.

Example generated compositions can be found under compositions/.

This project was completed as my team's final project for CSE 253. Our full writeup is contained in CSE_253_Report.pdf