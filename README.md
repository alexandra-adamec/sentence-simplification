Sentence simplification solved as monolingual machine translation. An input complex sentence is encoded by an RNN-Encoder into a fixed length vector representation. RNN-Decoder generates words of an output simple sentence according to RNN language model, in the context of the encoded input sentence.

Source files:
-------------------
simpl_brnn_lstm.py: Implementation of a bi-directional RNN Encoder-Decoder model with attentions that simplifies sentences. Uses LSTM units. Contains also the code for training, validation and testing the model.

simpl_rnn_lstm.py: Implementation of a simple RNN Encoder-Decoder using LSTM units.

simpl_rnn_gru.py: Implementation of a simple RNN Encoder-Decoder using GRU units.

data_utils.py: Utility code to load/preprocess data and to organize data-set into mini-batches. Sequences in mini-batches are 0-padded to the length of the longest sentence, 0/1 mask matrix is also provided.

evaluation.py: BLEU implementation used to validate and test the model.

run.py: Starts the whole magic.