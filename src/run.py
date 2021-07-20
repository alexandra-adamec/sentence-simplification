'''
Main file that starts the program.
The model is specified in the import: import the concrete model file
First the model is trained, than tested.
To load specific parameters, specify the file name as a last argument in SimplRNN class.
'''
__author__ = 'Alexandra'


import data_utils as dtu
import simpl_brnn_lstm as model
import logging

logging.info('STARTING SimplRNN')

#hyperparameters
h_size = 1000
s_size = 1000
al_size = 1000


enc_l2 = 0.000001
dec_l2 = 0.000001
n_epochs = 10
batch_size = 50
lr_init = 0.01
lr_decay = 0.998
mom = 0.01
beam_size = 10

dataset = dtu.DataSet(batch_size, 'data/', 'aligned.zip')
simplRNN = model.SimplRNN(
    dataset.voc_main,
    dataset.voc_simple,
    dataset.voc_main.emb,
    dataset.voc_simple.emb,
    h_size,
    s_size,
    al_size,
    enc_l2,
    dec_l2,
    lr_init,
    lr_decay,
    mom,
    batch_size,
    beam_size,
    None)#'param/model_2015-08-09T17:07:01_me640h1000se640s1000.pickle')

simplRNN.train(dataset, n_epochs)


dataset = dtu.DataSet(batch_size, 'data/', 'aligned.zip', testing=True)
simplRNN.test(dataset)