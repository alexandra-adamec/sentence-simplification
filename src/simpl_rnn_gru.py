__author__ = 'Alexandra'

'''
RNN Encoder-Decoder model with Gated Recurrent Units

Author: Alexandra Adamec
Student ID: s1475373
'''

import numpy as np
import theano.tensor.extra_ops as ops
import theano
from theano import tensor as T
from theano import gradient
from collections import OrderedDict
import math

import cPickle
import datetime
import evaluation
import data_utils as dtu
import copy
import logging
import datetime



FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, filename='log/log.txt')


class RecEncoder(object):
    '''
    Recurrent encoder, that takes as input a variable-length sequence of indices
    and returns a fixed-length vector of their encodings.
    '''


    def __init__(self, init_emb, hidden_size, backwards=False):
        '''
        Initialization of the RNN encoder, with all its shared parameters
        Theano symbolic variables and expressions for each layer activation.

        :param init_emb: embeddings matrix (P)
        :param hidden_size: size of one hidden state (|h_n|)
        :param backwards: forward or backward direction
        '''

        logging.debug('Encoder: initializing...')


        # Declaration of the shared encoder parameters.
        # weights - are initialized as Gaussian RV with mean=0 and variance=0.08
        # biases - are initialized to 0
        # word embeddings are not random, but initialized with the pre-trained embeddings in init_emb
        emb_size = init_emb.shape[1]
        emb = theano.shared(value=np.array([[]], dtype=theano.config.floatX))
        emb.set_value(init_emb, borrow=True)
        a = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, hidden_size)).astype(theano.config.floatX))
        az = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, hidden_size)).astype(theano.config.floatX))
        ar = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, hidden_size)).astype(theano.config.floatX))

        b = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, hidden_size)).astype(theano.config.floatX))
        bz = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, hidden_size)).astype(theano.config.floatX))
        br = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, hidden_size)).astype(theano.config.floatX))

        bias_h = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX))

        # deltas of the parameters store previous parameters updates,
        # used to compute the momentum in the next iteration of the training
        delta_a = theano.shared(value=np.zeros(a.get_value().shape, dtype=theano.config.floatX))
        delta_az = theano.shared(value=np.zeros(az.get_value().shape, dtype=theano.config.floatX))
        delta_ar = theano.shared(value=np.zeros(ar.get_value().shape, dtype=theano.config.floatX))

        delta_b = theano.shared(value=np.zeros(b.get_value().shape, dtype=theano.config.floatX))
        delta_bz = theano.shared(value=np.zeros(bz.get_value().shape, dtype=theano.config.floatX))
        delta_br = theano.shared(value=np.zeros(br.get_value().shape, dtype=theano.config.floatX))
        delta_bias_h = theano.shared(value=np.zeros(bias_h.get_value().shape, dtype=theano.config.floatX))

        h0 = theano.shared(value=np.array([0], dtype=theano.config.floatX))

        self.params = [a, az, ar, b, bz, br, bias_h]
        self.deltas = [delta_a, delta_az, delta_ar, delta_b, delta_bz, delta_br, delta_bias_h]

        # -----------------------------------------------
        # Symbolic declaration of the Encoder RNN model
        # -----------------------------------------------

        # input_batch: intput matrix with words indices; 1 sentence in 1 row
        self.input_batch = T.dmatrix()

        # transposed and converted indices -> embeddings vectors
        # transpose required because scan iterates over rows
        input_emb = emb[T.cast(self.input_batch.dimshuffle((1, 0)), 'int32')]

        def encode(x_j, h_jm1):
            # update gate
            z_j = T.nnet.sigmoid(T.dot(x_j, az) + T.dot(h_jm1, bz) + bias_h)
            # reset gate
            r_j = T.nnet.sigmoid(T.dot(x_j, ar) + T.dot(h_jm1, br) + bias_h)
            # memory unit
            h_mem = T.tanh(r_j * T.dot(h_jm1, b) + T.dot(x_j, a) + bias_h)
            # hidden state at j time-step
            h_j = ((1 - z_j) * h_mem) + (z_j * h_jm1)
            return h_j

        # scan iterates over input_emb rows, passes j-1 hidden state to each itaration
        # backwards defines the direction of the scan depending on whether it is a forward or backward encoder
        h, _ = theano.scan(fn=encode, sequences=[input_emb], outputs_info=[T.alloc(h0, input_emb.shape[1], hidden_size)], n_steps=input_emb.shape[0], go_backwards=backwards)
        self.encoding = h[-1, T.arange(self.input_batch.shape[0])]

class RecDecoder(object):
    '''
    Recurrent decoder, that computes the distribution of the future word given the history and the encoding or
    directly generates a sequence of such words given the encoding.
    '''

    def __init__(self, context, voc_size, init_emb, context_size, hidden_size, beam_size, start_symbol_idx = 0.0):
        '''
        Initialization of the RNN decoder, with all its shared parameters
        Theano symbolic variables and expressions for each layer activation.

        :param context: reference to the symbolic encoding of the input sequence (c)
        :param voc_size: size of the output vocabulary (|V_s|)
        :param init_emb: embeddings matrix (P)
        :param context_size: size of one encoding (|c_n|)
        :param hidden_size: size of one hidden state (|s_n|)
        :param beam_size: beam size for n-best search
        :param start_symbol_idx: index of the special LB (left boundary) symbol
        :return:
        '''
        logging.debug('Decoder: initializing...')

        # Declaration of the shared decoder parameters.
        # weights - are initialized as Gaussian RV with mean=0 and variance=0.08, only alignment params hasvariance=0.001
        # biases - are initialized to 0
        # word embeddings are not random, but initialized with the pre-trained embeddings in init_emb
        emb_size = init_emb.shape[1]
        emb = theano.shared(name='dec_emb', value=np.array([[]], dtype=theano.config.floatX))
        emb.set_value(init_emb, borrow=True)
        mask_emb = theano.shared(name='mask_emb', value=np.array([[0]*voc_size,[1]*voc_size], dtype=theano.config.floatX))

        #  hidden layer
        u = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, hidden_size)).astype(theano.config.floatX))
        uz = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, hidden_size)).astype(theano.config.floatX))
        ur = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, hidden_size)).astype(theano.config.floatX))

        w = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, hidden_size)).astype(theano.config.floatX))
        wz = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, hidden_size)).astype(theano.config.floatX))
        wr = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, hidden_size)).astype(theano.config.floatX))

        f = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (context_size, hidden_size)).astype(theano.config.floatX))
        fz = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (context_size, hidden_size)).astype(theano.config.floatX))
        fr = theano.shared(value=0.08 * np.random.uniform(-1.0, 1.0, (context_size, hidden_size)).astype(theano.config.floatX))

        bias_s = theano.shared(name='bs', value=np.zeros(hidden_size, dtype=theano.config.floatX))
        s0 = theano.shared(name='s0', value=np.array([0], dtype=theano.config.floatX))

        # output layer
        d = theano.shared(name='d', value=0.08 * np.random.uniform(-1.0, 1.0, (emb_size, voc_size)).astype(theano.config.floatX))
        v = theano.shared(name='v', value=0.08 * np.random.uniform(-1.0, 1.0, (hidden_size, voc_size)).astype(theano.config.floatX))
        g = theano.shared(name='g', value=0.08 * np.random.uniform(-1.0, 1.0, (context_size, voc_size)).astype(theano.config.floatX))
        by = theano.shared(name='by', value=np.zeros(voc_size, dtype=theano.config.floatX))

        # deltas of the parameters store previous parameters updates,
        # used to compute the momentum in the next iteration of the training
        delta_u = theano.shared(value=np.zeros(u.get_value().shape, dtype=theano.config.floatX))
        delta_uz = theano.shared(value=np.zeros(uz.get_value().shape, dtype=theano.config.floatX))
        delta_ur = theano.shared(value=np.zeros(ur.get_value().shape, dtype=theano.config.floatX))

        delta_w = theano.shared(value=np.zeros(w.get_value().shape, dtype=theano.config.floatX))
        delta_wz = theano.shared(value=np.zeros(wz.get_value().shape, dtype=theano.config.floatX))
        delta_wr = theano.shared(value=np.zeros(wr.get_value().shape, dtype=theano.config.floatX))

        delta_f = theano.shared(value=np.zeros(f.get_value().shape, dtype=theano.config.floatX))
        delta_fz = theano.shared(value=np.zeros(fz.get_value().shape, dtype=theano.config.floatX))
        delta_fr = theano.shared(value=np.zeros(fr.get_value().shape, dtype=theano.config.floatX))
        delta_bias_s = theano.shared(value=np.zeros(bias_s.get_value().shape, dtype=theano.config.floatX))

        delta_d = theano.shared(value=np.zeros(d.get_value().shape, dtype=theano.config.floatX))
        delta_v = theano.shared(value=np.zeros(v.get_value().shape, dtype=theano.config.floatX))
        delta_g = theano.shared(value=np.zeros(g.get_value().shape, dtype=theano.config.floatX))
        delta_by = theano.shared(value=np.zeros(by.get_value().shape, dtype=theano.config.floatX))

        self.params = [u, uz, ur, w, wz, wr, f, fz, fr, d, v, g, bias_s, by]
        self.deltas = [delta_u, delta_uz, delta_ur, delta_w, delta_wz, delta_wr, delta_f, delta_fz, delta_fr, delta_d, delta_v, delta_g, delta_bias_s, delta_by]

        # -----------------------------------------------
        # Symbolic declaration of the Decoder RNN model
        # -----------------------------------------------

        # Part 1: training
        # input_batch has indices of 1-previous word preceding the predicted output words
        # 1 row -> 1 sentence
        # 1 cell -> index of 1 word that precedes the predicted word at time step i
        # technically, in 1 row is a sequence [0:-1] of a training simple sentence
        self.input_batch = T.dmatrix()
        # transpose to have sentences over columns
        input_emb = T.cast(self.input_batch.dimshuffle((1, 0)), 'int32')
        # mask of the right-padded input sentences
        self.input_mask = T.dmatrix()
        mask = mask_emb[T.cast(self.input_mask.dimshuffle(1, 0), 'int32')]

        def decode(y_im1, s_im1, c):
            # update gate
            z_j = T.nnet.sigmoid(T.dot(emb[y_im1], uz) + T.dot(s_im1, wz) + T.dot(c, fz) + bias_s)
            # reset gate
            r_j = T.nnet.sigmoid(T.dot(emb[y_im1], ur) + T.dot(s_im1, wr) + T.dot(c, fr) + bias_s)
            # memory unit
            s_mem = T.tanh(r_j * T.dot(s_im1, w) + T.dot(emb[y_im1], u) + T.dot(c, f) + bias_s)
            # hidden state at time-step 1
            s_i = ((1 - z_j) * s_mem) + (z_j * s_im1)
            # prediction of the next word as a distribution over the output vocabulary
            y_i = T.nnet.softmax(T.dot(emb[y_im1], d) + T.dot(s_i, v) + T.dot(c, g) + by)
            return [s_i, y_i]

        [_, y], _ = theano.scan(fn=decode, sequences=input_emb, outputs_info=[T.alloc(s0, input_emb.shape[1], hidden_size), None], non_sequences=context, n_steps=input_emb.shape[0])

        # applying the mask to eliminate results from the 'padded' words
        # transposing back, to have sentences per row again
        self.prob_next = (((y - 1) * mask) + 1).dimshuffle(1, 0, 2)

        # Part 2: generating a sequence
        # context is provided explicitely
        self.gen_contexts = T.dmatrix()
        # predefined length of the generated sequences
        self.max_len = T.iscalar()
        y0 = theano.shared(name='y0', value=np.array([0], dtype=np.int64))
        ymax0 = theano.shared(name='ymax0', value=np.array([0.0], dtype=theano.config.floatX))


        def predict_word(y_im1, s_im1, loglkl_im1, c):
            # GRU computation
            z_j = T.nnet.sigmoid(T.dot(emb[y_im1], uz) + T.dot(s_im1, wz) + T.dot(c, fz) + bias_s)
            r_j = T.nnet.sigmoid(T.dot(emb[y_im1], ur) + T.dot(s_im1, wr) + T.dot(c, fr) + bias_s)
            s_mem = T.tanh(r_j * T.dot(s_im1, w) + T.dot(emb[y_im1], u) + T.dot(c, f) + bias_s)
            s_i = ((1 - z_j) * s_mem) + (z_j * s_im1)
            # next word distribution
            y_i = T.nnet.softmax(T.dot(emb[y_im1], d) + T.dot(s_i, v) + T.dot(c, g) + by)
            # added smoothing on the probabilities, to avoid log(0) computation
            return [y_i, s_i, T.log(y_i + 0.000000001) + loglkl_im1[:,np.newaxis]]


        # processing of one hypothesis across all the sentences
        def process_hypothesis(y_im1, s_im1, loglkl_im1, c):
            y_i, s_i, loglkl = predict_word(y_im1, s_im1, loglkl_im1, c)
            return [y_i, s_i, loglkl]

        # this is for selecting the n best hypothesis, there is |voc_size| * |beam_size| options
        # selected arg-maxs and max values from a sequence of log likelihoods of each hypothesis
        # selection of n-max implemented in a symbolic way as a scan-loop of n iterations
        def find_max(seq):
            max = T.max(seq, axis=1)
            argmax = T.argmax(seq, axis=1)
            min = T.min(seq, axis=1)
            # this is to eliminate max values that have been already selected
            one_hot = ops.to_one_hot(argmax, voc_size*beam_size, dtype=np.int8)
            return (seq * ((-1) * (one_hot - 1))) - (one_hot * (T.abs_(min[:,np.newaxis]) + 0.0000001)), max, argmax

        # this is for first time-step only,
        # there is only |voc_size| hypothesis at first time-step compared to |voc_size|*|beam_size| later
        # but we are reusing the same size of the variables
        # therefore, n-max is selected from only |0:voc_size| range and the rest |voc_size : beam_size*voc_size| has a very small negative value that can never be selected as max
        hypo_mask = theano.shared(value=np.array([[0]*voc_size + [-55555555555555]*(voc_size*(beam_size-1)),[0]*voc_size*beam_size], dtype=np.float32))


        def generate(y_im1, s_im1, log_lkl_im1, all_pred_y_im1, iter, c):
            # process all hypothesis, for each get a vocabulary distribution
            [_, tmp_s_i, tmp_loglkl],_ = theano.scan(fn=process_hypothesis, sequences=[y_im1.dimshuffle((1,0)), s_im1.dimshuffle((1,0,2)), log_lkl_im1.dimshuffle((1,0))], non_sequences=c)
            loglkl = tmp_loglkl.dimshuffle((1,0,2)).reshape((self.gen_contexts.shape[0],voc_size*beam_size))

            # select the |beam_size| best hypothesis;
            # first select arguments of the best hypothesis, than use the arguments to pick best y_predictions, states at i and log likelihoods
            [_, lkl_max, lkl_argmax],_  = theano.scan(fn=find_max, outputs_info=[loglkl + hypo_mask[T.minimum(iter,1)], None, None], n_steps=beam_size)
            pred_y_i = lkl_argmax.dimshuffle(1,0)%voc_size
            s_i = tmp_s_i.dimshuffle((1,0,2))[T.arange(self.gen_contexts.shape[0])[:,np.newaxis],lkl_argmax.dimshuffle(1,0)/voc_size]
            log_lkl = lkl_max.dimshuffle((1,0))
            sel_all_pred_y = all_pred_y_im1[T.arange(self.gen_contexts.shape[0])[:,np.newaxis],lkl_argmax.dimshuffle(1,0)/voc_size,:]
            all_pred_y = T.concatenate([sel_all_pred_y[:,:,1:], pred_y_i[:,:,np.newaxis]], axis=2)

            return [pred_y_i, s_i, log_lkl, all_pred_y, iter+1]

        # generate sequences of the predicted length, one word across all the sentences, at a time
        [_, _, log_lkl, all_pred_y, _], _ = theano.scan(fn=generate, outputs_info=[T.alloc(y0, self.gen_contexts.shape[0],beam_size), T.alloc(s0, self.gen_contexts.shape[0], beam_size, hidden_size), T.alloc(ymax0, self.gen_contexts.shape[0], beam_size), T.alloc(y0, self.gen_contexts.shape[0],beam_size,self.max_len), 0], non_sequences=self.gen_contexts, n_steps=self.max_len)
        self.gen_sequences = all_pred_y.dimshuffle((1, 0, 2, 3))
        self.gen_nll = log_lkl.dimshuffle((1, 0, 2))




class SimplRNN(object):
    '''
    RNN model joinning encoder and decoder.
    '''
    def __init__(self, voc_main, voc_simple, voc_main_emb, voc_simple_emb, h_size, s_size, al_size, enc_l2, dec_l2, lr_init, lr_decay, mom, batch_size, beam_size, params_file=None):
        '''
        Initialization of shared parameters for data and training configuration.

        :param voc_main: main vocabulary
        :param voc_simple: simple vocabulary
        :param voc_main_emb: embeddings matrix P
        :param voc_simple_emb: embeddings matrix P (in our case same as voc_simple_emb)
        :param h_size: size of the encoder hidden layer
        :param s_size: size of the decoder hidden layer
        :param al_size: size of the alignment layer
        :param enc_l2: L2 regularization weight (for encoder parameters)
        :param dec_l2: L2 regularization weight (for decoder parameters)
        :param lr_init: initial learning rate
        :param lr_decay: decay coefficient for learning rate
        :param mom: the momentum coefficient
        :param batch_size: size of the mini-batch
        :param beam_size: beam size for n-best search
        :param params_file: if parameters shall be loaded from a file
        :return:
        '''
        logging.debug('RNN initializing...')

        self.lr_decay = lr_decay
        self.voc_main = voc_main
        self.voc_simple = voc_simple
        self.batch_size = batch_size

        # shared variables where training/validation and test data-sets is stored
        self.train_main = theano.shared(name='train_main', value=np.array([[]], dtype=theano.config.floatX))
        self.train_main_len = theano.shared(name='train_main_len', value=np.array([], dtype=theano.config.floatX))
        self.train_simple = theano.shared(name='train_simple', value=np.array([[]], dtype=theano.config.floatX))
        self.train_simple_mask = theano.shared(name='train_simple_mask', value=np.array([[]], dtype=theano.config.floatX))
        self.train_simple_len = theano.shared(name='train_simple_len', value=np.array([], dtype=theano.config.floatX))
        self.train_simple_next = theano.shared(name='train_simple_next', value=np.array([[]], dtype=theano.config.floatX))

        self.test_main = theano.shared(name='test_main', value=np.array([[]], dtype=theano.config.floatX))
        self.test_main_len = theano.shared(name='test_main_len', value=np.array([], dtype=theano.config.floatX))
        self.test_simple = theano.shared(name='test_simple', value=np.array([[]], dtype=theano.config.floatX))
        self.test_simple_len = theano.shared(name='test_simple_len', value=np.array([], dtype=theano.config.floatX))

        # shared variables for training parameters
        self.mom = theano.shared(value=np.cast[theano.config.floatX](mom))
        self.lr = theano.shared(value=np.cast[theano.config.floatX](lr_init))
        self.decay = T.dscalar()

        self.model_desc = 'me%ih%ise%is%i' % (voc_main_emb.shape[1], h_size, voc_simple_emb.shape[1], s_size)

        encoder = RecEncoder(voc_main_emb, h_size)
        decoder = RecDecoder(encoder.encoding, voc_simple.size(), voc_simple_emb, h_size, s_size, beam_size)
        self.params = encoder.params + decoder.params
        self.deltas = encoder.deltas + decoder.deltas

        # loading parameters if file specified
        if params_file is not None:
            logging.info('Loading parameters: %s' % (params_file))
            self.loadParams(params_file)

        # indices to slice data-set
        idx_from = T.iscalar()
        idx_to = T.iscalar()
        true_next = T.dmatrix()

        # sym. expression for negative log likelihood of the full mini-batch
        sent_nll = -T.mean(T.sum(T.log(decoder.prob_next[T.arange(decoder.input_batch.shape[0])[:, np.newaxis], T.arange(decoder.input_batch.shape[1]), T.cast(true_next, 'int32')] + 0.000000001), axis=1))

        # L2 regularization
        # avoid biases from regularization
        enc_l2_reg = enc_l2 * T.sum([T.sum(p ** 2) for p in encoder.params[:-1]])
        dec_l2_reg = dec_l2 * T.sum([T.sum(p ** 2) for p in decoder.params[:-2]])
        cost = sent_nll + enc_l2_reg + dec_l2_reg

        # computing gradients from the cost
        # gradient clipping (-10.0, 10.0) to avoid exploding gradient
        sent_grad = T.grad(gradient.grad_clip(cost, -10.0, 10.0), self.params)

        # Nesterov accelerated gradient
        params_updates = []
        for p, gp, d in zip(self.params, sent_grad, self.deltas):
            # compute updates
            new_delta = -self.lr * gp + mom * d
            # update parameters by deltas
            # and then add extra term (mom * new_delta) ... this is to allow to compute grad_nll(params + (mom * new_delta)) given by Nesterov
            # this also means that before we even apply the new update, we need to first revert the extra term added at the previous iteration
            params_updates.append((p, (p - (self.mom*d)) + new_delta + (self.mom*new_delta)))
            params_updates.append((d, new_delta))
        params_updates = OrderedDict(params_updates)

        # theano function train, inputs are indices to the data-set variables
        # returns NLL of entire sentences
        # at the end update the shared variables of parameters by calculated deltas
        self.train_fn = theano.function(
            inputs=[idx_from, idx_to],
            outputs=sent_nll,
            updates=params_updates,
            givens={
                encoder.input_batch: self.train_main[idx_from:idx_to],
                decoder.input_batch: self.train_simple[idx_from:idx_to],
                decoder.input_mask: self.train_simple_mask[idx_from:idx_to],
                true_next: self.train_simple_next[idx_from:idx_to]
            })

        # function to decay the learning rate
        self.decay_learning_rate = theano.function(inputs=[self.decay], outputs=self.lr, updates={self.lr: self.lr * self.decay})

        # function to compute the matrix of encodings of the input sentences
        # used at testing and validation
        self.encode_test_fn = theano.function(
            inputs=[],
            outputs=encoder.encoding,
            givens={
                encoder.input_batch: self.test_main
            }
        )

        # function that generates sequences of specified length
        self.generate_fn = theano.function(inputs=[decoder.gen_contexts, decoder.max_len], outputs=[decoder.gen_sequences, decoder.gen_nll])


    def train(self, data_set, n_epochs):
        '''
        Train over n epochs over all mini-batches. Update parameters after each mini-batch.
        After each finished epoch, decay the learning and save the current parameters.

        :param data_set: training and validation data sets
        :param n_epochs: number of epoch for training
        :return:
        '''
        logging.info('Starting training')

        res_nll = []
        res_bleu = []
        res_val_nll = []

        for n in range(n_epochs):
            it_nll = 0
            it_bleu = 0
            nll_cum = 0
            bleu_cum = 0
            val_nll_cum = 0

            for b, batch in enumerate(data_set.train_batches):
                tr_main = np.array(batch[0], dtype=theano.config.floatX)
                tr_simple, tr_simple_mask, tr_simple_len = data_set.pad_with_mask(batch[1])
                tr_simple_next, _ = data_set.pad_with_lengths(batch[2])

                self.train_main.set_value(tr_main, borrow=True)
                self.train_simple.set_value(tr_simple, borrow=True)
                self.train_simple_mask.set_value(tr_simple_mask, borrow=True)
                self.train_simple_len.set_value(tr_simple_len, borrow=True)
                self.train_simple_next.set_value(tr_simple_next, borrow=True)

                nll = self.train_fn(0, self.batch_size)
                logging.info('Epoch=%i, batch=%i/%i: nll=%.5f' % (n, b+1, len(data_set.train_batches), nll))
                it_nll += 1
                nll_cum += nll

                # validate after 100th batch or at the end of the epoch
                if (b+1) % 100 == 0 or (b+1) == len(data_set.train_batches):
                    # revert extra term from the parameters (after the last update)
                    for param, delta in zip(self.params, self.deltas):
                        param.set_value(param.get_value(borrow=True) - (self.mom.get_value(borrow=True)*delta.get_value(borrow=True)), borrow=True)

                    bleu, val_nll = self.validate(data_set)
                    logging.info('BLEU=%.5f, Validation NLL=%.5f' % (bleu, val_nll))
                    it_bleu += 1
                    bleu_cum += bleu
                    val_nll_cum += val_nll

                    # revert back the reverting from above
                    for param, delta in zip(self.params, self.deltas):
                        param.set_value(param.get_value(borrow=True) + (self.mom.get_value(borrow=True)*delta.get_value(borrow=True)), borrow=True)


            self.decay_learning_rate(self.lr_decay)
            # this is for the statistics that are printed at the end
            res_nll.append(nll_cum/it_nll)
            res_bleu.append(bleu_cum/it_bleu)
            res_val_nll.append(val_nll_cum/it_bleu)

        # after the training finished, remove the extra term from the parameters, as this was necessary only during training to compute the special Netserov gradient
        for param, delta in zip(self.params, self.deltas):
            param.set_value(param.get_value(borrow=True) - (self.mom.get_value(borrow=True)*delta.get_value(borrow=True)), borrow=True)

        # log the statistics about training
        path = 'param/model_%s_%s.pickle' % (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), self.model_desc)
        logging.info('Saving parameters: %s' % (path))
        logging.info('Results:')
        logging.info('Avg. NLL per epoch: ' + str(res_nll))
        logging.info('Avg. BLEU per epoch: ' + str(res_bleu))
        logging.info('Avg. log-likelihood per epoch: ' + str(res_val_nll))
        self.saveParams(path)


    def validate(self, data_set):
        '''
        Validate on the validation set, the BLEU and log likelihood.
        :param data_set: validation data set
        :return:
        '''
        logging.info('Validating...')
        it_bleu = 0
        bleu = 0.0
        avg_log_lkl = 0.0

        for b, batch in enumerate(data_set.test_batches):
            te_main, te_main_len = data_set.pad_with_lengths(batch[0])
            te_simple, te_simple_mask, te_simple_len = data_set.pad_with_mask(batch[1])

            self.test_main.set_value(te_main, borrow=True)
            self.test_main_len.set_value(te_main_len, borrow=True)
            self.test_simple.set_value(te_simple, borrow=True)
            self.test_simple_len.set_value(te_simple_len, borrow=True)

            encodings = self.encode_test_fn()
            # these are the predefined lengths taken from the validation true simple sentences
            lengths = self.test_simple_len.get_value(borrow=True) - 1
            generated, gen_logl = self.generate_fn(encodings, max(lengths))
            log_likelihoods = []
            for i, logl in enumerate(gen_logl):
                log_likelihoods.append(logl[max(lengths[i]-1, 0)][0])
            avg_log_lkl += np.mean(log_likelihoods)
            generated, log_lkl = self.generate_fn(encodings, max(lengths))
            for i, seq in enumerate(generated):
                prediction = seq[max(lengths[i]-1, 0)][0]
                prediction = prediction[len(prediction) - max(lengths[i], 1):]
                it_bleu += 1
                bleu += evaluation.bleu(prediction.tolist(), self.test_simple.get_value(borrow=True)[i][:int(self.test_simple_len.get_value(borrow=True)[i])][1:].tolist())

        return [bleu/it_bleu, avg_log_lkl/len(data_set.test_batches)]

    # translate a sequence of indices into sequence of actual words, given the vocabulary
    def translate(self, prediction):
        return ' '.join([self.voc_simple.words[int(i)] for i in prediction])

    def translate_main(self, prediction):
        return ' '.join([self.voc_main.words[int(i)] for i in prediction])

    def test(self, data_set):
        '''
        Test the model on the test set.
        Output printed into several files.
        :param data_set:
        :return:
        '''
        logging.info('Testing...')
        bleu_list = []
        true_bleu_list = []
        src_len = []

        path = 'output/%s_model_%s.txt' % (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), self.model_desc)
        path2 = 'output/raw_%s_model_%s.txt' % (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), self.model_desc)
        logging.info('Output file: %s' % (path))
        file = open(path, 'w')
        file2 = open(path2, 'w')

        for b, batch in enumerate(data_set.test_batches):

            te_main, te_main_len = data_set.pad_with_lengths(batch[0])
            te_simple, te_simple_len = data_set.pad_with_lengths(batch[1])

            self.test_main.set_value(te_main, borrow=True)
            self.test_main_len.set_value(te_main_len, borrow=True)
            self.test_simple.set_value(te_simple, borrow=True)
            self.test_simple_len.set_value(te_simple_len, borrow=True)

            encodings = self.encode_test_fn()
            lengths = self.test_simple_len.get_value(borrow=True) - 1
            generated, log_lkl = self.generate_fn(encodings, max(lengths))
            for i, seq in enumerate(generated):
                prediction = seq[max(lengths[i]-1, 0)][0]
                prediction = prediction[len(prediction) - max(lengths[i], 1):]
                refference = self.test_simple.get_value(borrow=True)[i][:int(self.test_simple_len.get_value(borrow=True)[i])]
                bleu = evaluation.bleu(prediction.tolist(), refference[1:].tolist())
                bleu_list.append(bleu)
                true_bleu = evaluation.bleu(self.translate(prediction).split(), batch[4][i].split()[1:])
                true_bleu_list.append(true_bleu)
                src_len.append(len(batch[3][i][:-1].split()))
                file.write('%.5f | %.5f | %s | %s | %s | %s\n' % (bleu, true_bleu, self.translate(prediction), self.translate(refference), batch[3][i][:-1], batch[4][i][:-1]))
                file2.write('%s\n' % (self.translate(prediction)))
        file.close()
        file2.close()

        bleu_lens = {}
        for b,l in zip(bleu_list, src_len):
            if l in bleu_lens.keys():
                bleu_lens[l].append(b)
            else:
                bleu_lens[l] = [b]
        avg_bleus = {}
        for l in bleu_lens.keys():
            avg_bleus[l] = sum(bleu_lens[l])/len(bleu_lens[l])

        path3 = 'output/results_%s_model_%s.txt' % (datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), self.model_desc)
        file3 = open(path3, 'w')
        file3.write('BLEU: \n')
        file3.write(str(bleu_list))
        file3.write('\nSource sentence lengths: \n')
        file3.write(str(src_len))
        file3.write('\nAvg. BLEU: \n' + str(sum(bleu_list)/len(bleu_list)))
        file3.write('\nAvg. BLEU per source sentence length: \n')
        file3.write(str(avg_bleus))
        file3.write('\nAvg. True BLEU: \n' + str(sum(true_bleu_list)/len(true_bleu_list)))
        file3.close()

    def saveParams(self, path):
        file = open(path, 'wb')
        for param in self.params:
            cPickle.dump(param.get_value(borrow=True), file, -1)
        cPickle.dump(self.lr.get_value(borrow=True), file, -1)
        file.close()

    def loadParams(self, path):
        file = open(path, 'rb')
        for param in self.params:
            param.set_value(cPickle.load(file), borrow=True)
        self.lr.set_value(cPickle.load(file), borrow=True)
        file.close()



