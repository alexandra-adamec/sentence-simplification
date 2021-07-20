'''
Utility file to load and preprocess data-sets.
'''

__author__ = 'Alexandra'


import theano
import numpy as np
import cPickle
import zipfile
import logging
import re
import operator
import copy
import random


def load_embeddings(filepath):
    '''
    Load the Mikolov's embeddings from a given file.
    '''
    logging.info('Loading embeddings: %s' % (filepath))
    file = open(filepath, 'rb')
    emb = {}
    first = True
    for line in file:
        if first == True:
            first = False
            continue

        tokens = line.split()
        seq = []
        for t in tokens[1:]:
            seq.append(float(t))
        emb[tokens[0].lower()] = seq
    file.close()
    return emb

def load_vocabulary(filepath):
    '''
    Load vocabulary from a given Pickle file.
    '''
    logging.info('Loading vocabulary: %s' % (filepath))
    file = open(filepath, 'rb')
    voc = cPickle.load(file)
    file.close()
    return voc


class Vocabulary(object):
    '''
    Vocabulary class, initialize vocabulary with two special symbols: NUM and LB (left boundary).
    Provides operations to truncate the vocabulary, either by size or min. words frequency.
    '''

    def __init__(self):
        self.init_sets()
        self.freq = {'LB': 100000, 'NUM': 100000}
        self.emb = []

    def init_sets(self):
        self.words = ['LB', 'NUM']
        self.index = {'LB': 0, 'NUM': 1}

    def append(self, new_words):
        for w in new_words:
            if w not in self.words:
                self.words.append(w)
                self.index[w] = len(self.words) - 1
                self.freq[w] = 1
            else:
                self.freq[w] += 1

    def size(self):
        return len(self.words)

    def truncate_by_size(self, max_size):
        truncated = sorted(self.freq.items(), key=operator.itemgetter(1), reverse=True)[:max_size]
        self.init_sets()
        for w, _ in truncated:
            if w != 'LB' and w != 'NUM':
                self.words.append(w)
                self.index[w] = len(self.words) - 1

    def truncate_by_threshold(self, threshold):
        sorted_freqs = sorted(self.freq.items(), key=operator.itemgetter(1), reverse=True)
        truncated = [word for word in sorted_freqs if word[1] >= threshold]
        self.init_sets()
        for w, _ in truncated:
            if w != 'LB' and w != 'NUM':
                self.words.append(w)
                self.index[w] = len(self.words) - 1


    def get_index(self, word):
        try:
            if re.match("^\(*?\d+?[\.,\-]?\d*?[\)(th)]*?$", word) is not None:
                return 1
            return self.index[word]
        except KeyError:
            return None

    def add_embeddings(self, embeddings):
        emb_list = [[0.0] * 640, [1.0] * 640, [-1.0] * 640]
        for w in self.words[3:]:
            emb_list.append(embeddings[w])
        self.emb = np.asarray(emb_list, dtype=np.float32)

    def save(self, path):
        file = open(path, 'wb')
        cPickle.dump(self, file, -1)
        file.close()



class DataSet(object):
    '''
    Data-set class: stores preprocessed data-set organized in mini-batches.
    Loads data from the given train,valid resp. test files.
    Converts sentences with words into sequences of indices.
    Organizes into mini-batches by length and then shuffles the minibatches.
    '''
    def __init__(self, batch_size, data_dir, corpus_name, rev=False, testing=False, voc_name='voc'):
        logging.info('Corpus used: %s' % data_dir + corpus_name)
        dir = zipfile.ZipFile(data_dir + corpus_name)
        dir.extract(voc_name+'.main', path=data_dir)
        dir.extract(voc_name+'.simple', path=data_dir)
        self.voc_main = load_vocabulary(data_dir + voc_name + '.main')
        self.voc_simple = load_vocabulary(data_dir + voc_name + '.simple')

        if testing == False:
            self.load_train_sets(batch_size, dir, 'train.main', 'train.simple', rev)
            self.load_test_sets(batch_size, dir, 'valid.main', 'valid.simple', rev)
        else:
            self.load_test_sets(batch_size, dir, 'test.main', 'test.simple', rev)
        dir.close()


    def load_as_idx_seqs(self, dir, filename, vocabulary, rev=False):
        logging.info('Loading sentences: %s' % filename)
        f = dir.open(filename, 'r')
        seqs = []
        orig = []
        for line in f:
            orig.append(line)
            words = line.split()
            idx_seq = []
            for word in words:
                idx = vocabulary.get_index(word)
                if idx is not None:
                    idx_seq.append(idx)
            if rev == True:
                seqs.append([x for x in reversed(idx_seq)])
            else:
                seqs.append(idx_seq)
        f.close()
        return seqs, orig

    def load_as_idx_seqs_with_next(self, dir, filename, vocabulary):
        logging.info('Loading sentences: %s' % filename)
        f = dir.open(filename, 'r')
        seqs = []
        next_seqs = []
        orig = []
        for line in f:
            orig.append(line)
            words = line.split()
            idx_seq = []
            for word in words:
                idx = vocabulary.get_index(word)
                if idx is not None:
                    idx_seq.append(idx)
            seqs.append(idx_seq[:-1])
            next_seqs.append(idx_seq[1:])
        f.close()
        return seqs, orig, next_seqs

    def sort_by_length(self, batch_size, main_seq, main_orig, simple_seq, simple_orig, simple_next_seq=None):
        if simple_next_seq is None:
            zipped = [(m, s, mo, so) for m, s, mo, so in zip(main_seq, simple_seq, main_orig, simple_orig)]
        else:
            zipped = [(m, s, sn, mo, so) for m, s, sn, mo, so in zip(main_seq, simple_seq, simple_next_seq, main_orig, simple_orig)]
        zipped.sort(key=lambda x: len(x[0]))
        batches = []

        cur_len = 1
        # new_main_seq, new_simple_seq, new_simple_seq_next, new_main_orig, new_simple_orig
        cur_batch = [[],[],[],[],[]]

        for sent in zipped:
            if len(sent[0]) == 0:
                continue
            if simple_next_seq is None and len(sent[1]) <= 1:
                continue
            if simple_next_seq is not None and len(sent[2]) == 0:
                continue

            if len(cur_batch[0]) >= batch_size:
                batches.append(cur_batch)
                cur_batch = [[],[],[],[],[]]
            if len(sent[0]) > cur_len:
                if len(cur_batch[0]) > 0:
                    batches.append(copy.deepcopy(cur_batch))
                    cur_batch = [[],[],[],[],[]]
                cur_len = len(sent[0])

            cur_batch[0].append(sent[0])
            cur_batch[1].append(sent[1])
            if simple_next_seq is not None:
                cur_batch[2].append(sent[2])
                cur_batch[3].append(sent[3])
                cur_batch[4].append(sent[4])
            else:
                cur_batch[3].append(sent[2])
                cur_batch[4].append(sent[3])
        batches.append(cur_batch)

        random.seed(234)
        random.shuffle(batches)
        return batches

    def load_train_sets(self, batch_size, dir, train_main_fn, train_simple_fn, rev):
        main_idxs, main_orig = self.load_as_idx_seqs(dir, train_main_fn, self.voc_main, rev)
        simple_idxs, simple_orig, simple_idxs_next = self.load_as_idx_seqs_with_next(dir, train_simple_fn, self.voc_simple)
        self.train_batches = self.sort_by_length(
            batch_size,
            main_idxs, main_orig,
            simple_idxs, simple_orig, simple_idxs_next
        )

    def load_test_sets(self, batch_size, dir, test_main_fn, test_simple_fn, rev):
        main_idxs, main_orig = self.load_as_idx_seqs(dir, test_main_fn, self.voc_main, rev)
        simple_idxs, simple_orig, = self.load_as_idx_seqs(dir, test_simple_fn, self.voc_simple)
        self.test_batches = self.sort_by_length(
            batch_size,
            main_idxs, main_orig,
            simple_idxs, simple_orig, None
        )

    def pad_with_lengths(self, orig_seqs):
        seqs = copy.deepcopy(orig_seqs)
        lengths = map(len, seqs)
        max_len = max(lengths)
        map(lambda s: s.extend([0]*(max_len-len(s))), seqs)
        return np.array(seqs, theano.config.floatX), np.array(lengths, theano.config.floatX)

    def pad_with_mask(self, orig_seqs):
        seqs = copy.deepcopy(orig_seqs)
        lengths = map(len, seqs)
        max_len = max(lengths)
        map(lambda s: s.extend([0]*(max_len-len(s))), seqs)
        mask = map(lambda x: [1]*x + [0]*(max_len-x), lengths)
        return np.array(seqs, theano.config.floatX), np.array(mask, theano.config.floatX), np.array(lengths, theano.config.floatX)



if __name__ == "__main__":
    d = DataSet(100, 'data/', 'stats.zip')
    print d.voc_main.size()
    print d.voc_simple.size()
   




