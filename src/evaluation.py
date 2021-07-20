'''
Evaluation file. Provides only BLEU.
This BLEU is used only for development. The final tests use the mteval.pl tool for BLEU.
'''
__author__ = 'Alexandra'


import numpy as np


def bleu(prediction, refference, verbose=False):

    if verbose==True:
        print prediction
        print refference
    # Brevity Penalty:
    #   1 if predicted sentence is longer than the referrence
    #   otherwise exp(1 - (ref_len/pred_len))
    if len(prediction) > len(refference):
        bp = 1.0
    else:
        bp = np.exp(1 - (float(len(refference))/len(prediction)))
    pn = 0
    ngram = min(4, len(refference), len(prediction))
    for n in range(ngram):
        pn += (1.0/ngram)*log_n_gram_precision(n + 1, prediction, refference)
    return bp * np.exp(pn)


def log_n_gram_precision(n, pred, ref):
    ref_ngrams = []
    for i in range(len(ref) - n + 1):
        ref_ngrams.append(ref[i:i+n])
    rel = 0.0
    for i in range(len(pred) - n + 1):
        n_gram = pred[i:i+n]
        if n_gram in ref_ngrams:
            rel += 1
    precision = (rel + 0.0001)/((len(pred) - n + 1) + 0.0001)
    return np.log(precision)
