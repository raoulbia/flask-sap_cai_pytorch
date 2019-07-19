# -*- coding: utf-8 -*-

from pprint import pprint

def test_params(params):
    print(params.n_iteration)
    # assert params.n_iteration == 4000


def test_voc_params(vocabulary):
    pp = vocabulary.project_params
    # pprint(vars(project_params))
    print(pp.n_iteration)

def test_voc_sentence(vocabulary):

    sentence = 'This is the end my friend'
    vocabulary.addSentence(sentence)
    print(vocabulary.word2index)
    print(vocabulary.index2word)
    print(vocabulary.word2count)
    print(vocabulary.num_words)