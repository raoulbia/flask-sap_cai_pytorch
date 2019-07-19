# -*- coding: utf-8 -*-


def test_build_model_params(model):
    print(model.project_params.n_iteration)

def test_build_model(model, vocabulary):
    pairs = [['this is the end my friend', 'what should i do next?']]

    for pair in pairs:
        vocabulary.addSentence(pair[0])
        vocabulary.addSentence(pair[1])

    model.project_params.n_iteration = 1
    model.project_params.print_every = 1
    model.project_params.save_every = 1

    encoder, decoder = model.train_model(vocabulary, pairs)
    print(type(encoder))