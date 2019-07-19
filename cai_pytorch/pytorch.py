#!/usr/bin/env python3

import torch
from .build_corpus import BuildCorpus
from .build_model import Model
from .search import GreedySearchDecoder
from .utils_load_model import LoadModel

class ProjectParams():
    def __init__(self):

        # Configure models
        self.corpus_name = "cornell movie-dialogs corpus"
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        # self.attn_model = 'general'
        # self.attn_model = 'concat'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64
        self.device = torch.device("cpu")

        self.loadFilename = '/home/ubuntu/cai_pytorch/cai_pytorch/static/10000_checkpoint.tar'

        # Configure training/optimization
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.n_iteration = 4000
        self.print_every = 500
        self.save_every = 500

        self.attn_model = 'dot'
        # self.attn_model = 'general'
        # self.attn_model = 'concat'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64

        # Set checkpoint to load from;
        self.checkpoint_iter = 4000

        # Default word tokens
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token
        self.MAX_LENGTH = 10  # Maximum sentence length to consider
        self.MIN_COUNT = 3  # Minimum word count threshold for trimming

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")


if __name__ == "__main__":
    pp = ProjectParams()

    # build model
    voc, pairs = BuildCorpus(pp).build_corpus_movies()
    encoder, decoder = Model(pp).train_model(voc, pairs)


    # load pre-trained model
    # encoder, decoder, voc = LoadModel(project_params).load_model()


    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, pp)
    # Begin chatting
    Model(pp).evaluateInput(searcher, voc)




