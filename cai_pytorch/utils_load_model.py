#!/usr/bin/env python3
import torch
import torch.nn as nn
# import torch.utils.model_zoo
from .models import EncoderRNN, LuongAttnDecoderRNN
from .build_corpus import Voc


class LoadModel():
    def __init__(self, pp):
        self.pp = pp

    def load_model(self):

        device = self.pp.device

        # If loading on same machine the model was trained on
        # checkpoint = torch.load(loadFilename)

        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(self.pp.loadFilename, map_location=device)

        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        embedding_sd = checkpoint['embedding']

        # instantiate Voc object to access word2index
        voc = Voc(self.pp)
        voc.__dict__ = checkpoint['voc_dict']

        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']

        print('Building encoder and decoder ...Initialize word embeddings')
        embedding = nn.Embedding(voc.num_words, self.pp.hidden_size)
        embedding.load_state_dict(embedding_sd)

        print('Initialize encoder & decoder models')
        encoder = EncoderRNN(self.pp.hidden_size, embedding, self.pp.encoder_n_layers, self.pp.dropout)
        decoder = LuongAttnDecoderRNN(self.pp.attn_model, embedding, self.pp.hidden_size, voc.num_words, self.pp.decoder_n_layers, self.pp.dropout)


        encoder.load_state_dict(encoder_sd, strict=False)
        decoder.load_state_dict(decoder_sd, strict=False)

        # Use appropriate device
        encoder.to(device)
        decoder.to(device)

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()


        print('Models built and ready to go!')


        return encoder, decoder, voc

if __name__ == "__main__":
    
    pass
    # encoder, decoder, voc = load_model(loadFilename='static/4000_checkpoint.tar',
    #                                    hidden_size=500,
    #                                    encoder_n_layers=2,
    #                                    decoder_n_layers=2,
    #                                    dropout=0.1,
    #                                    attn_model='dot'
    #                                    )
    # # print(voc['voc_dict'].keys())
    # # print(voc['voc_dict']['word2index'])
    # # print(voc.word2index)
    # print(voc.word2index['dreams'])
    #

