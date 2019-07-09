import torch
import torch.nn as nn
# import torch.utils.model_zoo
from models import EncoderRNN, LuongAttnDecoderRNN
from build_corpus import Voc



import urllib.request
import tarfile


def load_model(loadFilename, 
               hidden_size, 
               encoder_n_layers,
               decoder_n_layers,
               dropout,
               attn_model):

    device = torch.device('cpu')

    # If loading on same machine the model was trained on
    # checkpoint = torch.load(loadFilename)
    
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=device)

    # print("checkpoint.keys()... ", checkpoint.keys())
    # print("checkpoint['voc_dict']... ", checkpoint['voc_dict'].keys())
    # print("checkpoint['voc_dict']['num_words']... ", checkpoint['voc_dict']['num_words'])
    # print("checkpoint['en']... ", checkpoint['en'].keys())
    # print("checkpoint['de']... ", checkpoint['de'].keys())
    
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    num_words = checkpoint['voc_dict']['num_words']
    
    # encoder_optimizer_sd = checkpoint['en_opt']
    # decoder_optimizer_sd = checkpoint['de_opt']
    
    
    print('Building encoder and decoder ...Initialize word embeddings')
    embedding = nn.Embedding(num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    
    print('Initialize encoder & decoder models')
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, num_words, decoder_n_layers, dropout)
    
    
    encoder.load_state_dict(encoder_sd, strict=False)
    decoder.load_state_dict(decoder_sd, strict=False)
    
    # Use appropriate device
    encoder.to(device)
    decoder.to(device)
    
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    
    # instantiate Voc object to access word2index
    voc = Voc()
    voc.__dict__ = checkpoint['voc_dict']

    print('Models built and ready to go!')
    
    
    return encoder, decoder, voc

if __name__ == "__main__":
    
    encoder, decoder, voc = load_model(loadFilename='static/4000_checkpoint.tar',
                                       hidden_size=500,
                                       encoder_n_layers=2,
                                       decoder_n_layers=2,
                                       dropout=0.1,
                                       attn_model='dot'
                                       )
    # print(voc['voc_dict'].keys())
    # print(voc['voc_dict']['word2index'])
    # print(voc.word2index)
    print(voc.word2index['dreams'])
    

