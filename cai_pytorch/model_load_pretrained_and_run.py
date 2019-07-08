import torch
import torch.nn as nn
from model import * #EncoderRNN, LuongAttnDecoderRNN, Voc, GreedySearchDecoder, evaluateInput

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
device = torch.device("cpu")

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000

loadFilename = '/home/vagrant/vmtest/cai_pytorch/cai_pytorch/static/4000_checkpoint.tar'


# If loading on same machine the model was trained on
# checkpoint = torch.load(loadFilename)

# If loading a model trained on GPU to CPU
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']

voc = Voc("cornell movie-dialogs corpus")
voc.__dict__ = checkpoint['voc_dict']
print(checkpoint['voc_dict'].keys())
# voc = checkpoint['voc_dict']



print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)

embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)


encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()


# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)