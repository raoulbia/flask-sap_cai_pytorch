from build_corpus_movies import *
from build_model import Model, GreedySearchDecoder, evaluateInput
from utils_load_model import load_model

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
# device = torch.device("cpu")

# build_corpus_movies corpus
movie_lines, movie_conversations = get_data()
voc, pairs = build_corpus_movies(movie_lines, movie_conversations)

# train and save model
Model(voc, pairs).train_model()


# load pre-trained model
loadFilename = 'static/4000_checkpoint.tar'
encoder, decoder, voc = load_model(loadFilename,
                                   hidden_size,
                                   encoder_n_layers,
                                   decoder_n_layers,
                                   dropout,
                                   attn_model
                                   )

# chat
# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)
# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc)


if __name__ == "__main__":
    

    
