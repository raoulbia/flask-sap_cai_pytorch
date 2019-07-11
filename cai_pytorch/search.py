import torch
import torch.nn as nn
from .build_model import Model

######################################################################
    # Define Evaluation
    # **Computation Graph:**
    #
    #    1) Forward input through encoder model.
    #    2) Prepare encoder's final hidden layer to be first hidden input to the decoder.
    #    3) Initialize decoder's first input as SOS_token.
    #    4) Initialize tensors to append decoded words to.
    #    5) Iteratively decode one word token at a time:
    #        a) Forward pass through decoder.
    #        b) Obtain most likely word token and its softmax score.
    #        c) Record token and score.
    #        d) Prepare current token to be next decoder input.
    #    6) Return collections of word tokens and scores.
    #


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, pp):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pp = pp

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.pp.device, dtype=torch.long) * self.pp.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.pp.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.pp.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores






if __name__ == "__main__":
    pass
