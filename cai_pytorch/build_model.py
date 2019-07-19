#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from torch import optim
# from .build_corpus import *
from .models import *
from .build_corpus import normalizeString
import torch.nn as nn
import random
import itertools


class Model():
    
    def __init__(self, pp):

        self.pp = pp
        

    def train_model(self, voc, pairs):
        self.voc = voc
        self.pairs = pairs
        self.model_name = 'cb_model'
    
        print('Building encoder and decoder ...')
        
        # Initialize word embeddings
        self.embedding = nn.Embedding(self.voc.num_words, self.pp.hidden_size)
    
        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.pp.hidden_size,
                                  self.embedding, 
                                  self.pp.encoder_n_layers,
                                  self.pp.dropout)
        self.decoder = LuongAttnDecoderRNN(self.pp.attn_model,
                                           self.embedding, 
                                           self.pp.hidden_size,
                                           self.voc.num_words, 
                                           self.pp.decoder_n_layers,
                                           self.pp.dropout)
    
        # Use appropriate device
        self.encoder = self.encoder.to(self.pp.device)
        self.decoder = self.decoder.to(self.pp.device)
        print('Encoder/Decoder Models built and ready to go!')
        
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()
    
        # Initialize optimizers
        print('Building optimizers ...')
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.pp.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), 
                                       lr=self.pp.learning_rate * self.pp.decoder_learning_ratio)
    
        # TODO what is this for?
        # if loadFilename:
        #     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        #     decoder_optimizer.load_state_dict(decoder_optimizer_sd)
    
        # Run training iterations
        print("Starting Training!")
        self.trainIters()
        return self.encoder, self.decoder


    def train(self, input_variable, lengths, target_variable, mask, max_target_len):

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(self.pp.device)
        lengths = lengths.to(self.pp.device)
        target_variable = target_variable.to(self.pp.device)
        mask = mask.to(self.pp.device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[self.pp.SOS_token for _ in range(self.pp.batch_size)]])
        decoder_input = decoder_input.to(self.pp.device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.pp.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.pp.batch_size)]])
                decoder_input = decoder_input.to(self.pp.device)
                # Calculate and accumulate loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.pp.clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.pp.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals


    def trainIters(self):

        # Load batches for each iteration
        training_batches = [self.batch2TrainData([random.choice(self.pairs) for _ in range(self.pp.batch_size)])
                          for _ in range(self.pp.n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0

        #TODO how to continue training from some checkpoint
        # if loadFilename:
        #     start_iteration = checkpoint['iteration'] + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, self.pp.n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self.train(input_variable, lengths, target_variable, mask, max_target_len)
            print_loss += loss

            # Print progress
            if iteration % self.pp.print_every == 0:
                print_loss_avg = print_loss / self.pp.print_every
                print("Iteration: {}; Percent complete: {:.1f}%; "
                      "Average loss: {:.4f}".format(iteration, iteration / self.pp.n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % self.pp.save_every == 0):
                save_dir = os.path.join("data", "save")
                directory = os.path.join(save_dir, self.pp.model_name, self.pp.corpus_name,
                                         '{}-{}_{}'.format(self.pp.encoder_n_layers, self.pp.decoder_n_layers,
                                                           self.pp.hidden_size))

                # directory = os.path.join('static/local-data/', self.pp.corpus_name, self.pp.model_name,
                #                          '{}-{}_{}'.format(self.pp.encoder_n_layers,
                #                                            self.pp.decoder_n_layers,
                #                                            self.pp.hidden_size))
                print(directory)

                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'iteration': iteration,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': self.voc.__dict__,
                    'embedding': self.embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


    def zeroPadding(self, l):
        return list(itertools.zip_longest(*l, fillvalue=self.pp.PAD_token))


    def binaryMatrix(self, l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == self.pp.PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m


    # Returns padded input sequence tensor and lengths
    def inputVar(self, l):
        indexes_batch = [indexesFromSentence(sentence, self.voc, self.pp.EOS_token) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths


    # Returns padded target sequence tensor, padding mask, and max target length
    def outputVar(self, l):
        indexes_batch = [indexesFromSentence(sentence, self.voc, self.pp.EOS_token) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        mask = self.binaryMatrix(padList)
        mask = torch.ByteTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len


    # Returns all items for a given batch of pairs
    def batch2TrainData(self, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = self.inputVar(input_batch)
        output, mask, max_target_len = self.outputVar(output_batch)
        return inp, lengths, output, mask, max_target_len


    # Define Training Procedure
    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self.pp.device)
        return loss, nTotal.item()


    def evaluate(self, searcher, voc, sentence):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(sentence, voc, self.pp.EOS_token)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.pp.device)
        lengths = lengths.to(self.pp.device)

        # Decode sentence with searcher
        tokens, scores = searcher.forward(input_batch, lengths, self.pp.MAX_LENGTH)

        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words


    def evaluateInput(self, searcher, voc):

        input_sentence = ''
        while (1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = self.evaluate(searcher, voc, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")


    def evaluateInputSapCai(self, input, searcher, voc):
        input_sentence = input
        while (1):
            try:

                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = self.evaluate(searcher, voc, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                return output_words

            except KeyError:
                return "Error: Encountered unknown word."


def indexesFromSentence(sentence, voc, EOS_token):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

if __name__ == "__main__":
    pass
    
    
    # # Example for validation
    # small_batch_size = 5
    # batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    # input_variable, lengths, target_variable, mask, max_target_len = batches
    # 
    # print("input_variable:", input_variable)
    # print("lengths:", lengths)
    # print("target_variable:", target_variable)
    # print("mask:", mask)
    # print("max_target_len:", max_target_len)

    