#!/usr/bin/env python3

import itertools
import pandas as pd
import urllib.request
import tempfile
import zipfile
from zipfile import ZipFile
import codecs, unicodedata, os, csv, re
# from pytorch import ProjectParams

class BuildCorpus():
    def __init__(self, pp):
        self.pp = pp

        # Define path to new file
        directory = 'static/local-data/cornell movie-dialogs corpus'
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.pairs_datafile = os.path.join(directory, 'formatted_movie_lines.txt')


    def build_corpus_movies(self):
        movie_lines, movie_conversations = get_remote_data()

        # Initialize lines dict, conversations list, and field ids
        lines = {}
        conversations = []

        print("\nProcessing lines corpus...")
        movie_lines = self.loadLines(movie_lines)

        print("\nLoading conversations corpus...")
        conversations = self.loadConversations(movie_conversations, movie_lines)

        pairs = extractSentencePairs(conversations)

        # write(pairs, tofile=self.pairs_datafile)

        # print("Start preparing training data ...")
        pairs = self.normalizePairs(pairs)

        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))

        print("Voc: Counting words...")
        # instantiate Voc object
        voc = Voc(self.pp, "cornell movie-dialogs corpus")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)


        ################################
        # Load/Assemble voc and pairs  #
        ################################

        # create Voc dict and pairs
        # voc, pairs = self.loadPrepareData(self.pairs_datafile)

        # Print some pairs to validate
        print("\npairs:")
        for pair in pairs[:10]:
            print(pair)

        # Trim voc and pairs
        pairs = trimRareWords(voc, pairs, self.pp.MIN_COUNT)
        return voc, pairs


    # Splits each line of the file into a dictionary of fields
    def loadLines(self, fileName):
        fields = ["lineID", "characterID", "movieID", "character", "text"]
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
            # print(lines.keys()) keys == lineID's
            print(lines['L1045'])
        return lines


    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def loadConversations(self, file_movie_conversations, movie_lines):
        fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        conversations = []
        with open(file_movie_conversations, 'r', encoding='iso-8859-1') as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # example: ['u1183', 'u1191', 'm78', "['L254006', 'L254007']\n"]

                # Extract movie_conversations_fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = eval(convObj["utteranceIDs"])

                # Add lines from the movie_lines object to the convObj
                convObj["movie_lines"] = []
                for lineId in lineIds:
                    # example movie_lines[lineId]
                    # {'lineID': 'L36930', 'text': 'Coffee?\n', 'character': 'BATEMAN', 'characterID': 'u327', 'movieID': 'm20'}

                    ###### RB restrict to one movie ############
                    if movie_lines[lineId]['movieID'] == 'm0':

                        convObj["movie_lines"].append(movie_lines[lineId])
                conversations.append(convObj)
        return conversations


    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(self, datafile):

        # print("Start preparing training data ...")
        # pairs = self.normalizePairs(datafile)

        # print("Read {!s} sentence pairs".format(len(pairs)))
        # pairs = self.filterPairs(pairs)
        # print("Trimmed to {!s} sentence pairs".format(len(pairs)))

        print("Voc: Counting words...")
        # instantiate Voc object
        voc = Voc(self.pp, "cornell movie-dialogs corpus")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        # print(voc.word2index)
        return voc, pairs


    # Read query/response pairs
    def normalizePairs(self, pairs):
        print("Read query/response pairs...")
        # Read the file and split into lines
        # lines = open(datafile, encoding='utf-8').\
        #     read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in pair] for pair in pairs]
        # print(pairs)
        return pairs

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(self, p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < self.pp.MAX_LENGTH and len(p[1].split(' ')) < self.pp.MAX_LENGTH

    # Filter pairs using filterPair condition
    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]


def get_remote_data():
    # temp_dir = tempfile.mkdtemp()
    # data_source = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    # zipname = temp_dir + '/temp.zip'
    # urllib.request.urlretrieve(data_source, zipname)
    #
    # zip_ref = zipfile.ZipFile(zipname, 'r')
    # zip_ref.extractall(temp_dir)
    # print(zip_ref)
    # print(zip_ref.printdir())
    # zip_ref.close()

    # movie_lines = temp_dir + '/cornell movie-dialogs corpus/movie_lines.txt'
    # movie_conversations = temp_dir + '/cornell movie-dialogs corpus/movie_conversations.txt' # adapt file name

    movie_lines = 'static/local-data/cornell movie-dialogs corpus/movie_lines.txt'
    movie_conversations = 'static/local-data/cornell movie-dialogs corpus/movie_conversations.txt'  # adapt file name


    return movie_lines, movie_conversations


def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # print(conversation)
        # Iterate over all the lines of the conversation
        # RB
        for i in range(0, len(conversation["movie_lines"]) - 1, 2):  # We ignore the last line (no answer for it)
            inputLine = conversation["movie_lines"][i]["text"].strip()
            targetLine = conversation["movie_lines"][i+1]["text"].strip()

            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                # print(inputLine)
                # print(targetLine, '\n')
                qa_pairs.append([inputLine, targetLine])

    qa_pairs.sort()

    # [print(pair) for pair in qa_pairs]
    return qa_pairs

    #RB dedupe
    # qa_pairs_dpd = list(qa_pairs for qa_pairs, _ in itertools.groupby(qa_pairs))
    # [print(pair) for pair in qa_pairs_dpd]
    # return qa_pairs_dpd


# Write new csv file
def write(pairs, tofile):
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    print("\nWriting newly formatted file...")
    with open(tofile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in pairs:
            writer.writerow(pair)

    # Print a sample of lines
    print("\nSample lines from newly formatted  file:")
    printLines(tofile)


class Voc():
    def __init__(self, pp, name=''):
        self.project_params = pp
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.project_params.PAD_token: "PAD",
                           self.project_params.SOS_token: "SOS",
                           self.project_params.EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        # print(sentence)
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1



    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.project_params.PAD_token: "PAD",
                           self.project_params.SOS_token: "SOS",
                           self.project_params.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


if __name__ == "__main__":
    pass