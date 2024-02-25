import numpy as np
from keras.utils import pad_sequences
from keras.layers import Embedding
import re

# utils
def cleanText(text):
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", " ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()'\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

def cleanTexts(texts):
    return [cleanText(text) for text in texts]

def getTokenizerDicts(tokenizer, numWords):
    wordToIdx = {}
    idxToWord = {}
    for k, v in tokenizer.word_index.items():
        if v < numWords:
            wordToIdx[k] = v
            idxToWord[v] = k
        if v >= numWords - 1:
            continue
    return wordToIdx, idxToWord

def padding(sequences, maxLen):
    return pad_sequences(
        sequences,
        maxlen = maxLen,
        dtype = 'int',
        padding = 'post',
        truncating = 'post'
    )

def getDecoderOutput(decoderInput, maxLen):
    decoderOutput = np.zeros((len(decoderInput), maxLen), dtype='float32')
    for i, seq in enumerate(decoderInput):
        decoderOutput[i] = np.append(seq[1:], 0.)

    return decoderOutput

def getEmbeddingLayer(
    GLOVE_FILE,
    numWords,
    embeddingDimension,
    maxLen,
    wordToIdx,
    name = 'embeddingLayer'
):
    embeddingsIndex = {}
    with open(GLOVE_FILE) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = coefs
        f.close()

    embeddingMatrix = np.zeros((len(wordToIdx) + 1, embeddingDimension), dtype='float32')
    for word, i in wordToIdx.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return Embedding(
        input_dim = numWords,
        output_dim = embeddingDimension,
        input_length = maxLen,
        weights = [embeddingMatrix],
        name = name,
        trainable = False,
        mask_zero = True
    )