'''
This implements a Seq to Seq network (Encoder-Decoder Network), 
in which two recurrent nets work together to transform one sequence to another. 
An encoder network condenses an input sequence into a vector, 
and a decoder network unfolds that vector into a new sequence.
To improve upon this model we’ll use an attention mechanism, 
which lets the decoder learn to focus over a specific range 
of the input sequence.
'''
import numpy as np
import fire
from preprocess import Lang
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from preprocess import MAX_LENGTH, SOS_token, EOS_token
# cPickle provides faster serializability than pickle and JSON
import _pickle as cPickle 
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('processed_text_Data.save', 'rb') as f:
    input_lang, output_lang, pairs = cPickle.load(f)
    f.close()

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    # hidden state and the cell state reset to zero for every epoch regardless
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size=256, dropout_p=0.1, 
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

'''
    Saves the models to the directory Model
'''
def save_model(encoder, decoder):
    torch.save(encoder.state_dict(), f="Model/encoder.model")
    torch.save(decoder.state_dict(), f="Model/decoder.model")
    print("Models saved successfully.")

'''
    Loads the network trained by GPU to CPU for inference. 
'''
def load_model(encoder, decoder):
    try:
        encoder.load_state_dict(torch.load("Model/encoder.model", 
                                           map_location='cpu'))
        decoder.load_state_dict(torch.load("Model/decoder.model", 
                                           map_location='cpu'))
        print("Loading model...")
    except RuntimeError:
        print("Runtime Error!")
        print(("Saved model must have the same network architecture with"
               " the CopyModel.\nRe-train and save again or fix the" 
               " architecture of CopyModel."))
        exit(1) # stop execution with error

def train_network(input_tensor, target_tensor, encoder, decoder, 
                  encoder_optimizer, decoder_optimizer, criterion, 
                  teacher_forcing_ratio=0.5, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=256, 
               learning_rate=0.01):
    print_loss_total = 0  # Reset every print_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train_network(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, 
                             criterion)
        print_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, 
                  print_loss_avg))
    save_model(encoder, decoder)
    
def train():
    encoder = EncoderRNN(input_lang.n_words).to(device)
    attn_decoder = AttnDecoderRNN(output_lang.n_words).to(device)
    trainIters(encoder, attn_decoder, 100000)

def test():
    load_model(encoder, decoder)

def process_input(input_to_translate):
    input_to_translate = input_to_translate.lower().strip()
    input_to_translate = re.sub(r"([.!?])", r" \1", input_to_translate)
    input_to_translate = re.sub(r"[^a-zA-Z.!?]+", r" ", input_to_translate)
    if input_to_translate[-1].isalpha() == True:
        input_to_translate = input_to_translate + " ."
    auxiliary = {'he is': 'he s', 
                 'she is': 'she s',
                 'it is': 'it s',
                 'i am': 'i m', 
                 'we are': 'we re',
                 'you are': 'you re',
                 'they are': 'they re'}
    for key, value in auxiliary.items():
        if key in input_to_translate:
            input_to_translate = input_to_translate.replace(key, value)
    return input_to_translate    

def translate():
    encoder = EncoderRNN(input_lang.n_words).to(device)
    decoder = AttnDecoderRNN(output_lang.n_words).to(device)
    load_model(encoder, decoder)
    # Gets input from the user
    input_to_translate = input("Enter your sentence: ")
    input_to_translate = process_input(input_to_translate)
    print(input_to_translate)
    def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, 
                                          device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
            decoder_hidden = encoder_hidden
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
        return decoded_words, decoder_attentions[:di + 1]

    output_words, _ = evaluate(encoder, decoder, input_to_translate)
    output_sentence = ' '.join(output_words)
    print("French translation: ", output_sentence)


if __name__ == '__main__':
    fire.Fire()
