import torch.nn as nn
import torch

# GRU + FC layer
# one_shot generation
class Basic_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Basic_GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=batch_first,
                          dropout=dropout)
        self.dense = nn.Linear(hidden_dim, input_dim)

    def forward(self, input):
        output, _ = self.gru(input)
        pred = self.dense(output)
        return pred

# Basic Encoder
# No output layer
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Encoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)

    def forward(self, inputs):
        enc_output, enc_state = self.encoder(inputs)
        return enc_output, enc_state

# Encoder with FC layer as output layer
class Encoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Encoder2, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dense2 = nn.Linear(hidden_dim // 2, input_dim)
        self.leaky = nn.LeakyReLU()

    def forward(self, inputs):#, h_0):
        enc_output, enc_state = self.encoder(inputs)#, h_0)
        enc_output = self.dense1(enc_output)
        enc_output = self.leaky(enc_output)
        enc_output = self.dense2(enc_output)
        # enc_output = self.leaky(enc_output)
        return enc_output, enc_state

# Decoder with single FC as output layer
class Decoder0(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Decoder0, self).__init__()
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)
        self.dense = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs, state):
        output, dec_state = self.decoder(inputs, state)
        pred = self.dense(output)
        return pred, dec_state


# Decoder with several FC layer as output layer
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(Decoder, self).__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dense2 = nn.Linear(hidden_dim // 2, input_dim)
        self.leaky = nn.LeakyReLU()

    def forward(self, inputs, state):
        outputs, dec_state = self.encoder(inputs, state)
        # outputs = self.tanh(output)
        pred = self.dense1(outputs)
        pred = self.leaky(pred)
        pred = self.dense2(pred)
        # pred = self.leaky(pred)
        return pred, dec_state


# This decoder does not contain an output layer
class Decoder2(nn.Module):
    def __init__(self, hidden_dim, num_layers, batch_first, dropout):
        super(Decoder2, self).__init__()
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)

    def forward(self, inputs, state):
        outputs, dec_state = self.decoder(inputs, state)
        return outputs, dec_state

# Decoder with MLP as output layer
class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, batch_first, dropout):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=batch_first, dropout=dropout)
        self.mlp = MLP(hidden_dim, input_dim)

    def forward(self, inputs, state):
        outputs, dec_state = self.decoder(inputs, state)
        # outputs = self.tanh(output)
        pred = self.mlp(outputs)
        return pred, dec_state

# Basic Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input):
        _, enc_state = self.encoder(encoder_input)
        pred, dec_state = self.decoder(decoder_input, enc_state)

        return pred, dec_state

# Encoder-Decoder model with encoder state init
class EncoderDecoder2(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, h_0, decoder_input):
        enc_outputs, enc_state = self.encoder(encoder_input, h_0)
        pred, dec_state = self.decoder(decoder_input, enc_state)
        return enc_outputs, pred, dec_state

# Encoder-Decoder model with shared FC output layer
class EncoderDecoder3(nn.Module):
    def __init__(self, encoder, decoder, input_dim, output_dim):
        super(EncoderDecoder3, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Linear(input_dim // 2, output_dim)
        )

    def forward(self, encoder_input, h_0, decoder_input):
        _, enc_state = self.encoder(encoder_input, h_0)
        pred, dec_state = self.decoder(decoder_input, enc_state)
        pred = self.output(pred)
        return pred, dec_state


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, input_dim // 2)
        self.layer2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.layer3 = nn.Linear(input_dim // 4, input_dim // 8)
        self.layer4 = nn.Linear(input_dim // 8, output_dim)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output
