import torch.nn as nn
import dnc


class DNCLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(DNCLM, self).__init__()
        self.nhid = nhid
        self.nlayers = nlayers

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = dnc.DNC(input_size=ninp,
                           hidden_size=nhid,
                           rnn_type='lstm',
                           num_layers=nlayers,
                           nr_cells=500,
                           cell_size=16,
                           read_heads=2,
                           dropout=dropout,
                           batch_first=False,
                           gpu_id=0,
                           debug=False)
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        return None, None, None
