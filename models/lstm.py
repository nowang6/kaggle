import torch
from torch import nn


class LSTMModel(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob, device):
        """
        Initialize the model by setting up the layers.
        Arguments:
        vocab_size - The size of the vocabulary, i.e., the total number of unique words in the input data.
        output_size - The size of the output, which is usually set to 1 for binary classification tasks like sentiment analysis.
        embedding_dim - The dimensionality of the word embeddings. Each word in the input data will be represented by a dense vector of this dimension.
        hidden_dim - The number of units in the hidden state of the LSTM layer.
        n_layers - The number of layers in the LSTM.
        drop_prob - The probability of dropout, which is a regularization technique used to prevent overfitting.
        """
        super(LSTMModel, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        # an embedding layer that maps each word index to its dense vector representation.
        # this layer is used to learn word embeddings during training.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # an LSTM layer that processes the input sequence of word embeddings
        # and produces a sequence of hidden states.
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=0.5,
                            batch_first=True, bidirectional=True)

        # a dropout layer that randomly sets elements of the input to zero
        # with probability drop_prob.
        # this layer helps in preventing overfitting.
        self.dropout = nn.Dropout(p=drop_prob)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # compute the word embeddings for the input sequence.
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size)
        embeds = self.embedding(x)

        # pass the embeddings through the LSTM layer to get the LSTM outputs and the updated hidden state.
        lstm_out, hidden = self.lstm(embeds, hidden)
        hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        fc_output = self.fc(hidden_cat)
        return fc_output

    def _init_hidden(self, batch_size):
        """ 
        Initializes hidden state 
        """
        hidden = torch.zeros(self.n_layers*2,
                             batch_size, self.hidden_dim, device=self.device)
        return hidden
