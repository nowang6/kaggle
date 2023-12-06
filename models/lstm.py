import torch
from torch import nn


class LSTMModel(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob):

        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            n_layers,
                            dropout=0.5,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        embeds = self.embedding(x)

        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
