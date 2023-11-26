import torch


class GRUModel(torch.nn.Module):
    # input_size 128, hidden_size 100, output_size 一共18个国家
    def __init__(self, voc_size, embeeding_size, hidden_size, output_size, n_layers=1):
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2
        self.embedding = torch.nn.Embedding(voc_size, embeeding_size)
        self.gru = torch.nn.GRU(embeeding_size, hidden_size,
                                n_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(
            hidden_size*self.n_directions, output_size)  # 如果是双向则维度*2

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,
                             batch_size, self.hidden_size, device=self.device)
        return hidden

    def forward(self, input):
        # input shape Batchsize*SeqLen->SeqLen*Batchsize
        batch_size = input.size(0)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)

        output, hidden = self.gru(embedding, hidden)

        hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        fc_output = self.fc(hidden_cat)
        return fc_output
