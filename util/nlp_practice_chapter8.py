from models.lstm import LSTMModel
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from models.lstm import LSTMModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from util.data import build_vocab_and_data
from util.data import init_network


EPOCHS = 2


def train(model, train_loader, valid_loader):
  model.train()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()
  for e in range(EPOCHS):
      # train
      running_loss = 0
      model.train()
      for input, targets in tqdm(train_loader):
          out = model(input)
          loss = criterion(out, targets)
          running_loss += loss.item()
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      print(f"{e} Epoches loss: {running_loss}")

      # validate
      correct = 0
      total = len(X_val)
      print("evaluating trained model...")
      model.eval()
      for input, targets in tqdm(valid_loader):
          output = model(input)
          pred = output.max(dim=1, keepdim=True)[1]
          correct += pred.eq(targets.view_as(pred)).sum().item()

      percent = '%.2f' % (100*correct/total)
      print(f'Test set:Accuracy {correct}/{total} {percent}%')



if __name__ == '__main__':
  BATCH_SIZE = 100
  EMBEEDING_DIM = 100
  HIDDEN_DIM = 256


  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  train_data = "data/THUCNews/train.txt"

  vocab, train_data, train_target = build_vocab_and_data(train_data)
  VOC_SIZE = len(vocab)
  X_train, X_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.3)
  
  train_data = TensorDataset(torch.from_numpy(X_train, DEVICE), torch.from_numpy(y_train, DEVICE))
  valid_data = TensorDataset(torch.from_numpy(X_val, DEVICE), torch.from_numpy(y_val, DEVICE))
  
  train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
  valid_loader = DataLoader(valid_data, shuffle=True, batch_size=BATCH_SIZE)
    

  model = LSTMModel(VOC_SIZE, embedding_dim=EMBEEDING_DIM, hidden_dim=HIDDEN_DIM, n_layers=2, drop_prob=0.5,output_size=10).to(DEVICE)
  init_network(model)
  train(model, train_loader, valid_loader)
