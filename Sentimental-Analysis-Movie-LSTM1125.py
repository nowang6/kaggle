# https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch
from torch import nn, optim
from util.data import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from models.gru import GRUModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# TRAIN_FILE = "data/Kaggle_Sentiment_Analysis/train1.tsv.zip"
TRAIN_FILE = "data/Kaggle_Sentiment_Analysis/train.tsv.zip"
# TEST_FILE = "data/Kaggle_Sentiment_Analysis/test1.tsv.zip"
TEST_FILE = "data/Kaggle_Sentiment_Analysis/test.tsv.zip"

VOC_SIZE = 50000
CLASS_NUM = 5
EPOCHS = 2
BATCH_SIZE = 50
MODEL_PATH = "SentimentRNN.pt"
MAX_SEQ_LENGTH = 100


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_data(file_path):
    df = pd.read_csv(file_path, compression="zip", sep='\t')
    phrase_ids = list(df['PhraseId'])

    phrases = []
    stopwords_set = set(stopwords.words("english"))
    for text in tqdm(df['Phrase']):
        # convert to lowercase
        if (not isinstance(text, str)):
            text = ""
        text = text.lower()
        # remove punctuation and additional empty strings
        tokens = text.split()
        filtered_tokens = [
            token for token in tokens if not token in stopwords_set]
        phrases.append(tokens)
    sentiments = list(df["Sentiment"])
    return phrase_ids, phrases, sentiments


_, data_x, data_y = get_data(TRAIN_FILE)

print(f"Train data sample, phrase: {data_x[0]}, sentiment: {data_y[0]}")

vocab = build_vocab_from_iterator(data_x, specials=[
                                  "<unk>", "<pad>",], max_tokens=VOC_SIZE)
voc_dic = vocab.vocab.get_stoi()
voc_dic_itos = vocab.vocab.get_itos()


def encode_data(data):
    phrases_ints = []
    for ph in data:
        one_phrase_ints = []
        for word in ph:
            word_ix = voc_dic.get(word, 0)  # 0 is <unk>
            one_phrase_ints.append(word_ix)
        phrases_ints.append(one_phrase_ints)
    return phrases_ints


data_x = encode_data(data_x)

print("Encoded train data ix sample: ", data_x[0])
print("Decoded train data sample: ", [voc_dic_itos[i] for i in data_x[0]])


data_x = pad_sequence(data_x, MAX_SEQ_LENGTH)
print("Train data after pading: ", [voc_dic_itos[i] for i in data_x[0]])


X_train, X_val, y_train, y_val = train_test_split(
    data_x, data_y, test_size=0.2)

train_data_set = TensorDataset(torch.tensor(
    X_train, device=DEVICE), torch.tensor(y_train, device=DEVICE))
val_data_set = TensorDataset(torch.tensor(
    X_val, device=DEVICE), torch.tensor(y_val, device=DEVICE))

train_loader = DataLoader(train_data_set, shuffle=True,
                          batch_size=BATCH_SIZE, drop_last=True)
val_loader = DataLoader(val_data_set, shuffle=True,
                        batch_size=BATCH_SIZE, drop_last=True)

model = GRUModel(voc_size=VOC_SIZE, embeeding_size=300,
                 hidden_size=256, output_size=5, n_layers=1)

model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for e in range(EPOCHS):
    # train
    running_loss = 0
    model.train()
    for input, targets in tqdm(train_loader):

        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (h).
        out = model(input)

        # calculate the loss between the predicted output and the target values
        loss = criterion(out, targets)

        running_loss += loss.item()

        # reset the gradients of the model's parameters
        optimizer.zero_grad()

        # compute the gradients of the loss with respect to the model's parameters
        loss.backward()

        # update the model's parameters
        optimizer.step()
    print(f"{e} Epoches loss: {running_loss}")

    # validate
    correct = 0
    total = len(X_val)
    print("evaluating trained model...")
    model.eval()
    for input, targets in tqdm(val_loader):
        output = model(input)
        pred = output.max(dim=1, keepdim=True)[1]
        correct += pred.eq(targets.view_as(pred)).sum().item()

    percent = '%.2f' % (100*correct/total)
    print(f'Test set:Accuracy {correct}/{total} {percent}%')


#     # validate
#     model.eval()
#     accuracy = []
#     val_loss = 0
#     for seq, targets in val_loader:

#         # convert the elements of the hidden state tuple val_h to tensors with the same device as the input data.
#         val_h = tuple([each.data for each in val_h])

#         # move data to device
#         seq = seq.to(DEVICE)
#         targets = targets.to(DEVICE)

#         # perform a forward pass through the model.
#         # returns the model's output (out) and the updated hidden state (val_h).
#         out, val_h = model(seq, val_h)

#         # calculate the loss
#         loss = criterion(out, targets.float())
#         val_loss += loss.item()*seq.shape[0]
#     val_loss /= len(val_loader.sampler)
#     print("Val loss:", val_loss)

# torch.save(model.state_dict(), MODEL_PATH)


# predict

# phraseIds, phrases= get_data(TEST_FILE)
# test_phrases_ints = encode_data(phrases)
# print("Encoded test data ix sample: ",test_reviews_ints[0])
# print("Decoded test data sample: ",[voc_dic_itos[i] for i in test_reviews_ints[0]])
# test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
# X_test = pad_sequence(test_reviews_ints,max_seq_length)


# df = pd.DataFrame({'PhraseId': pd.Series(dtype='int'),
#                     'Sentiment': pd.Series(dtype='int')})
# test_h = model.init_hidden(batch_size, device)
# model.eval()
# for seq, id_ in test_loader:
#     test_h = tuple([each.data for each in test_h])
#     seq = seq.to(device)
#     out, test_h = model(seq, test_h)
#     out = get_prediction(out)
#     for ii, row in zip(id_, out):
#         if ii in test_zero_idx:
#             predicted = 2
#         else:
#             predicted = int(torch.argmax(row))
#         subm = {'PhraseId': int(ii),
#                 'Sentiment': predicted}
#         df = df.append(subm, ignore_index=True)
# return df
# print("Test data after pading: ",[voc_dic_itos[i] for i in X_test[0]])
