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
from models.lstm import LSTMModel
from sklearn.model_selection import train_test_split


# TRAIN_FILE = "data/Kaggle_Sentiment_Analysis/train1.tsv.zip"
TRAIN_FILE = "data/Kaggle_Sentiment_Analysis/train.tsv.zip"
#TEST_FILE = "data/Kaggle_Sentiment_Analysis/test1.tsv.zip"
TEST_FILE = "data/Kaggle_Sentiment_Analysis/test.tsv.zip"

VOC_SIZE = 10000
CLASS_NUM = 5
EPOCHS = 5
BATCH_SIZE = 30
MODEL_PATH = "SentimentRNN.pt"
MAX_SEQ_LENGTH = 40
CLIP_VALUE = 5


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_data(file_path):
    df = pd.read_csv(file_path, compression="zip", sep = '\t')
    phraseIds = []
    for id in tqdm(df['PhraseId']):
        phraseIds.append(id)
    stopwords_set = set(stopwords.words("english"))
    phrases = []
    for text in tqdm(df['Phrase']):
        # convert to lowercase
        if (not isinstance(text, str)):
            text = ""
        text = text.lower()
        # remove punctuation and additional empty strings
        tokens = text.split()
        filtered_tokens = [token for token in tokens if not token in stopwords_set]
        phrases.append(filtered_tokens)
    return phraseIds, phrases

def get_target(file_path):
    df = pd.read_csv(file_path, compression="zip", sep = '\t')
    sentiments = []
    for sentiment in tqdm(df['Sentiment']):
        sentiments.append(sentiment)
    return sentiments


_,train_data_pp = get_data(TRAIN_FILE)
train_target_pp = get_target(TRAIN_FILE)

print("Train data sample: ", train_data_pp[0])

vocab = build_vocab_from_iterator((train_data_pp), specials=["<unk>","<pad>",], max_tokens=VOC_SIZE)
voc_dic =  vocab.vocab.get_stoi()
voc_dic_itos = vocab.vocab.get_itos()

def encode_data(data):
    phrases_ints = []
    for ph in data:
        one_phrase_ints = []
        for word in ph:
            word_ix = voc_dic.get(word,0) #0 is <unk>
            one_phrase_ints.append(word_ix)
        phrases_ints.append(one_phrase_ints) 
    return phrases_ints
train_reviews_ints = encode_data(train_data_pp)

print("Encoded train data ix sample: ", train_reviews_ints[0])
print("Decoded train data sample: ",[voc_dic_itos[i] for i in train_reviews_ints[0]])


X_train = pad_sequence(train_reviews_ints,MAX_SEQ_LENGTH)
print("Train data after pading: ",[voc_dic_itos[i] for i in X_train[0]])

def one_hot_encode(datas, num_classes):
    seqs = []
    for data in datas:
        seq = [0] * num_classes
        seq[data] = 1
        seqs.append(seq)
    return seqs

y_train = one_hot_encode(train_target_pp, CLASS_NUM)
print('Example of target: ', y_train[0])

X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2)

train_data_set = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_data_set = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_data_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
val_loader = DataLoader(val_data_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

model = LSTMModel(VOC_SIZE, output_size=5, embedding_dim=300, hidden_dim=256, n_layers=2, drop_prob=0.5)

model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()


for e in range(EPOCHS):
    running_loss = 0
    model.train()
    h = model.init_hidden(BATCH_SIZE, DEVICE)
    for seq, targets in tqdm(train_loader):
        # move data to device
        seq = seq.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # convert the elements of the hidden state tuple h to tensors with the same device as the input data.
        h = tuple([each.data for each in h])
        
        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (h).
        out, h = model.forward(seq, h)
        
        # calculate the loss between the predicted output and the target values
        loss = criterion(out, targets.float())
        # print("Loss:", loss)
        running_loss += loss.item()*seq.shape[0]
        
        # reset the gradients of the model's parameters 
        optimizer.zero_grad()
        
        # compute the gradients of the loss with respect to the model's parameters
        loss.backward()
            
        # clip the gradients to prevent them from exploding
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
            
        # update the model's parameters
        optimizer.step()
    running_loss /= len(train_loader.sampler)
    print("Running loss:", running_loss)

    #validate
    val_h = model.init_hidden(BATCH_SIZE, device=DEVICE)
    model.eval()
    accuracy = []
    val_loss = 0
    for seq, targets in val_loader:
        
        # convert the elements of the hidden state tuple val_h to tensors with the same device as the input data.
        val_h = tuple([each.data for each in val_h])
        
        # move data to device
        seq = seq.to(DEVICE)
        targets = targets.to(DEVICE)

        # perform a forward pass through the model.
        # returns the model's output (out) and the updated hidden state (val_h).
        out, val_h = model(seq, val_h)
        
        # calculate the loss
        loss = criterion(out, targets.float())
        val_loss += loss.item()*seq.shape[0]
    val_loss /= len(val_loader.sampler)
    print("Val loss:", val_loss)
           
torch.save(model.state_dict(), MODEL_PATH)



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
#print("Test data after pading: ",[voc_dic_itos[i] for i in X_test[0]])
