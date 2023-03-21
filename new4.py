import pandas as pd
import numpy as np
import transformers
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pickle as pkl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import string

## Configuration
class Config:
    CLS = [101]
    SEP = [102]
    VALUE_TOKEN = [0]
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 8
    EPOCHS = 3
    TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

## Create Dataset
class Dataset:
  
  def __init__(self, texts, tags):
    
    #Texts: [['Diana', 'is', 'a', 'girl], ['she', 'plays'. 'football']]
    #tags: [[0, 1, 2, 5], [1, 3. 5]]
    
    self.texts = texts
    self.tags = tags
  
  def __len__(self):
    return len(self.texts)

  def __getitem__(self, index):
    texts = self.texts[index]
    tags = self.tags[index]
  
    #Tokenise
    ids = []
    target_tag = []

    for i, s in enumerate(texts):
        inputs = Config.TOKENIZER.encode(s, add_special_tokens=False)
     
        input_len = len(inputs)
        ids.extend(inputs)
        target_tag.extend(input_len * [tags[i]])
    
    #To Add Special Tokens, subtract 2 from MAX_LEN
    ids = ids[:Config.MAX_LEN - 2]
    target_tag = target_tag[:Config.MAX_LEN - 2]

    #Add Sepcial Tokens
    ids = Config.CLS + ids + Config.SEP
    target_tags = Config.VALUE_TOKEN + target_tag + Config.VALUE_TOKEN

    mask = [1] * len(ids)
    token_type_ids = [0] * len(ids)

    #Add Padding if the input_len is small

    padding_len = Config.MAX_LEN - len(ids)
    ids = ids + ([0] * padding_len)
    target_tags = target_tags + ([0] * padding_len)
    mask = mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)

    return {
        "ids" : torch.tensor(ids, dtype=torch.long),
        "mask" : torch.tensor(mask, dtype=torch.long),
        "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
        "target_tags" : torch.tensor(target_tags, dtype=torch.long)
      }
  

## Training and Evaluation Fucntion
device =  "cuda" if torch.cuda.is_available() else "cpu"

def train_fn(train_data_loader, model, optimizer, device, scheduler):
    #Train the Model
    model.train()
    loss_ = 0
    for data in tqdm(train_data_loader, total = len(train_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)

        #Backward Propagation
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ += loss.item()
    return model, loss_ / len(train_data_loader)

def val_fn(val_data_loader, model, optimizer, device, scheduler):
    model.eval()
    loss_ = 0
    for data in tqdm(val_data_loader, total = len(val_data_loader)):
        for i, j in data.items():
            data[i] = j.to(device)
        _, loss = model(**data)
        loss_ += loss.item()
    return loss_ / len(val_data_loader)


## BUILD A MODEL AND CALCULATE THE LOSS
class NERBertModel(nn.Module):
    
    def __init__(self, num_tag):
        super(NERBertModel, self).__init__()
        self.num_tag = num_tag
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        
    def forward(self, ids, mask, token_type_ids, target_tags):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bert_out = self.bert_drop(output) 
        tag = self.out_tag(bert_out)
    
        #Calculate the loss
        Critirion_Loss = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = tag.view(-1, self.num_tag)
        active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags))
        loss = Critirion_Loss(active_logits, active_labels)
        return tag, loss
    

## DATA PREPROCESSING
#Load File
data = pd.read_csv(r"/home/sushant/env/files/final.csv", encoding="latin-1")

print(data.head())

print(data.isna().sum(axis=0))

#Tag Label Counts
print(data.labels.value_counts())

# #Filling Missing Values and Label Encoding
# data["Sentence #"] = data["Sentence #"].fillna(method='ffill')
# le = LabelEncoder().fit(data['Tag'])
# data['Tag'] = le.transform(data['Tag'])
# pkl.dump(le, open('labelenc.pkl', 'wb'))
# data.head()

#Filling Missing Values and Label Encoding
data["Unnamed: 0"] = data["Unnamed: 0"].fillna(method='ffill')
le = LabelEncoder().fit(data['labels'])
data['labels'] = le.transform(data['labels'])
pkl.dump(le, open('labelenc.pkl', 'wb'))
data.head()

#Group By Sentences
data_gr = data.groupby("Unnamed: 0").agg({'text': list, 'labels':list})
print(data_gr.head())

#Train Test Split
train_sent, val_sent, train_tag, val_tag = train_test_split(data_gr['text'], data_gr['labels'], test_size=0.01, random_state=10)

#Create DataLoaders
train_dataset = Dataset(texts = train_sent, tags = train_tag)
val_dataset = Dataset(texts = val_sent, tags = val_tag)
train_data_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE)
val_data_loader = DataLoader(val_dataset, batch_size=Config.VAL_BATCH_SIZE)

for data_ in train_data_loader:
    print(data_)
    break

## TRAIN THE MODEL

#Model Architecture
num_tag = len(data.labels.value_counts())
model = NERBertModel(num_tag=num_tag)
model.to(device)
print(model)

#Function for getparameters
def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters

# Set hyperparameters (optimizer, weight decay, learning rate)
FULL_FINETUNING = True
optimizer_grouped_parameters = get_hyperparameters(model, FULL_FINETUNING)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-5)
num_train_steps = int(len(train_sent) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_train_steps)

for epoch in range(Config.EPOCHS):
    model, train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
    val_loss = val_fn(val_data_loader, model, optimizer, device, scheduler)
    print(f"Epoch: {epoch + 1}, Train_loss: {train_loss}, Val_loss: {val_loss}")


def prediction(test_sentence, model, le):
    for i in list(string.punctuation):
        test_sentence = test_sentence.replace(i, ' ' + i)
    test_sentence = test_sentence.split()
    print(test_sentence)
    Token_inputs = Config.TOKENIZER.encode(test_sentence, add_special_tokens=False)
    print(Token_inputs)
    test_dataset =  Dataset(test_sentence, tags= [[1] * len(test_sentence)])
    num_tag = len(le.classes_)
   
    with torch.no_grad():
        data = test_dataset[0]
        for i, j in data.items():
            data[i] = j.to(device).unsqueeze(0)
        tag, _ = model(**data)

        print(le.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:len(Token_inputs)+1])


test_sentence = "Charles I was born in Fife on 19 November 1600."
prediction(test_sentence, model, le)

torch.save(model.state_dict(), "model.bin")
