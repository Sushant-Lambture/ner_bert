import tensorflow  as tf
# import tensorflow.compat.v1 as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

# pip install transformers
 evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tensorflow.keras.callbacks import EarlyStopping


import transformers
from transformers import BertTokenizer
# from transformers import AutoTokenizer
from transformers import BertTokenizerFast
from transformers import TFBertModel

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# dataframe = pd.read_csv(r"/home/sushant/env/ner_bert/ner_bert/train_final.csv")
# dataframe = pd.read_csv("train_final.csv")
dataframe = pd.read_csv("final.csv")

# dataframe = dataframe.dropna()
# dataframe = dataframe.drop('Unnamed: 0',axis=1)

dataframe.rename({'Unnamed: 0':'Sentence','text':'Word','labels':'Tag'},axis=1,inplace=True)
# dataframe.rename({'Unnamed: 0':'Sentence','word':'Word','label':'Tag'},axis=1,inplace=True)
# dataframe.rename({'index':'Sentence','word':'Word','label':'Tag'},axis=1,inplace=True)
dataframe=dataframe[:100]

dataframe.Tag.unique()
print(f"Number of Tags : {len(dataframe.Tag.unique())}")

# EDA
pie = dataframe['Tag'].value_counts()
px.pie(names = pie.index,values= pie.values,hole = 0.5,title ='Total Count of Tags')

dataframe["Sentence"] = dataframe["Sentence"].fillna(method="ffill")
sentence = dataframe.groupby("Sentence")["Word"].apply(list).values
# pos = dataframe.groupby(by = 'Sentence')['POS'].apply(list).values
tag = dataframe.groupby(by = 'Sentence')['Tag'].apply(list).values
print(sentence)
print(tag)

def process_data(data_path):
    df = (data_path)#, encoding="latin-1")
    df.loc[:, "Sentence"] = df["Sentence"].fillna(method="ffill")

#     enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

#     df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence")["Word"].apply(list).values
#     pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence")["Tag"].apply(list).values
    return sentences, tag, enc_tag

# sentence,tag,enc_tag = process_data('dataframe = pd.read_csv(r"C:\Users\Admin\Downloads\train_set2.csv")')
sentence,tag,enc_tag = process_data(dataframe)
print(sentence)
print(tag)
print(enc_tag)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
MAX_LEN = 128
def tokenize(data,max_len = MAX_LEN):
    input_ids = list()
    attention_mask = list()
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(data[i],
                                        add_special_tokens = True,
                                        max_length = MAX_LEN,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding = 'max_length',
                                        truncation=True,return_tensors = 'np')
                        
        
        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
    return np.vstack(input_ids),np.vstack(attention_mask)

# splitting Data

X_train,X_test,y_train,y_test = train_test_split(sentence,tag,random_state=52,test_size=0.1)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

input_ids,attention_mask = tokenize(X_train,max_len = MAX_LEN)

val_input_ids,val_attention_mask = tokenize(X_test,max_len = MAX_LEN)

# TEST: Checking Padding and Truncation length's
was = list()
for i in range(len(input_ids)):
    was.append(len(input_ids[i]))
set(was)

# Test Padding
test_tag = list()
for i in range(len(y_test)):
    test_tag.append(np.array(y_test[i] + [0] * (128-len(y_test[i]))))
    
# TEST:  Checking Padding Length
was = list()
for i in range(len(test_tag)):
    was.append(len(test_tag[i]))
set(was)

# Train Padding
train_tag = list()
for i in range(len(y_train)):
    train_tag.append(np.array(y_train[i] + [0] * (128-len(y_train[i]))))
    
# TRAIN:  Checking Padding Length
was = list()
for i in range(len(train_tag)):
    was.append(len(train_tag[i]))
set(was)

# Building BERT Model : Transfer Learning
# bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def create_model(bert_model,max_len = MAX_LEN):
    input_ids = tf.keras.Input(shape = (max_len,),dtype = 'int32')
    attention_masks = tf.keras.Input(shape = (max_len,),dtype = 'int32')
    bert_output = bert_model(input_ids,attention_mask = attention_masks,return_dict =True)
    embedding = tf.keras.layers.Dropout(0.3)(bert_output["last_hidden_state"])
    output = tf.keras.layers.Dense(17,activation = 'softmax')(embedding)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = [output])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model

with strategy.scope():
    bert_model = TFBertModel.from_pretrained('bert-base-uncased',from_pt=True)
    model = create_model(bert_model,MAX_LEN)

model.summary()

# tf.keras.utils.plot_model(model)

# # Model Training

# early_stopping = EarlyStopping(mode='min',patience=5)
# history_bert = model.fit([input_ids,attention_mask],np.array(train_tag),epochs = 1,callbacks = early_stopping,verbose = True)

early_stopping = EarlyStopping(mode='min',patience=5)
history_bert = model.fit([input_ids,attention_mask],np.array(train_tag),validation_data = ([val_input_ids,val_attention_mask],np.array(test_tag)),epochs = 1,batch_size=10*2,callbacks = early_stopping,verbose = True)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# model.save_weights("ner_bert_weights")


plt.plot(history_bert.history['accuracy'])
plt.plot(history_bert.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history_bert.history['loss'])
plt.plot(history_bert.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Testing Model

def pred(val_input_ids,val_attention_mask):
    return model.predict([val_input_ids,val_attention_mask])

def testing(val_input_ids,val_attention_mask,enc_tag,y_test):
    val_input = val_input_ids.reshape(1,128)
    val_attention = val_attention_mask.reshape(1,128)
    
    # Print Original Sentence
    sentence = tokenizer.decode(val_input_ids[val_input_ids > 0])
    print("Original Text : ",str(sentence))
    print("\n")
    
    true_enc_tag = enc_tag.inverse_transform(y_test)

    print("Original Tags : " ,str(true_enc_tag))
    print("\n")
    
    pred_with_pad = np.argmax(pred(val_input,val_attention),axis = -1) 
    pred_without_pad = pred_with_pad[pred_with_pad>0]
    pred_enc_tag = enc_tag.inverse_transform(pred_without_pad)
    print("Predicted Tags : ",pred_enc_tag)

testing(val_input_ids[0],val_attention_mask[0],enc_tag,y_test[0])


true_with_pad = np.argmax((val_input_ids,val_attention_mask),axis = -1) 
true_without_pad = true_with_pad[true_with_pad>0]
for i in range(len(true_without_pad)):
  if true_without_pad[i]!=1:
    if true_without_pad[i]!=2:
      if true_without_pad[i]!=3:
        true_without_pad[i] = 2
true_enc_tag = enc_tag.inverse_transform(true_without_pad)
print("True Tags : ",true_enc_tag)
len(true_enc_tag)

pred_with_pad = np.argmax(pred(val_input_ids,val_attention_mask),axis = -1) 
pred_without_pad = pred_with_pad[pred_with_pad>0]
for i in range(len(pred_without_pad)):
  if pred_without_pad[i]!=1:
    if pred_without_pad[i]!=2:
      if pred_without_pad[i]!=3:
        pred_without_pad[i] = 2
pred_enc_tag = enc_tag.inverse_transform(pred_without_pad)
print("Predicted Tags : ",pred_enc_tag)
len(pred_enc_tag)

from sklearn.metrics import accuracy_score,classification_report,f1_score
print(accuracy_score(true_enc_tag,pred_enc_tag))
print(classification_report(true_enc_tag,pred_enc_tag))





test_df=pd.read_csv("test_set_ran.csv")
# dataframe = dataframe.drop('Unnamed: 0',axis=1)
test_df.rename({'Unnamed: 0.1':'Sentence','word':'Word','label':'Tag'},axis=1,inplace=True)
test_df

test_df.Tag.unique()
print(f"Number of Tags : {len(test_df.Tag.unique())}")

# EDA
pie = test_df['Tag'].value_counts()
px.pie(names = pie.index,values= pie.values,hole = 0.5,title ='Total Count of Tags')

test_df["Sentence"] = test_df["Sentence"].fillna(method="ffill")
sentence = test_df.groupby("Sentence")["Word"].apply(list).values
# pos = dataframe.groupby(by = 'Sentence')['POS'].apply(list).values
tag = test_df.groupby(by = 'Sentence')['Tag'].apply(list).values
print(sentence)
print('***************************************')
print(tag)

def process_data(data_path):
    df = (data_path)#, encoding="latin-1")
    df.loc[:, "Sentence"] = df["Sentence"].fillna(method="ffill")

#     enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

#     df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence")["Word"].apply(list).values
#     pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence")["Tag"].apply(list).values
    return sentences, tag, enc_tag

# sentence,tag,enc_tag = process_data('dataframe = pd.read_csv(r"C:\Users\Admin\Downloads\train_set2.csv")')
sentence,tag,enc_tag = process_data(test_df)
print(sentence)
print(tag)
print(enc_tag)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
MAX_LEN = 128
def tokenize(data,max_len = MAX_LEN):
    input_ids = list()
    attention_mask = list()
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(data[i],
                                        add_special_tokens = True,
                                        max_length = MAX_LEN,
                                        is_split_into_words=True,
                                        return_attention_mask=True,
                                        padding = 'max_length',
                                        truncation=True,return_tensors = 'np')
                        

        input_ids.append(encoded['input_ids'])
        attention_mask.append(encoded['attention_mask'])
    return np.vstack(input_ids),np.vstack(attention_mask)

# splitting Data

X_train,X_test,y_train,y_test = train_test_split(sentence,tag,random_state=52,test_size=0.99)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# input_ids,attention_mask = tokenize(X_train,max_len = MAX_LEN)

val_input_ids,val_attention_mask = tokenize(X_test,max_len = MAX_LEN)

# TEST: Checking Padding and Truncation length's
was = list()
for i in range(len(input_ids)):
    was.append(len(input_ids[i]))
set(was)

# Train Padding
test_tag = list()
for i in range(len(y_test)):
    test_tag.append(np.array(y_test[i] + [0] * (128-len(y_test[i]))))
    
# TEST:  Checking Padding Length
was = list()
for i in range(len(test_tag)):
    was.append(len(test_tag[i]))
set(was)

# Train Padding
train_tag = list()
for i in range(len(y_train)):
    train_tag.append(np.array(y_train[i] + [0] * (128-len(y_train[i]))))
    
# TRAIN:  Checking Padding Length
was = list()
for i in range(len(train_tag)):
    was.append(len(train_tag[i]))
set(was)

#Testing Model

def pred(val_input_ids,val_attention_mask):
    return model.predict([val_input_ids,val_attention_mask])

def testing(val_input_ids,val_attention_mask,enc_tag,y_test):
    val_input = val_input_ids.reshape(1,128)
    val_attention = val_attention_mask.reshape(1,128)
    
    # Print Original Sentence
    sentence = tokenizer.decode(val_input_ids[val_input_ids > 0])
    print("Original Text : ",str(sentence))
    print("\n")
    true_enc_tag = enc_tag.inverse_transform(y_test)

    print("Original Tags : " ,str(true_enc_tag))
    print("\n")
    
    pred_with_pad = np.argmax(pred(val_input,val_attention),axis = -1) 
    pred_without_pad = pred_with_pad[pred_with_pad>0]
    pred_enc_tag = enc_tag.inverse_transform(pred_without_pad)
    print("Predicted Tags : ",pred_enc_tag)

testing(val_input_ids[0],val_attention_mask[0],enc_tag,y_test[0])


true_with_pad = np.argmax((val_input_ids,val_attention_mask),axis = -1) 
true_without_pad = true_with_pad[true_with_pad>0]
for i in range(len(true_without_pad)):
  if true_without_pad[i]!=1:
    if true_without_pad[i]!=2:
      if true_without_pad[i]!=3:
        true_without_pad[i] = 2
true_enc_tag = enc_tag.inverse_transform(true_without_pad)
print("True Tags : ",true_enc_tag)
len(true_enc_tag)

pred_with_pad = np.argmax(pred(val_input_ids,val_attention_mask),axis = -1) 
pred_without_pad = pred_with_pad[pred_with_pad>0]
for i in range(len(pred_without_pad)):
  if pred_without_pad[i]!=1:
    if pred_without_pad[i]!=2:
      if pred_without_pad[i]!=3:
        pred_without_pad[i] = 2
pred_enc_tag = enc_tag.inverse_transform(pred_without_pad)
print("Predicted Tags : ",pred_enc_tag)
len(pred_enc_tag)

from sklearn.metrics import accuracy_score,classification_report,f1_score
print(accuracy_score(true_enc_tag,pred_enc_tag))
print(classification_report(true_enc_tag,pred_enc_tag))
