from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.layers.recurrent import LSTM
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
#from keras.layers.merge import multiply
from sklearn.model_selection import train_test_split
from rdkit import Chem
import sys
import os
from keras.layers.embeddings import Embedding
import MeCab
from keras.utils import np_utils
mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
mecab.parse('')

from keras_self_attention import SeqSelfAttention

def vectorize(smiles,embed,charset,char_to_int):
    one_hot =  np.zeros((smiles.shape[0],embed , len(charset)),dtype=np.int8)
    for i in range(smiles.shape[0]):
        node = mecab.parseToNode(smiles[i])
        w_array1=np.zeros(0)


        while node:
        #単語を取得
            word = node.surface
            pos = node.feature.split(",")[1]
            if pos=='格助詞':
                node = node.next
            elif pos=='非自立':
                node = node.next
            elif pos=='読点':
                node = node.next
            elif pos=='句点':
                node = node.next
            elif pos=='＊':
                node = node.next
            elif pos=='接尾':
                node = node.next
            elif pos=='係助詞':
                node = node.next
            elif pos=='括弧開':
                node = node.next
            elif pos=='括弧閉':
                node = node.next
            elif pos=='副助詞／並立助詞／終助詞':
                node = node.next
            elif word=='':
                node = node.next
            else:
                w_array1=np.append(w_array1,word)
                node = node.next
        x1 = w_array1.reshape(-1,1)
        for j in range(x1.shape[0]):
            num = char_to_int[''.join(x1[j])]
            one_hot[i,j,num]=1
    return one_hot

def create_normal_model(embed,vocab_size):
    #inputs = Input((embed-1,))
    inputs = Input((embed,))
    x = Embedding(vocab_size, embed-1)(inputs)#embed-1=56
    x = SeqSelfAttention(name='attention')(x)
    x = Conv1D(100,20,activation='relu')(x)
    x = BatchNormalization()(x)
    x = LSTM(150,return_sequences=True)(x)
    x = LSTM(200,return_sequences=True)(x)
    #x = SeqSelfAttention(name='attention')(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    prediction = Dense(2,activation='softmax')(x)
    print(prediction.shape)
    model = Model(inputs=inputs, outputs=prediction)
    model.summary()
    return model

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def node_check(data):

    x = ' '.join(data)
    #mecab.parse('')#文字列がGCされるのを防ぐ
    node = mecab.parseToNode(x)
    #node = mecab.parseToNode(X_test_smiles[1])
    w_array=np.zeros(0)
    while node:
        #単語を取得
        word = node.surface
        pos = node.feature.split(",")[1]
        if pos=='格助詞':
            node = node.next
        elif pos=='非自立':
            node = node.next
        elif pos=='読点':
            node = node.next
        elif pos=='句点':
            node = node.next
        elif pos=='＊':
            node = node.next
        elif pos=='接尾':
            node = node.next
        elif pos=='係助詞':
            node = node.next
        elif pos=='括弧開':
            node = node.next
        elif pos=='括弧閉':
            node = node.next
        elif pos=='副助詞／並立助詞／終助詞':
            node = node.next
        elif word=='':
            node = node.next
        else:
            #print(word)
            w_array=np.append(w_array,word)
        #次の単語に進める
            node = node.next
        return w_array





def main():
    data = pd.read_excel(r'./el_lc.xlsx')
    x_data = np.array(list(data["word"]))
    assay = "class"
    y_data = data[assay].values.reshape(-1,1)
    y_data = np_utils.to_categorical(y_data,2)
    DATA_DIR='./'
    files = os.listdir(DATA_DIR)
    
    (X_train_smiles, X_test_smiles,
     Y_train, Y_test) = train_test_split(
        x_data, y_data, test_size=0.2, random_state=10,
    )
    print(X_test_smiles[1])
    quit()
    #w_array=node_check(x_data)

    
    x = ' '.join(x_data)

    #mecab.parse('')#文字列がGCされるのを防ぐ
    node = mecab.parseToNode(x)
    w_array=np.zeros(0)
    while node:
        #単語を取得
        word = node.surface
        pos = node.feature.split(",")[1]
        if pos=='格助詞':
            node = node.next
        elif pos=='非自立':
            node = node.next
        elif pos=='読点':
            node = node.next
        elif pos=='句点':
            node = node.next
        elif pos=='＊':
            node = node.next
        elif pos=='接尾':
            node = node.next
        elif pos=='係助詞':
            node = node.next
        elif pos=='括弧開':
            node = node.next
        elif pos=='括弧閉':
            node = node.next
        elif pos=='副助詞／並立助詞／終助詞':
            node = node.next
        else:
            w_array=np.append(w_array,word)
        #次の単語に進める
            node = node.next


    charset = sorted(set(w_array))
    #print(charset)
    char_to_int = dict((c,i) for i,c in enumerate(charset))
    embed = max([len(smile) for smile in data.word]) + 5
    #print(char_to_int)
    print('embed',embed)
    X_train = vectorize(X_train_smiles,embed,charset,char_to_int)
    print("Training data is vectrizing",flush=True)
    X_test  = vectorize(X_test_smiles,embed,charset,char_to_int)    
    print("Test data is vectrizing",flush=True)
    vocab_size=len(charset)
    print(vocab_size)
    

    model=create_normal_model(embed,vocab_size)
    
    optimizer = Adam(lr=0.0001)
    lr_metric = get_lr_metric(optimizer)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy',lr_metric])


    callbacks_list = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-15, verbose=1, mode='auto',cooldown=0),
        ModelCheckpoint(filepath="weights0610.h5", monitor='val_loss', save_best_only=True, verbose=1, mode='auto')

    ]
    history =model.fit(x=np.argmax(X_train, axis=2), y=Y_train,
                                  batch_size=16,
                                  epochs=30,
                                  validation_data=(np.argmax(X_test, axis=2),Y_test),
                                  callbacks=callbacks_list
                                  )


    word=node_check(X_test_smiles[1])
    df=pd.DataFrame(word)
    df.to_csv("word.csv",index=False,header=False)

    layer_name = 'attention'
    hidden_layer_model=Model(inputs = model.input,outputs = model.get_layer(layer_name).output)
    hidden_output = hidden_layer_model.predict(np.argmax(X_test,axis=2))
    print(hidden_output.shape)
    hidden_output = np.max(hidden_output[1],axis=1)
    #hidden_output=hidden_output.reshape(embed-1,-1)
    #mean = np.array(hidden_output).mean()
    df1=pd.DataFrame(hidden_output)
    df1.to_csv("attention0610.csv",index=False,header=False)
    df2=pd.merge(df1, df)
    df2.to_csv("attention_word.csv",index=False,header=False)

    json_string = model.to_json()
    open('smiles2vec_LSTM_model.json', 'w').write(json_string)
    model.save_weights('smiles2vec_LSTM_weights.h5')

    y_pred=model.predict(np.argmax(X_test,axis=2))
    df1=pd.DataFrame(y_pred)
    df1.to_csv("result.csv",index=False,header=False)







if __name__=='__main__':
     main()
