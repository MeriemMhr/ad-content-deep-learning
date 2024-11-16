#!/usr/bin/env python
# coding: utf-8

# In[39]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[91]:


import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import os
import warnings 
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay, recall_score
from PIL import Image
import os
from pylab import *
from utils import * 
from sklearn.model_selection import train_test_split
import json
import lzma
import json

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


# In[23]:


files = []
likes = []
data = pd.DataFrame()
for i in glob.glob('data/*/*json.xz'):



    with lzma.open(i, "rt") as file:
        content = file.read()

    a = json.loads(content)
    like = a['node']['edge_media_preview_like']['count']
    files.append(i)
    likes.append(like)
data['files'] = files
data['likes'] = likes   


# In[ ]:


for i in glob.glob('data/*'):
    source_directory = f'{i}/'  
    target_directory = f'{i}/images'  

    resize_images(source_directory, target_directory)


# In[24]:


def get_path(path):
    parts = path.split("/")

    
    before_images = parts[0]+'/'+parts[1]
    return before_images

data['path'] = data['files'].apply(get_path)
data['path'].loc[0]


# In[25]:


name_keys = []
file_keys = []
df_file_name = pd.DataFrame()
for i in glob.glob('data/*/images/*'):
    parts = i.split("images")
    before_images = parts[0].strip("/")
    name_keys.append(before_images)
    for j in glob.glob(f'{i}*'):
        file_keys.append(j)
        

df_file_name['name_keys'] = name_keys
df_file_name['file_keys'] = file_keys
data_merged = pd.merge(data, df_file_name, left_on='path', right_on='name_keys', how='left')


# In[17]:


def get_features_images_path():
    df = pd.DataFrame()

    for i in glob.glob('data/*/images/*.jpg'):
        a = detect_objects(i)
        a['path'] = i
        df = pd.concat([df, a])
   
    return df
features = get_features_images_path()
features.to_csv('features.csv')


# In[19]:


b = features.pivot_table(
                            index=['path'],
                            columns='objects',
                            values='scores',
                            aggfunc='mean'
                        )
b.columns = [i.lower().replace(' ','_') for i in b.columns]
b = b.reset_index()
b = b.fillna(0)
b


# In[63]:


data_final = data_merged.merge(b, left_on = ['file_keys'], right_on = ['path'], how = 'left')
data_final['likes'] = data_final['likes'].replace(-1,0)
data_final['target'] = np.where(data_final['likes']>500000,1,0)
data_final = data_final.fillna(0)


# In[60]:


images = load_and_prepare_images(data_final, 'file_keys')
target = data_final['target'].values.astype('float32')


# In[15]:


#influencers = [
#    'mrbeast',
#    ]

Application_Credentials = '/Users/aladelca/Downloads/massive-acrobat-421018-1d8b6ce1a11a.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Application_Credentials


# In[267]:


#data = pd.DataFrame()
#df_features = pd.DataFrame()
#for influencer in influencers:
#    data = pd.concat([data, preprocessing(influencer)])
#    data = data.reset_index(drop=True)
#    df_features = pd.concat([df_features,get_features_images(influencer)])
#    df_features = df_features.reset_index(drop=True)


# In[269]:


#df_final = preprocess_features(df_features, data)


# In[67]:


#images = load_and_prepare_images(data)
#likes = df_final['likes'].values.astype('float32')



# In[61]:


x_train, x_test, y_train, y_test = train_test_split(images, target, test_size=0.2, random_state=123)
df_train, df_test = train_test_split(data_final, test_size=0.2, random_state=123)
x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=123)
df_fit, df_val = train_test_split(df_train, test_size=0.2, random_state=123)


# In[272]:


#enc = OneHotEncoder()
#df_fit_enc = pd.concat([pd.DataFrame(enc.fit_transform(df_fit[['influencer']]).toarray(), 
#                          columns = enc.categories_[0], 
#                          index = df_fit.index),
#                          df_fit.drop(columns = ['influencer'])], axis = 1)
#df_val_enc = pd.concat([pd.DataFrame(enc.transform(df_val[['influencer']]).toarray(),
#                          columns = enc.categories_[0],
#                          index = df_val.index),
#                          df_val.drop(columns = ['influencer'])], axis = 1)

#df_test_enc = pd.concat([pd.DataFrame(enc.transform(df_test[['influencer']]).toarray(),
#                           columns = enc.categories_[0],
#                           index = df_test.index),
#                           df_test.drop(columns = ['influencer'])], axis = 1)


# In[273]:


#df_fit_enc = df_fit_enc.fillna(0)
#df_val_enc = df_val_enc.fillna(0)
#df_test_enc = df_test_enc.fillna(0) 


# In[81]:


### Scaling
DROP_VARS = ['likes','fecha_utc','hashtags','comentarios','key_date','filename','caption','img_source','clean_path']
NOT_CONSIDERED_COLUMNS = ['files', 'likes', 'path_x', 'name_keys', 'file_keys', 'path_y','target']
esc = MinMaxScaler()
df_fit_esc = esc.fit_transform(df_fit.loc[:, ~df_fit.columns.isin(NOT_CONSIDERED_COLUMNS)])
df_val_esc = esc.transform(df_val.loc[:, ~df_val.columns.isin(NOT_CONSIDERED_COLUMNS)])
df_test_esc = esc.transform(df_test.loc[:, ~df_test.columns.isin(NOT_CONSIDERED_COLUMNS)])

df_fit_esc = np.nan_to_num(df_fit_esc, 0)
df_val_esc = np.nan_to_num(df_val_esc, 0)
df_test_esc = np.nan_to_num(df_test_esc, 0)


# In[110]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

K.clear_session()
image_input = Input(shape=(128, 128, 3))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D(2, 2)(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D(2, 2)(x)
x = Dropout(0.5)(x) 
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)

# Entrada para la información del influencer
influencer_input = Input(shape=(df_fit_esc.shape[1],))
y = Dense(10, activation='relu')(influencer_input)  # Ajusta el tamaño según el número de categorías

# Combinar las entradas
combined = concatenate([x, y])

# Capas totalmente conectadas después de la concatenación
z = Dense(256, activation='relu')(combined)
z = Dense(1, activation = 'sigmoid')(z)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
model = Model(inputs=[image_input, influencer_input], outputs=z)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit([x_fit, df_fit_esc], y_fit, 
          epochs=10, 
          validation_data=([x_val, df_val_esc], y_val),
          callbacks=[early_stopping])


# In[111]:


### Finding the best threshold
predictions = model.predict([x_test, df_test_esc])
list_sensitivity = []
list_specificity = []
for i in np.arange(0.01, 1, 0.01):
    preds = np.where(predictions>i,1,0)
    list_sensitivity.append(recall_score(y_test, preds, pos_label=1))
    list_specificity.append(recall_score(y_test, preds, pos_label=0))


# In[112]:


best_threshold = np.arange(0.01, 1, 0.01)[np.argmax(np.array(list_sensitivity) + np.array(list_specificity) - 1)]
final_preds = np.where(predictions>best_threshold,1,0)


# In[113]:


cm = confusion_matrix(y_test, final_preds)
ConfusionMatrixDisplay(cm).plot()


# In[115]:


print(roc_auc_score(y_test, predictions))
print(accuracy_score(y_test, final_preds))
print(recall_score(y_test, final_preds, pos_label=1))
print(recall_score(y_test, final_preds, pos_label=0))

