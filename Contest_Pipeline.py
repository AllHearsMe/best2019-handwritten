
# coding: utf-8

# In[1]:


import numpy as np
import charUtils
import glob
import cv2
import sys
from keras import layers, optimizers, models, callbacks
from keras import backend as K


# In[2]:


def get_model(image_size=(225, None, 1), split=True):
    Conv = lambda n: layers.Conv2D(n, (3, 3), activation='relu')
    Pooling = lambda: layers.MaxPool2D((2, 2))
    VertConv = lambda n: layers.Conv2D(n, (3, 1), activation='relu')
    VertPooling = lambda: layers.MaxPool2D((2, 1))
    BatchNorm = lambda: layers.BatchNormalization()
    BiGRU = lambda n: layers.Bidirectional(layers.GRU(n, activation='relu', return_sequences=True))
    
    inp = layers.Input(image_size)
    x = inp
    x = Conv(32)(x)
    x = Conv(32)(x)
    x = Pooling()(x)
    x = BatchNorm()(x)
    x = Conv(32)(x)
    x = Conv(32)(x)
    x = Pooling()(x)
    x = BatchNorm()(x)
    x = Conv(64)(x)
    x = Conv(64)(x)
    x = Pooling()(x)
    x = BatchNorm()(x)
    x = Conv(128)(x)
    x = Conv(128)(x)
    x = VertPooling()(x)
    x = BatchNorm()(x)
    x = Conv(128)(x)
    x = Conv(128)(x)
    x = layers.Lambda(lambda x: K.max(x, axis=1), name='vertical_max')(x)
    x = BatchNorm()(x)
    
    x = BiGRU(128)(x)
    x = BiGRU(128)(x)
    
    if split:
        outp = []
        for i, tp in enumerate(charUtils.componentList):
            outp.append(
                layers.TimeDistributed(layers.Dense(charUtils.charCount[i], activation='softmax'),
                                                    name=tp)(x))
    else:
        outp = layers.TimeDistributed(layers.Dense(charUtils.onehotLen, activation='softmax'),
                                                   name='combined')(x)
    
    model = models.Model(inputs=inp, outputs=outp)
    return model


# In[3]:


def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
#     # the 2 is critical here since the first couple outputs of the RNN
#     # tend to be garbage:
#     y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

dummy_loss_function = lambda y_true, y_pred: y_pred

def ctc_trainer(predictor_model, lr=0.001, max_label_length=80):
    input_data = predictor_model.inputs[0]
    y_pred = predictor_model(input_data)
    combined_labels = layers.Input(name='the_labels',
                                   shape=[max_label_length, charUtils.componentCount], dtype='int64')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    
    labels = []
    outp = []
    for i, tp in enumerate(charUtils.componentList):
        labels.append(layers.Lambda(lambda x: x[..., i], name=tp+'_labels')(combined_labels))
        outp.append(layers.Lambda(
            ctc_lambda_func, output_shape=(1,),
            name=tp+'_ctc')([labels[i], y_pred[i], input_length, label_length]))
    
    trainer_model = models.Model(inputs=[input_data, combined_labels, input_length, label_length],
                          outputs=outp)
    trainer_model.compile(loss={tp+'_ctc': dummy_loss_function for tp in charUtils.componentList},
                          optimizer=optimizers.Adam(lr))
    return trainer_model


# In[4]:


def fit_image(im, shape):
    fy = shape[0]/im.shape[0]
    fx = shape[1]/im.shape[1]
    if fy < fx:
        im2 = cv2.resize(im, dsize=(0, 0), fx=fy, fy=fy, interpolation=cv2.INTER_AREA if fy < 1 else cv2.INTER_CUBIC)
        im2 = np.pad(im2, [(0, 0), (0, shape[1]-im.shape[1])], mode='constant', constant_values=255)
    else:
        im2 = cv2.resize(im, dsize=(0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_AREA if fx < 1 else cv2.INTER_CUBIC)
        im2 = np.pad(im2, [(0, shape[1]-im.shape[1]), (0, 0)], mode='constant', constant_values=255)
    return im2


# In[5]:


def predict_text(model, x, batch_size=32):
    y_pred = model.predict(x, batch_size=batch_size, verbose=1)
    y_pred_idx = np.concatenate([np.argmax(y, axis=-1)[..., None] for y in y_pred], axis=-1).tolist()
    y_pred_str = [charUtils.decode(y) for y in y_pred_idx]
    return y_pred_str


# In[6]:

print('Loading model...')

predictor_model = get_model()
predictor_model.load_weights('Models/predictor-contest-round2-py36.hdf5')

print('Done.')

# In[7]:


input_files = sorted(glob.glob('Input/*'))
file_ids = [s.split('/')[-1].split('.')[0] for s in input_files]
output_files = ['Output/{}.txt'.format(s) for s in file_ids]

print('Found {} images.'.format(len(input_files)))

# In[8]:

print('Loading images...')
      
image_shape = (225, 2200, 1)
images_list = []
for imname in input_files:
    tmp = cv2.imread(imname, cv2.IMREAD_GRAYSCALE)
    tmp = fit_image(tmp, image_shape[:2])
    images_list.append(tmp[..., None])
images = np.stack(images_list)
del images_list

print('Done.')
      

# In[9]:

print('Predicting...')

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 16
pred_txt = predict_text(predictor_model, images, batch_size=batch_size)

print('Done.')

# In[10]:


for fn, s in zip(output_files, pred_txt):
    with open(fn, 'w', encoding='utf8') as f:
        f.write(s)

print('Written results to files.')