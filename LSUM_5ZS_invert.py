#!/usr/bin/env python
# coding: utf-8

# In[49]:


import os
import pytimber
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import GaussianNoise
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau

get_ipython().run_line_magic('matplotlib', 'inline')


# In[177]:


#train_data = pd.read_csv('../../ZSdata/train_data_01_11_2018_norm.csv')
pred_data = pd.read_csv('../../ZSdata/test_data_30_03_2018_norm.csv')
train_data = pd.read_csv('../../ZSdata/train_data_01_11_2018_norm.csv')
#pred_data = pd.read_csv('../../ZSdata/test_data_26_04_2018_norm.csv')
#pred_data = pd.read_csv('../../ZSdata/test_data_26_04_2018_3_norm.csv')
#pred_data = pd.read_csv('../../ZSdata/test_data_05_04_2018_norm.csv')
#pred_data = pd.read_csv('../../ZSdata/test_data_26_04_2018_3_norm.csv')
#pred_data = pd.read_csv('../../ZSdata/test_data_30_03_2018_norm.csv')
#pred_data3 = pd.read_csv('../../ZSdata/TIMBER_DATA_2017_05.csv')

target = ['SPS.BLM.21636.ZS1:LOSS_CYCLE',
          'SPS.BLM.21652.ZS2:LOSS_CYCLE',
          'SPS.BLM.21658.ZS3:LOSS_CYCLE',
          'SPS.BLM.21674.ZS4:LOSS_CYCLE',
          'SPS.BLM.21680.ZS5:LOSS_CYCLE',
          'SPS.BLM.21694.TCE:LOSS_CYCLE',
          'SPS.BLM.21772.TPST:LOSS_CYCLE',
          'SPS.BLM.21775.MST1:LOSS_CYCLE',
          'SPS.BLM.21776.MST1:LOSS_CYCLE',
          'SPS.BLM.21792.MST3:LOSS_CYCLE',]

predictors =  ['ZS1.LSS2.ANODE:UP_PPM','ZS1.LSS2.ANODE:DO_PPM',
               'ZS2.LSS2.ANODE:UP_PPM','ZS2.LSS2.ANODE:DO_PPM',
               'ZS3.LSS2.ANODE:UP_PPM','ZS3.LSS2.ANODE:DO_PPM',
               'ZS4.LSS2.ANODE:UP_PPM','ZS4.LSS2.ANODE:DO_PPM',
               'ZS5.LSS2.ANODE:UP_PPM','ZS5.LSS2.ANODE:DO_PPM',]

#predictors = ['ZS1.LSS2.ANODE:DO_PPM',
#              'ZS2.LSS2.ANODE:UP_PPM','ZS2.LSS2.ANODE:DO_PPM',
#              'ZS3.LSS2.ANODE:UP_PPM','ZS3.LSS2.ANODE:DO_PPM',]

new_train_data = train_data[predictors]

target_data = train_data[target]
pred_y_data = pred_data[target]

Xmin = min(np.min(new_train_data))
Xmax = max(np.max(new_train_data))

target_data = (target_data-Nmin)/(Nmax-Nmin)

#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    new_train_data.values,target_data.values,test_size=0.25, random_state=1)

indim = X_train.shape[1]
outdim = y_train.shape[1]


# In[178]:


def setupSM(idim=indim, odim = outdim, units=8):
    model = Sequential()
    model.add(Dense(units=units, input_dim=idim, activation='relu')) 
    model.add(Dropout(0.0))
    model.add(Dense(units=units, input_dim=idim, activation='tanh')) 
    model.add(Dense(units=outdim, activation= 'linear'))
    return model


# In[179]:


optmiser = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=0.00001, decay=0.0000001)
SM = setupSM()
SM.compile(loss='mse', optimizer=optmiser, metrics=['mse'])
print(SM.summary())
batch_size = 32
hist = SM.fit(x=X_train, y=y_train,
        epochs=250,
        #shuffle = True,
        batch_size=batch_size,
        validation_data=(X_test, y_test))


# In[180]:


# summarize history for loss
plt.plot(hist.history['loss'][1:])
plt.plot(hist.history['val_loss'][1:])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.grid()
plt.show()

# summarize history for loss
plt.plot(hist.history['mean_squared_error'][1:])
plt.plot(hist.history['val_mean_squared_error'][1:])
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.grid()
plt.show()


# In[181]:


temp_preds = SM.predict(pred_data[predictors])
test_preds = 0*temp_preds
for i in range(temp_preds.shape[0]):
    test_preds[i,:] = temp_preds[i,:]*(Nmax-Nmin)+Nmin

# summarize history for loss
plt.plot(pred_y_data,'g-')
plt.title('5/4/18')
plt.plot(test_preds,'r-')
plt.xlabel('test #')
plt.ylabel('normalised loss [Gy/e10 p]')
plt.grid()
plt.show()

# scatter
for i in range(test_preds.shape[1]):
    plt.plot((pred_y_data),(test_preds),'.')
plt.plot(np.linspace(0,1,1000),np.linspace(0,1,1000),'-r')
plt.xlabel('measured normalised loss [Gy/e10 p]')
plt.ylabel('SM prediction normalised loss [Gy/e10 p]')
plt.ylim([0.0,0.0004])
plt.xlim([0.0,0.0004])
plt.grid()
plt.show()

for i in range(test_preds.shape[1]):
    plt.plot(pred_y_data.iloc[:,i],'g-')
    plt.title('5/4/18 BLM' + str(i+1))
    plt.plot(test_preds[:,i],'r-')
    plt.xlabel('test #')
    plt.ylabel('normalised loss [Gy/e10 p]')
    plt.grid()
    plt.show()



# In[182]:


leg = [item[14:-11] for item in target]
out = np.zeros((40*indim,indim))

preds = np.arange(-2,2,0.1)
for i in range(indim):
    out[i*40:(i+1)*40,i] = preds
    
t_preds = SM.predict(out)
n_preds = t_preds *(Nmax-Nmin)+Nmin

print(n_preds.shape)

for j in range(indim):
    a = [sum(n_preds[k+j*40,:]) for k in range(40)]
    plt.plot(preds,a)
    plt.xlim((-2.5,2.5))
    plt.ylim((0,0.005))
    plt.grid()
    plt.xlabel(predictors[j])
    plt.ylabel('normalised SUM loss [Gy/e10 p]')
    plt.show()
    for i in range(outdim):
        plt.plot(preds,n_preds[40*j:40*(j+1),i])
        plt.legend(leg,loc='upper left')
        plt.xlim((-2.5,2.5))
        plt.ylim((0,0.0008))
        plt.grid()
        plt.xlabel(predictors[j] + ' [mm]')
        plt.ylabel('normalised loss [Gy/e10 p]')
    plt.show()


# In[176]:


# ================ Invert network ========

# anode ground truth, for comparison later
sample_num = 100
gta = new_train_data.iloc[sample_num,:]

# start with random anode alignment
zp = tf.Variable(np.random.uniform(-0.1, 0.1, size=(1,indim)), dtype=tf.float32)

# define the target loss pattern
fz = tf.Variable(target_data.iloc[sample_num,:], tf.float32)
fz = tf.expand_dims(fz, 0)
#fz = tf.cast(fz,tf.float32)

# define the model and loss function
fzp = SM(zp)
loss = tf.losses.mean_squared_error(labels=fz, predictions=fzp)

# gradient descent definition
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.99
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.005)
opt = tf.train.GradientDescentOptimizer(learning_rate)

# optimize on the variable zp
train = opt.minimize(loss, var_list=zp, global_step=global_step)

# perform the optimization
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000): # Use more iterations (10000)
    _, loss_value, zp_val, eta = sess.run((train, loss, zp, learning_rate))
    z_loss = np.sqrt(np.sum(np.square(zp_val[0] - gta.values))/len(zp_val[0]))
    print("%03d) eta=%03f, loss = %f, z_loss = %f" % (i, eta, loss_value, z_loss))

# check the recovered anode vector
zp_val = sess.run(zp)
print(zp_val[0])
print(gta.values)


# In[ ]:




