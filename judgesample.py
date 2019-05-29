import numpy as np
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Activation

data = np.array([[1,1,1,1,1],
                 [1,1,2,1,1],
                 [1,1,1,1,2],
                 [1,2,1,1,1],
                 [1,1,1,2,1],
                 [1,3,2,1,1],
                 [1,2,3,1,1],
                 [1,1,3,1,1],
                 [1,2,3,2,1],
                 [1,3,3,2,1],
                 [2,0,0,0,0],
                 [2,0,3,0,0],
                 [2,0,0,2,3],
                 [2,0,0,0,3],
                 [2,0,3,0,2],
                 [2,3,3,2,3],
                 [2,3,2,2,1],
                 [2,0,3,2,1],
                 [2,2,2,3,0],
                 [2,3,2,3,0]])
print(data)

labels = data[:,0]
data = data[:,1:5]
labels[labels==2]=0
labels = np_utils.to_categorical(labels)
print(data)
print(labels)

model = Sequential()
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data,labels,batch_size=5,epochs=100,validation_split=0.3)


test = np.array([[1,3,1,1,1],
                 [1,1,2,2,1],
                 [1,1,2,1,2],
                 [1,2,2,1,3],
                 [1,1,1,2,1],
                 [1,2,2,1,1],
                 [1,2,1,3,1],
                 [1,1,3,1,1],
                 [1,2,1,1,3],
                 [1,3,1,2,3],
                 [2,0,0,3,0],
                 [2,0,3,2,0],
                 [2,0,3,2,3],
                 [2,0,3,0,3],
                 [2,0,3,2,2],
                 [2,3,3,2,3],
                 [2,3,2,2,1],
                 [2,0,3,2,2],
                 [2,2,2,3,2],
                 [2,3,2,3,1]])

real = test[:,0]
test = test[:,1:5]
real[real==2]=0
predict = np.argmax(model.predict(test),axis=1)
sum(predict == real)/20.0
