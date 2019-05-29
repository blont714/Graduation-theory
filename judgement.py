import numpy as np
from keras import backend as kb
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Dense,Embedding
from keras import optimizers
from keras.utils import np_utils

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()
 
    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


def build_model():
    model = Sequential()
    model.add(Embedding(5,2,input_length=50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    
    batch_size = 10
    epochs = 5
    
    # 多層ニューラルネットワークモデルを構築
    model = build_model()
    
    # モデルのサマリを表示
    #model.summary()
    
    # モデルをコンパイル
    # 確率的勾配降下法オプティマイザ．
    # モーメンタム，学習率減衰，Nesterov momentumをサポートした確率的勾配降下法．
    sgd = optimizers.SGD(lr=0.1,decay=0.0,momentum=0.0,nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    train_X = train.flow_from_directory(
        'data/train',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')

    validation_X = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical')
    
    
    # モデルの訓練
    history = model.fit(X_train,Y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        verbose=1,
                        validation_split=0.1)
    
    # モデルの評価
    loss, acc = model.evaluate(X_test,Y_test,verbose=0)
    
    print('Test loss:', loss)
    print('Test acc:', acc)  
    
    plot_history(history)      
    
    kb.clear_session()
