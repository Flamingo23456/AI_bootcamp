import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import Flatten
from sklearn import accuracy_score


#Load Dtata
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.dtype)
print(x_train.shape)
print(y_test.shape)
print(x_train[0])
plt.imshow(x_train[0])
plt.show()
print("***********")
print(f"label is : {y_train[0]}")

#preprocessing
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

#to_categorical
print(f"before label is: {y_train[100]}")
y_train=to_categorical(y_train)
print(f"after label is: {y_train[100]}")
y_test=to_categorical(y_test)

#architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))#input layert
model.add(Dense(128,activation='relu'))#hidden layer
model.add(Dense(10, activation='softmax')) #last layer
print("#########")
#compile
model.compile(
    optimizer='adam',                   # defines how weights are updated
    loss='categorical_crossentropy',    # defines how loss is calculated
    metrics=['accuracy']                # defines which metrics to track
)

#train
model.fit(x_train,y_train,epochs=10,batch_size=64)

#test
pred=model.predict(x_test)
acc=accuracy_score(y_test,pred)


