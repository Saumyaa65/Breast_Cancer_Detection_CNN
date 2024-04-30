import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler # for feature scaling

cancer= datasets.load_breast_cancer()

# matrix of features
x=pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
y=cancer.target
# cancer.target_names= '0 malignant', '1 benign'
# x.shape= (569, 30), y.shape= (569,)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,
                                                   random_state=0)
# x_train.shape= (455, 30), x_test.shape= (114, 30)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
x_train=x_train.reshape(455,30,1)
x_test=x_test.reshape(114,30,1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, input_shape=[30,1],
                                 activation='relu'))
# Batch Normalisation: helps each layer of model learn more by itself
# independent of other layers to make processing faster
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

opt=tf.keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

y_pred=(model.predict(x_test) > 0.5).astype ('int32')
print(y_pred[0], y_test[0])
print(y_pred[-1], y_test[-1])
print(y_pred[100], y_test[100])
print(y_pred[25], y_test[25])
print(y_pred[60], y_test[60])

cm=confusion_matrix(y_test, y_pred)
print(cm)

acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)

epoch_range= range(1, 51)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()

plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train', 'val'], loc='upper left')
plt.show()
