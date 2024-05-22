import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL as pil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import seaborn as sns

meta = pd.read_csv(r"C:\Users\peech\OneDrive\Desktop\Dermitology Detection-Using AI\Skin-Disease-Prediction-Web-Application\HAM10000_metadata.csv")
print(meta.info())
print(meta.head())

g = sns.catplot(x="dx", kind="count", palette='bright', data=meta)
g.fig.set_size_inches(16, 5)
g.ax.set_title('Skin Cancer by Class', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)

g = sns.catplot(x="dx", kind="count", hue="sex", palette='coolwarm', data=meta)
g.fig.set_size_inches(16, 5)
g.ax.set_title('Skin Cancer by Sex', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)
g._legend.set_title('Sex')

g = sns.catplot(x="dx", kind="count", hue="age", palette='bright', data=meta)
g.fig.set_size_inches(16, 9)
g.ax.set_title('Skin Cancer by Age', fontsize=20)
g.set_xlabels('Skin Cancer Class', fontsize=14)
g.set_ylabels('Number of Data Points', fontsize=14)
g._legend.set_title('Age')

df = pd.read_csv(r"C:\Users\peech\OneDrive\Desktop\Dermitology Detection-Using AI\Skin-Disease-Prediction-Web-Application\hmnist_28_28_RGB.csv")

x = df.drop('label', axis=1)
y = df['label']
x = x.to_numpy()
x = x / 255
y = to_categorical(y)

df['label'].value_counts()
print(df['label'].value_counts())

label = {
    'Actinic keratoses': 0,
    'Basal cell carcinoma': 1,
    'Benign keratosis-like lesions': 2,
    'Dermatofibroma': 3,
    'Melanocytic nevi': 4,
    'Melanoma': 5,
    'Vascular lesions': 6
}

x = x.reshape(-1, 28, 28, 3)
print(x.shape)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=13, stratify=df['label'])

datagen = ImageDataGenerator(rotation_range=20,
                             width_shift_range=0.10,
                             height_shift_range=0.10,
                             rescale=1/255,
                             shear_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

datagen.fit(xtrain)

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential()

model.add(Conv2D(64, (2, 2), input_shape=(28, 28, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(1024, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Conv2D(1024, (1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[accuracy])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early = EarlyStopping(monitor='val_accuracy', patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1, mode='min', min_lr=0.0001)

class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 0.5, 5: 1, 6: 1}

history = model.fit(xtrain, ytrain, epochs=30, validation_data=(xtest, ytest), callbacks=[reduce_lr, early], class_weight=class_weights)

plt.figure(figsize=(15, 10))
loss = pd.DataFrame(history.history)
loss[['accuracy', 'val_accuracy']].plot()

plt.figure(figsize=(15, 10))
loss[['loss', 'val_loss']].plot()

decode = {
    0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular lesion'
}

plt.figure(figsize=(10, 8))

pred = model.predict(xtest)

from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(ytest[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(7):
    plt.plot(fpr[i], tpr[i], label=decode[i], linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='random guess')
plt.legend(loc="lower right")

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(xtest)
predicted_classes = np.argmax(predictions, axis=1)

model.save(r"C:\Users\peech\OneDrive\Desktop\Dermitology Detection-Using AI\Skin-Disease-Prediction-Web-Application\ham28(93&89).h5")

check = []
for i in range(len(ytest)):
    for j in range(7):
        if ytest[i][j] == 1:
            check.append(j)
check = np.asarray(check)
print(classification_report(check, predicted_classes))
