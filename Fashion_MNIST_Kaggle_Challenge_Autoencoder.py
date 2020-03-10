
import keras
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.layers import Input,Dense,Flatten, Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical

#import files
train_images = np.load("/Users/wzd/Desktop/Work/Fashion-MNIST/train_images.npy")
test_images = np.load("/Users/wzd/Desktop/Work/Fashion-MNIST/test_images.npy")
train_labels = pd.read_csv("/Users/wzd/Desktop/Work/Fashion-MNIST/train_labels.csv")
del train_labels['ID']

label_dictionary = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

def show_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

#preprocessing
train_images = train_images.reshape(-1, 28,28, 1)
test_images = test_images.reshape(-1, 28,28, 1)

train_images = train_images/255.0
test_images = test_images/255.0

from sklearn.model_selection import train_test_split

train_X,valid_X,train_ground,valid_ground = train_test_split(train_images,
                                                             train_images,
                                                             test_size=0.2,
                                                             random_state=13)


batch_size = 64
epochs = 3
inChannel = 1
x, y = 28, 28
input_images =Input(shape = (x, y, inChannel))
num_classes = 10

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

autoencoder = Model(input_images, decoder(encoder(input_images)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())


autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
autoencoder.save_weights('autoencoder.h5')

train_Y_one_hot = to_categorical(train_labels)

train_X,valid_X,train_label,valid_label = train_test_split(train_images,train_Y_one_hot,test_size=0.2,random_state=13)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out
encode = encoder(input_images)
full_model = Model(input_images,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())


for layer in full_model.layers[0:19]:
    layer.trainable = False

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=3,verbose=1,validation_data=(valid_X, valid_label))

full_model.save_weights('autoencoder_classification.h5')

for layer in full_model.layers[0:19]:
    layer.trainable = True

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=3,verbose=1,validation_data=(valid_X, valid_label))

full_model.save_weights('classification_complete.h5')

predicted_labels = full_model.predict(test_images)
predicted_labels = np.argmax(np.round(predicted_labels),axis=1)

df_test = pd.read_csv('/Users/wzd/Desktop/Work/Fashion-MNIST/sample_submission.csv')
df_test['label'] = predicted_labels
df_test.to_csv('/Users/wzd/Desktop/Work/Fashion-MNIST/submission.csv', index=False)
