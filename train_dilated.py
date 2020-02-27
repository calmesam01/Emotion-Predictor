from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D, AtrousConvolution2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import backend as K
from keras.callbacks import EarlyStopping

img_width, img_height = 32, 32

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 20000
nb_validation_samples = 3000
epochs = 150
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

early_stopping = EarlyStopping(patience=5, mode='auto')
    
model = Sequential()
model.add(Conv2D(10, (5, 5), input_shape=input_shape))
model.add(LeakyReLU(alpha=.003))

model.add(Conv2D(64, (3, 3), padding="same", dilation_rate=1))
model.add(LeakyReLU(alpha=.003))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))

model.add(Conv2D(10, (3, 3)))
model.add(LeakyReLU(alpha=.003))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(10, (2, 2)))
model.add(LeakyReLU(alpha=.003))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(10, (1, 1)))
model.add(LeakyReLU(alpha=.003))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))



model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.35))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[early_stopping])

model.save('models/classifier-dilated-{}.h5'.format(epochs))
