from keras.preprocessing.image import ImageDataGenerator
import sys
from keras.models import load_model

img_width, img_height = 64, 64

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 20000
nb_validation_samples = 3000
epochs = 800
batch_size = 32

model_name = sys.argv[1]

model = load_model(model_name)

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
    validation_steps=nb_validation_samples // batch_size)

model.save('models/classifier-{}-c.h5'.format(epochs))
