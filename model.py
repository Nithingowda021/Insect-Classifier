from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow_from_directory(
        'E:/CP-2020/Insect Project/My_Code/Insects/',
        target_size=(200, 200),
        batch_size=batch_size,

        classes = ['Auchenorrhyncha', 'Heteroptera', 'Hymenoptera', 'Lepidoptera', 'Megalptera', 'Neuroptera', 'Odonata','Orthoptera', 'Diptera'],

        class_mode='categorical')

import tensorflow as tf

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(9, activation='softmax')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

total_sample=train_generator.n

n_epochs = 5



"""
history = Model.fit(
        train_generator,
        steps_per_epoch=int(total_sample/batch_size),
        epochs=n_epochs,
        verbose=1)

"""
history = model.fit_generator(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=n_epochs,
        verbose=1)

model.save('model.h5')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('E:/CP-2020/Insect Project/My_Code/Diptera.jpg', target_size = (200,200))

test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
print(result)



if result[0][0] == 1:
    print("The result of classification is:"+"Auchenorrhyncha")
elif result[0][1] == 1:
    print("The result of classification is:"+"Heteroptera")
elif result[0][2] == 1:
    print("The result of classification is:"+"Hymenoptera")
elif result[0][3] == 1:
    print("The result of classification is:"+"Lepidoptera")
elif result[0][4] == 1:
    print("The result of classification is:"+"Megalptera")
elif result[0][5] == 1:
    print("The result of classification is:"+"Neuroptera")
elif result[0][6] == 1:
    print("The result of classification is:"+"Odonata")
elif result[0][7] == 1:
    print("The result of classification is:"+"Orthoptera")
elif result[0][8] == 1:
    print("The result of classification is:"+"Diptera")