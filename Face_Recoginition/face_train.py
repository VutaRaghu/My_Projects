# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:25:13 2019

@author: raghu
"""

import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


raghu_dir = os.path.join(r'C:\Users\raghu\Desktop\faces\raghu')

# Directory with dandelion pictures
rajesh_dir = os.path.join(r'C:\Users\raghu\Desktop\faces\rajesh')
# Directory with rose pictures
srikanth_dir = os.path.join(r'C:\Users\raghu\Desktop\faces\srikanth')




train_raghu_names = os.listdir(raghu_dir)
print(train_raghu_names[:5])

train_rajesh_names = os.listdir(rajesh_dir)
print(train_rajesh_names[:5])

train_srikanth_names = os.listdir(srikanth_dir)
print(train_srikanth_names[:5])

"""
print('total daisy images:', len(os.listdir(daisy_dir)))
print('total dandelion images:', len(os.listdir(dandelion_dir)))
print('total rose images:', len(os.listdir(rose_dir)))
print('total sunflower images:', len(os.listdir(sunflower_dir)))
print('total tulip images:', len(os.listdir(tulip_dir)))


nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_daisy_pix = [os.path.join(daisy_dir, fname) 
                for fname in train_daisy_names[pic_index-8:pic_index]]
next_rose_pix = [os.path.join(rose_dir, fname) 
                for fname in train_rose_names[pic_index-8:pic_index]]

print ("Showing some daisy pictures...")
print()
for i, img_path in enumerate(next_daisy_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print ("Showing some rose pictures...")
print()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_rose_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

"""

batch_size = 20#128



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'C:\Users\raghu\Desktop\faces',  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200 x 200
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['raghu','rajesh','srikanth'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(200,200)
#$input_shape = tuple(list(target_size)+[3])
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()


# Optimizer and compilation
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])#binary cross entropy,categorical_crossentropy
# Total sample count
total_sample=train_generator.n
# Training
num_epochs =20
#model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),epochs=num_epochs,verbose=1)







history = model.fit_generator(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=num_epochs,
        verbose=1)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")



#reuse



