import os
import random
import numpy as np
from io import BytesIO

# Plotting and dealing with images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

# Interactive widgets
from ipywidgets import widgets

TRAIN_DIR = 'horse-or-human'

print(f"files in current directory: {os.listdir()}")
trainDirectory="deeplearningia_tensorflow/"+TRAIN_DIR

print(f"\nsubdirectories within '{trainDirectory}' dir: {os.listdir(trainDirectory)}")

# Directory with the training horse pictures
train_horse_dir = os.path.join(trainDirectory, 'horses')

# Directory with the training human pictures
train_human_dir = os.path.join(trainDirectory, 'humans')

# Check the filenames
train_horse_names = os.listdir(train_horse_dir)
print(f"5 files in horses subdir: {train_horse_names[:5]}")
train_human_names = os.listdir(train_human_dir)
print(f"5 files in humans subdir:{train_human_names[:5]}")

print(f"total training horse images: {len(os.listdir(train_horse_dir))}")
print(f"total training human images: {len(os.listdir(train_human_dir))}")


# nrows = 4
# ncols = 4
# fig = plt.gcf()
# fig.set_size_inches(ncols * 3, nrows * 3)

# next_horse_pix = [os.path.join(train_horse_dir, fname)
#                 for fname in random.sample(train_horse_names, k=8)]
# next_human_pix = [os.path.join(train_human_dir, fname)
#                 for fname in random.sample(train_human_names, k=8)]

# for i, img_path in enumerate(next_horse_pix + next_human_pix):
#     sp = plt.subplot(nrows, ncols, i + 1)
#     sp.axis('Off') 
#     img = mpimg.imread(img_path)
#     plt.imshow(img)

# plt.show()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.Input(shape=(300, 300, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
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
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0 to 1 where 0 is for 'horses' and 1 for 'humans'
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

train_dataset = tf.keras.utils.image_dataset_from_directory(
    trainDirectory,
    image_size=(300,300),
    batch_size=32,
    label_mode='binary'
)

dataset_type = type(train_dataset)
print(f'train_dataset inherits from tf.data.Dataset: {issubclass(dataset_type, tf.data.Dataset)}')

# Get one batch from the dataset
sample_batch = list(train_dataset.take(1))[0]

# Check that the output is a pair
print(f'sample batch data type: {type(sample_batch)}')
print(f'number of elements: {len(sample_batch)}')

# Extract image and label
image_batch = sample_batch[0]
label_batch = sample_batch[1]

# Check the shapes
print(f'image batch shape: {image_batch.shape}')
print(f'label batch shape: {label_batch.shape}')

print(image_batch[0].numpy())

# Check the range of values
print(f'max value: {np.max(image_batch[0].numpy())}')
print(f'min value: {np.min(image_batch[0].numpy())}')

rescale_layer = tf.keras.layers.Rescaling(scale=1./255)

image_scaled = rescale_layer(image_batch[0]).numpy()

print(image_scaled)

print(f'max value: {np.max(image_scaled)}')
print(f'min value: {np.min(image_scaled)}')

train_dataset_scaled = train_dataset.map(lambda image, label: (rescale_layer(image), label))
# Get one batch of data
sample_batch =  list(train_dataset_scaled.take(1))[0]

# Get the image
image_scaled = sample_batch[0][1].numpy()

# Check the range of values for this image
print(f'max value: {np.max(image_scaled)}')
print(f'min value: {np.min(image_scaled)}')

SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = (train_dataset_scaled
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                      )

history = model.fit(
    train_dataset_final,
    epochs=15,
    verbose=2
    )

model.save("C1W4.keras")