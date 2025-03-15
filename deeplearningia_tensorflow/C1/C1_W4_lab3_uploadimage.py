import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from pathlib import Path
import random

model = tf.keras.models.load_model('deeplearningia_tensorflow/compressed_images.keras')
model.summary()
rescale_layer = tf.keras.layers.Rescaling(scale=1./255)

TRAIN_DIR = 'horse-or-human'
trainDirectory = Path.cwd() / "deeplearningia_tensorflow" / TRAIN_DIR / "horses"
file = trainDirectory/'horse50-2.png'

image = tf.keras.utils.load_img(file, target_size=(150, 150))
image = tf.keras.utils.img_to_array(image)
image = rescale_layer(image)
image = np.expand_dims(image, axis=0)
    
prediction = model.predict(image, verbose=0)[0][0]
if prediction > 0.5:
    print( " is a human")
else:
    print(" is a horse")

# -------------------- Visualizing Intermediate Representations--------------------

TRAIN_DIR = 'horse-or-human'
VAL_DIR = 'validation-horse-or-human'
trainDirectory = Path.cwd() / "deeplearningia_tensorflow" / TRAIN_DIR 
validationDirectory = Path.cwd() / "deeplearningia_tensorflow" / VAL_DIR
train_horse_dir = os.path.join(trainDirectory, 'horses')
train_human_dir = os.path.join(trainDirectory, 'humans')
validation_horse_dir = os.path.join(validationDirectory, 'horses')
validation_human_dir = os.path.join(validationDirectory, 'humans')
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
validation_horse_hames = os.listdir(validation_horse_dir)
validation_human_names = os.listdir(validation_human_dir)


# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.inputs, outputs = successive_outputs)

# Prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)
img = tf.keras.utils.load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = tf.keras.utils.img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Scale by 1/255
# Since you are not using a tf.data.Dataset you apply the normalization via a simple numpy vectorization
x /= 255

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x, verbose=False)

# These are the names of the layers, so you can have them as part of the plot
layer_names = [layer.name for layer in model.layers[1:]]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:

    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map

    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]

    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')

      # Tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x

    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()