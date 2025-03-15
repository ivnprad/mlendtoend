#from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

loaded_model = tf.keras.models.load_model('my_model.keras')

# Evaluate on the test set
print("\nMODEL EVALUATION:")
test_loss, test_accuracy = loaded_model.evaluate(test_images, test_labels, verbose=0)
print(f'test set accuracy: {test_accuracy}')
print(f'test set loss: {test_loss}')

#loaded_model.save('my_model.keras')

print(f"First 100 labels:\n\n{test_labels[:100]}")

print(f"\nShoes: {[i for i in range(100) if test_labels[:100][i]==9]}")

print(loaded_model.summary())

FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
LAYER_CHANNEL = 1
layers_to_visualize = [tf.keras.layers.Conv2D, tf.keras.layers.MaxPooling2D]

layer_outputs = [layer.output for layer in loaded_model.layers if type(layer) in layers_to_visualize]
activation_model = tf.keras.models.Model(inputs = loaded_model.inputs, outputs=layer_outputs)

f, axarr = plt.subplots(3,len(layer_outputs))

for layerIdx in range(len(layer_outputs)):
    LayerOutputFirstImage = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1), verbose=False)[layerIdx]
    axarr[0,layerIdx].imshow(LayerOutputFirstImage[0, :, :, LAYER_CHANNEL], cmap='inferno')
    axarr[0,layerIdx].grid(False)
  
    LayerOutputSecondImage = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1), verbose=False)[layerIdx]
    axarr[1,layerIdx].imshow(LayerOutputSecondImage[0, :, :, LAYER_CHANNEL], cmap='inferno')
    axarr[1,layerIdx].grid(False)
  
    LayerOutputThirdImage = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1), verbose=False)[layerIdx]
    axarr[2,layerIdx].imshow(LayerOutputThirdImage[0, :, :, LAYER_CHANNEL], cmap='inferno')
    axarr[2,layerIdx].grid(False)

plt.tight_layout()  
plt.show()



