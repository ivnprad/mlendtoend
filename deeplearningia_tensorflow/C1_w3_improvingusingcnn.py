import tensorflow as tf
import  matplotlib.pyplot as plt
import numpy as np

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        '''
        Halts the training when the loss falls below 0.4

        Args:
            epoch (integer) - index of epoch (required but unused in the function definition below)
            logs (dict) - metric results from the training epoch
        '''

        # Check the loss
        if logs['loss'] < 0.4:

            # Stop if threshold is met
            print("\nLoss is lower than 0.4 so cancelling training!")
            self.model.stop_training = True
# model = tf.keras.models.Sequential([
#     tf.keras.Input(shape=(28,28,1)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128,activation=tf.nn.relu),
#     tf.keras.layers.Dense(10,activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# print("\nMODEL TRAINING:")
# model.fit(training_images, training_labels, epochs=5)

# print("\nMODEL EVALUATION:")
# test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
# print(f'test set accuracy: {test_accuracy}')
# print(f'test set loss: {test_loss}')

cnnmodel = tf.keras.models.Sequential([
    #add convolutions and max pooling
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #add the same layers as before
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

cnnmodel.summary()

# Use same settings
cnnmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("\nMODEL TRAINING:")
cnnmodel.fit(training_images, training_labels, epochs=5, callbacks=[myCallback()])

# Evaluate on the test set
print("\nMODEL EVALUATION:")
test_loss, test_accuracy = cnnmodel.evaluate(test_images, test_labels, verbose=0)
print(f'test set accuracy: {test_accuracy}')
print(f'test set loss: {test_loss}')

cnnmodel.save('cnnmodel.h5')

