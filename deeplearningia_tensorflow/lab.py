import os
import base64
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml

handdigits_mnist=fetch_openml('mnist_784',as_frame=False)

x,y = handdigits_mnist.data, handdigits_mnist.target

# Convert y to integers
y = y.astype(np.uint8)

# Reshape X into (70000, 28, 28, 1)
X = x.reshape(-1, 28, 28, 1)

# Select first 60,000 samples (like TensorFlow MNIST)
X_train, y_train = X[:60000], y[:60000]

print("X shape:", X_train.shape)  # (60000, 28, 28, 1)
print("y shape:", y_train.shape)  # (60000,)

x_norm = X_train/X_train.max()

class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):
        
        # Check if the accuracy is greater or equal to 0.995
       if logs['accuracy'] >= .995:
                            
            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\nReached 99.5% accuracy so cancelling training!") 


def convolutional_model():
    """Returns the compiled (but untrained) convolutional model.

    Returns:
        tf.keras.Model: The model which should implement convolutions.
    """

    ## START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([ 
        tf.keras.Input(shape=(28,28,1)),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ]) 

    ### END CODE HERE ###

    # Compile the model
    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)
          
    return model


model = convolutional_model()

training_history = model.fit(x_norm, y_train, epochs=10, callbacks=[EarlyStoppingCallback()])

model.save("handwritingmodel.keras")