import tensorflow as tf

# List available GPUs
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))