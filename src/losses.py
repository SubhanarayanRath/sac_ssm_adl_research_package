import tensorflow as tf

def activity_loss(y_true,y_pred): return tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred)
def boundary_loss(y_true,y_pred): return tf.keras.losses.binary_crossentropy(y_true,y_pred)
def phase_loss(y_true,y_pred): return tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred)