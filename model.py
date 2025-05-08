import tensorflow as tf

# === Initialize parameters (weights and biases) ===
def initialize_model():
    initializer = tf.keras.initializers.GlorotNormal()
    W1 = tf.Variable(initializer(shape=(512, 3072)))
    b1 = tf.Variable(initializer(shape=(512, )))
    W2 = tf.Variable(initializer(shape=(256, 512)))
    b2 = tf.Variable(initializer(shape=(256, )))
    W3 = tf.Variable(initializer(shape=(128, 256)))
    b3 = tf.Variable(initializer(shape=(128, )))
    W4 = tf.Variable(initializer(shape=(64, 128)))
    b4 = tf.Variable(initializer(shape=(64, )))
    W5 = tf.Variable(initializer(shape=(10, 64)))
    b5 = tf.Variable(initializer(shape=(10, )))

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2,
                  "W3": W3, "b3": b3,
                  "W4": W4, "b4": b4,
                  "W5": W5, "b5": b5}
    
    return parameters


# === Forward Propagation ===
def forward_propagation(X, parameters):
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']
    W4, b4 = parameters['W4'], parameters['b4']
    W5, b5 = parameters['W5'], parameters['b5']

    X = tf.reshape(X, [-1, 3072])

    # === Apply linear transformations and activation functions ===
    Z1 = tf.linalg.matmul(X, tf.transpose(W1)) + b1
    Z1 = tf.keras.layers.BatchNormalization()(Z1)
    A1 = tf.keras.activations.relu(Z1)
    A1 = tf.keras.layers.Dropout(0.5)(A1)

    Z2 = tf.linalg.matmul(A1, tf.transpose(W2)) + b2
    Z2 = tf.keras.layers.BatchNormalization()(Z2)
    A2 = tf.keras.activations.relu(Z2)
    A2 = tf.keras.layers.Dropout(0.5)(A2)

    Z3 = tf.linalg.matmul(A2, tf.transpose(W3)) + b3
    Z3 = tf.keras.layers.BatchNormalization()(Z3)
    A3 = tf.keras.activations.relu(Z3)
    A3 = tf.keras.layers.Dropout(0.5)(A3)

    Z4 = tf.linalg.matmul(A3, tf.transpose(W4)) + b4
    Z4 = tf.keras.layers.BatchNormalization()(Z4)
    A4 = tf.keras.activations.relu(Z4)
    A4 = tf.keras.layers.Dropout(0.5)(A4)

    Z5 = tf.linalg.matmul(A4, tf.transpose(W5)) + b5

    return Z5
