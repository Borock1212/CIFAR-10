import tensorflow as tf
from sklearn.model_selection import train_test_split

# === Data normalization function ===
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    return image, label

    # === Loading CIFAR-10 dataset ===
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

    # === Converting data to tf.data.Dataset ===
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(normalize).shuffle(10000).batch(32).repeat().prefetch(8)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.map(normalize).batch(32).prefetch(8)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(normalize).batch(32).prefetch(8)

    steps_per_epoch = len(x_train) // 32

    return train_dataset, val_dataset, test_dataset, steps_per_epoch