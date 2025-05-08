import tensorflow as tf 
from model import forward_propagation

# === Compute Loss ===
def compute_total_loss(logits, labels):
    labels = tf.squeeze(tf.one_hot(labels, depth=10), axis=1)
    total_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))
    return total_loss

# === Training Loop ===
def train_model(train_dataset, val_dataset, parameters, steps_per_epoch, learning_rate=0.001, epochs=10):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    val_acc = tf.keras.metrics.CategoricalAccuracy()
    history = []

    for epoch in range(epochs):
        total_loss = 0
        train_acc.reset_state()
        val_acc.reset_state()
        for step, (X_batch, Y_batch) in enumerate(train_dataset):
            if step >= steps_per_epoch:
                break

            with tf.GradientTape() as tape:
                Z5 = forward_propagation(X_batch, parameters)
                loss = compute_total_loss(Z5, Y_batch)

            gradients = tape.gradient(loss, list(parameters.values()))
            optimizer.apply_gradients(zip(gradients, parameters.values()))

            total_loss += loss
            train_acc.update_state(tf.one_hot(Y_batch, depth=10), Z5)
        
        total_loss /= steps_per_epoch

        for X_val, Y_val in val_dataset:
            Z5_val = forward_propagation(X_val, parameters)
            val_acc.update_state(tf.one_hot(Y_val, depth=10), Z5_val)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.numpy():.4f}, Train acc: {train_acc.result().numpy():.4f}, Val acc: {val_acc.result().numpy():.4f}")
        history.append((total_loss.numpy(), train_acc.result().numpy(), val_acc.result()))

    return history
