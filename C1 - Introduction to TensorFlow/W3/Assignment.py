import tensorflow as tf

print(tf.__version__)


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs['accuracy'] > 0.998):
                print("Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # YOUR CODE STARTS HERE
    training_images = training_images / 255.0
    training_images = training_images.reshape([60000, 28, 28, 1])
    test_images = test_images / 255.0
    test_images = test_images.reshape([10000, 28, 28, 1])
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # model fitting
    history = model.fit(training_images, training_labels, epochs=20, callbacks=[MyCallback()])
    # model fitting

    return history.epoch, history.history['accuracy'][-1]

_, _ = train_mnist_conv()
