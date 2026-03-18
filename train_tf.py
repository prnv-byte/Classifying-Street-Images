import tensorflow as tf
from tensorflow.keras import layers, models

print("Downloading real image dataset (CIFAR-10)...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

print("Building the Convolutional Neural Network (CNN)...")
model = models.Sequential()

# --- THE FEATURE EXTRACTORS ---
# FIX: The modern way to tell the AI the size of the image!
model.add(layers.Input(shape=(32, 32, 3)))

# Now we slide the magnifying glasses
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# --- THE CLASSIFIER ---
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting Training! (This may take a few minutes...)")
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

print("\nSaving the professional AI Brain...")
model.save('tf_vision_model.keras')
print("Saved successfully as 'tf_vision_model.keras'!")