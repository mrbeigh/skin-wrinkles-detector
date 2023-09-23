import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'datasets/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')  # Change class_mode to 'categorical' for multi-class classification

test_generator = test_datagen.flow_from_directory(
    'datasets/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')  # Change class_mode to 'categorical' for multi-class classification

# Model Architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary_crossentropy for binary classification
              metrics=['accuracy'])

# Training
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the trained model
model.save('saved_models/model.h5')
