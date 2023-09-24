import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Data Preparation
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    'datasets/train',
    target_size=(64, 64),  # Reduced image size
    batch_size=32,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    'datasets/test',
    target_size=(64, 64),  # Reduced image size
    batch_size=32,
    class_mode='binary')

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save the trained model
model.save('saved_models/model.h5')
