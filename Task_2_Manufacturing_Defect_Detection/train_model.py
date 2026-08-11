import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Image settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Paths
train_path = "dataset/train"
test_path = "dataset/test"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training images
train_data = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary")

# Load testing images
test_data = test_datagen.flow_from_directory(test_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False)

# Display class labels
print("Class indices:", train_data.class_indices)

# Build CNN model
model = Sequential([Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224,3)), MaxPooling2D(2, 2), Conv2D(64, (3, 3), activation="relu"), MaxPooling2D(2, 2), Conv2D(128, (3, 3), activation="relu"), MaxPooling2D(2, 2), Flatten(), Dense(128, activation="relu"), Dropout(0.5), Dense(1, activation="sigmoid")])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display model architecture
model.summary()

# Train model
history = model.fit(train_data, validation_data=test_data, epochs=10)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save trained model
model.save("defect_detection_model.keras")
print("Model saved successfully!")