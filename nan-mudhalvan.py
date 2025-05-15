# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Create image data generators with data augmentation for training and rescaling for both
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values
    rotation_range=40,           # Randomly rotate images
    width_shift_range=0.2,       # Randomly shift images horizontally
    height_shift_range=0.2,      # Randomly shift images vertically
    shear_range=0.2,             # Apply shear transformation
    zoom_range=0.2,              # Apply zoom
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest'          # Fill in new pixels with nearest values
)

# Only rescale test images, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Define paths to training and test image directories
train_dir = 'path_to_train_images'
test_dir = 'path_to_test_images'

# Create training image generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),      # Resize images
    batch_size=32,               # Images per batch
    class_mode='binary'          # Binary classification (0 or 1)
)

# Create testing image generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # First convolutional layer
    MaxPooling2D(2, 2),                                                # Max pooling

    Conv2D(64, (3, 3), activation='relu'),                             # Second convolutional layer
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),                            # Third convolutional layer
    MaxPooling2D(2, 2),

    Flatten(),                                                        # Flatten feature maps into 1D vector
    Dense(512, activation='relu'),                                    # Fully connected hidden layer
    Dense(1, activation='sigmoid')                                    # Output layer with sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',                          # Optimizer
    loss='binary_crossentropy',                # Loss function for binary classification
    metrics=['accuracy']                       # Track accuracy during training
)

# Train the model for 10 epochs
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator            # Use test data for validation
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Save the trained model to a file
model.save('diagnostic_imaging_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
