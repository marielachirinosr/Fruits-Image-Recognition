import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# Load the data
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# Define a list of class labels(new model)
class_labels = ['apple', 'banana','eggplant','garlic', 'grapes', 'jalapeno', 'orange', 'paprika', 'pineapple', 'potato', 'tomato', 'watermelon']

# Create ImageDataGenerator objects
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8, 1.2],
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8, 1.2],
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  #'binary' for binary classification and 'categorical' for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'  
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

#Early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_labels), activation='softmax'))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) 
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)


# Save the model
os.makedirs('models', exist_ok=True) 
model.save('models/fruits_classifier.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)
print("Training complete. Evaluating model...")
loss, accuracy = model.evaluate(validation_generator)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Make predictions

"""TEST MODEL(apple and pineapple only)
img_path = 'data/test/1/308.jpg'  
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make a prediction
predictions = model.predict(img_array)
predicted_class = 'apple' if np.argmax(predictions[0]) == 0 else 'pineapple'
"""

# Load and preprocess an image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  
    return img_array

def predict_image_class(model, img_path):
    img_array = preprocess_image(img_path)  
    predictions = model.predict(img_array)  
    predicted_class = np.argmax(predictions[0])

    predicted_label = class_labels[predicted_class]
    
    return predicted_label

# Example usage
img_path = 'data/test/potato/Image_3.jpg' 
predicted_label = predict_image_class(model, img_path)
print(f"Predicted class: {predicted_label}")

# Evaluate the precision of the model
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

