import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('models/fruits_classifier.h5')

# Load and preprocess an image
img_path = 'data/test/watermelon/Image_4.jpg' 
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])  

# Map the index to the class label
class_labels = {0: 'apple', 1: 'banana', 2: 'eggplant', 3: 'garlic', 4: 'grapes', 5: 'jalapeno', 6: 'orange', 7: 'paprika', 8: 'pineapple', 9: 'potato', 10: 'tomato', 11: 'watermelon'}
predicted_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_label}")
