import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2

data = []
labels = []
paths = []
batch_size = 64
img_size = (48,48)
channels = 3
img_shape = (img_size[0],img_size[1],channels)
valid_gen = ImageDataGenerator(rescale=1./255)
dataset_path = '/Users/pranaymishra/Desktop/task_wizz/archive/test'

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        paths.append(img_path)
        img = cv2.imread(img_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)
        labels.append(class_name)
        
        
df = pd.DataFrame({'data': paths, 'label': labels})
data = np.array(data)
labels = np.array(labels)

# Load the CNN model
loaded_model = load_model('/Users/pranaymishra/Desktop/task_wizz/model_optimal.h5')

# Evaluate the model on the test set
test_gen = valid_gen.flow_from_dataframe(dataframe=df, x_col='data', y_col='label', target_size=img_size,
                                          class_mode='categorical', color_mode='grayscale', shuffle=False,
                                          batch_size=batch_size)

# Predict labels for the test set
test_pred_probs = loaded_model.predict(test_gen)
test_pred_labels = np.argmax(test_pred_probs, axis=1)

# Get true labels
test_true_labels = test_gen.classes

# Evaluate the model on the test set
strat = df['label']
train_df,validate_df = train_test_split(df,train_size = 0.80,shuffle = True, random_state = 42, stratify = strat)
train_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range = 0.1,   
    height_shift_range = 0.1,
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)
train_gen = train_gen.flow_from_dataframe(dataframe = train_df, x_col = 'data' ,y_col = 'label', target_size = img_size, class_mode = 'categorical',color_mode = 'grayscale'
                                      , shuffle = True, batch_size = batch_size)
validate_gen = valid_gen.flow_from_dataframe(dataframe = validate_df,x_col = 'data' ,y_col = 'label', target_size = img_size, class_mode = 'categorical',color_mode = 'grayscale',
                                          shuffle = True, batch_size = batch_size)
class_names = list(train_gen.class_indices.keys())

def predict_single_image(img_path, model, class_names):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    predicted_class_name = class_names[predicted_label]

    print(f"Predicted Label: {predicted_class_name}")
    return predicted_class_name

# Test a single image
# test_image_path = '/Users/pranaymishra/Downloads/peopleimages_6.jpg' 
# predict_single_image(test_image_path, loaded_model, class_names)
