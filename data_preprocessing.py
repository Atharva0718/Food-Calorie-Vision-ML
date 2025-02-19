import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ✅ Set absolute path to the dataset folder
DATASET_PATH = r"C:\Users\laptop\Downloads\food-101\food-101"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
META_PATH = os.path.join(DATASET_PATH, "meta")

# ✅ Load train.txt and test.txt files
train_txt_path = os.path.join(META_PATH, "train.txt")
test_txt_path = os.path.join(META_PATH, "test.txt")

if not os.path.exists(train_txt_path) or not os.path.exists(test_txt_path):
    raise FileNotFoundError("❌ Error: train.txt or test.txt not found. Check dataset extraction.")

with open(train_txt_path, "r") as f:
    train_files = f.read().splitlines()

with open(test_txt_path, "r") as f:
    test_files = f.read().splitlines()

# ✅ Convert file names to full paths
train_files = [os.path.join(IMAGES_PATH, f"{file}.jpg") for file in train_files]
test_files = [os.path.join(IMAGES_PATH, f"{file}.jpg") for file in test_files]

# ✅ Create labels from folder names (food categories)
categories = sorted(os.listdir(IMAGES_PATH))
label_map = {category: idx for idx, category in enumerate(categories)}

# ✅ Extract labels from file paths
# Extract labels from file paths correctly
train_labels = [label_map[os.path.basename(os.path.dirname(path))] for path in train_files]
test_labels = [label_map[os.path.basename(os.path.dirname(path))] for path in test_files]


# ✅ Split a validation set from training data (10%)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.1, random_state=42
)

# ✅ Define ImageDataGenerator for data augmentation & preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Create data generators
# ✅ Convert labels to string format
train_df = pd.DataFrame({"filename": train_files, "class": [str(label) for label in train_labels]})
val_df = pd.DataFrame({"filename": val_files, "class": [str(label) for label in val_labels]})
test_df = pd.DataFrame({"filename": test_files, "class": [str(label) for label in test_labels]})


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("✅ Data preprocessing completed successfully!")
