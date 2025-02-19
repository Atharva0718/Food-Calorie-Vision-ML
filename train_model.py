import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from data_preprocessing import train_generator, val_generator  # Import data generators

# ✅ Define the correct path to images (same as in save_categories.py)
IMAGES_PATH = r"C:\Users\laptop\Downloads\food-101\food-101\images"

# ✅ Load category names (food classes) from dataset folder
# Ensure only valid class folders are included
categories = sorted([folder for folder in os.listdir(IMAGES_PATH) if os.path.isdir(os.path.join(IMAGES_PATH, folder))])

# ✅ Load EfficientNetB0 as a base model (pretrained on ImageNet)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)  # Fully connected layer
output = Dense(len(categories), activation="softmax")(x)  # ✅ Use `len(categories)`

# ✅ Define final model
model = Model(inputs=base_model.input, outputs=output)

# ✅ Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ✅ Train the model
history = model.fit(
    train_generator,  # Training data generator
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of steps per epoch
    validation_data=val_generator,  # Validation data generator
    validation_steps=val_generator.samples // val_generator.batch_size,  # Validation steps
    epochs=1  # Number of epochs (you can adjust this)
)

# ✅ Save the trained model
model.save("food_classifier.h5")

print("✅ Model training completed and saved as 'food_classifier.h5'!")