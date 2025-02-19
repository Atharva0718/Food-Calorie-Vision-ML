import os
import cv2
import numpy as np
import tensorflow as tf
import pickle

# ✅ Define the correct path to images
IMAGE_PATH = r"C:\Users\laptop\Downloads\food-101\food-101\images"

# ✅ Load the trained model
model = tf.keras.models.load_model("food_classifier.h5")

# ✅ Load category names
with open("categories.pkl", "rb") as f:
    categories = pickle.load(f)

# ✅ Define calorie values for food items
calorie_dict = {
    "apple_pie": 237,
    "baby_back_ribs": 290,
    "baklava": 330,
    "banana_bread": 196,
    "beignets": 250,
    "cannoli": 372,
    "cheesecake": 321,
    "ceviche": 144,
    "chicken_curry": 200,
    "chicken_wings": 290,
    "chocolate_cake": 352,
    "chocolate_mousse": 225,
    "churros": 280,
    "clam_chowder": 150,
    "club_sandwich": 350,
    "cup_cakes": 250,
    "donuts": 195,
    "dumplings": 200,
    "eggs_benedict": 300,
    "falafel": 333,
    "filet_mignon": 267,
    "fish_and_chips": 595,
    "foie_gras": 462,
    "french_fries": 312,
    "french_onion_soup": 93,
    "french_toast": 229,
    "fried_calamari": 200,
    "fried_rice": 235,
    "frozen_yogurt": 159,
    "garlic_bread": 350,
    "gnocchi": 250,
    "greek_salad": 150,
    "grilled_cheese_sandwich": 400,
    "grilled_salmon": 367,
    "guacamole": 160,
    "hamburger": 250,
    "hot_and_sour_soup": 100,
    "hot_dog": 290,
    "ice_cream": 207,
    "lasagna": 135,
    "lobster_bisque": 160,
    "lobster_roll_sandwich": 320,
    "macaroni_and_cheese": 310,
    "macarons": 360,
    "miso_soup": 40,
    "mussels": 172,
    "nachos": 343,
    "omelette": 154,
    "onion_rings": 411,
    "oysters": 41,
    "pad_thai": 357,
    "paella": 300,
    "pancakes": 227,
    "panna_cotta": 300,
    "peking_duck": 337,
    "pho": 350,
    "pizza": 285,
    "pork_chop": 242,
    "poutine": 550,
    "prime_rib": 367,
    "ramen": 436,
    "ravioli": 250,
    "red_velvet_cake": 367,
    "risotto": 300,
    "samosa": 262,
    "sashimi": 40,
    "scallops": 112,
    "seaweed_salad": 45,
    "shrimp_and_grits": 250,
    "spaghetti_bolognese": 300,
    "spaghetti_carbonara": 400,
    "spring_rolls": 100,
    "steak": 271,
    "strawberry_shortcake": 300,
    "sushi": 200,
    "tacos": 226,
    "takoyaki": 150,
    "tiramisu": 240,
    "tuna_tartare": 200,
    "waffles": 291
}

# ✅ Function to predict food and estimate calories
def predict_food_with_calories(image_path):
    # ✅ Read and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dims for model input

    # ✅ Make prediction
    preds = model.predict(img)
    predicted_class = np.argmax(preds)
    food_name = categories[predicted_class]

    # ✅ Fetch calorie information
    estimated_calories = calorie_dict.get(food_name, "Calories not available")

    return food_name, estimated_calories

# ✅ Example test
image_path = "test_food.jpg"  # Change this to your test image
predicted_food, estimated_calories = predict_food_with_calories(image_path)
print(f"🍔 Predicted Food: {predicted_food}")
print(f"🔥 Estimated Calories: {estimated_calories} kcal per 100g")