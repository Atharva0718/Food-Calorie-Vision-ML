import os
import pickle

# ✅ Define the correct dataset path
IMAGE_PATH = r"C:\Users\laptop\Downloads\food-101\food-101\images"

# ✅ Get the category names
categories = sorted(os.listdir(IMAGE_PATH))

# ✅ Save categories as a pickle file
with open("categories.pkl", "wb") as f:
    pickle.dump(categories, f)

print("✅ categories.pkl file has been created successfully!")
