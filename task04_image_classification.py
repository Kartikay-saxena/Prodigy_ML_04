import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

DATASET_DIR = r"F:\ML-PROJECT\projecttask4\ds_hand_gest_task4\leapGestRecog"  # top folder
IMG_SIZE = 32
MAX_IMAGES = 50  # per gesture subfolder

data = []
labels = []

# Loop over top-level folders (00, 01, ...)
for top_idx, top_folder in enumerate(os.listdir(DATASET_DIR)):
    top_path = os.path.join(DATASET_DIR, top_folder)
    if not os.path.isdir(top_path):
        continue

    # Loop over subfolders (01_palm, 02_l, ...)
    for sub_idx, sub_folder in enumerate(os.listdir(top_path)):
        sub_path = os.path.join(top_path, sub_folder)
        if not os.path.isdir(sub_path):
            continue

        count = 0
        for file in os.listdir(sub_path):
            if count >= MAX_IMAGES:
                break
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(sub_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img.flatten())
            labels.append(top_idx)  # label by top-level folder
            count += 1

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Check data
print("Total images loaded:", X.shape[0])
print("Total labels loaded:", y.shape[0])

if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
    raise ValueError("Data or labels are empty / mismatched!")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear', random_state=42)
print("Training SVM model...")
svm_model.fit(X_train, y_train)

# Evaluate
y_pred = svm_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
