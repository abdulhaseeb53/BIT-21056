import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_data(dataset):
    images = []
    labels = []
    
    for label in ['real', 'fake']:
        project = os.path.join(dataset, label)
        for filename in os.listdir(project):
            img_path = os.path.join(project, filename)
            img = cv2.imread('dataset/real/real1.jpeg')
            img = cv2.resize(img, (100, 100))  # Resize for consistency
            images.append(img)
            labels.append(0 if label == 'real' else 1)  # 0 for real, 1 for fake
            
    return np.array(images), np.array(labels)

data_dir = 'dataset'  # Update this path
X, y = load_data('real')
X, y = load_data('fake')

X_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X])
X_flat = X_gray.reshape(X_gray.shape[0], -1)  # Flatten images

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_currency(img_path):
    img = cv2.imread('dataset/real/real1.jpeg')
    img = cv2.resize(img, (100, 100))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1, -1)
    prediction = model.predict(img_gray)
    return "Real" if prediction[0] == 0 else "Fake"

# Example usage
result = predict_currency('dataset/real/real1.jpeg')
print(result)