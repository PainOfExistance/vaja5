from email import utils
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import tensorflow as tf

def lbp(image):

    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    for ih in range(1, height - 1):
        for iw in range(1, width - 1):
            center = image[ih, iw]
            neighbors = [
                image[ih - 1, iw - 1],
                image[ih - 1, iw],
                image[ih - 1, iw + 1],
                image[ih, iw - 1],
                image[ih, iw + 1],
                image[ih + 1, iw - 1],
                image[ih + 1, iw],
                image[ih + 1, iw + 1],
            ]

            binary_pattern = (np.array(neighbors) >= center).astype(int)
            powers_of_two = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
            lbp_value = np.sum(binary_pattern * powers_of_two)

            lbp_image[ih, iw] = lbp_value

    return lbp_image


def hog(gray_image, cell_size, block_size, num_bins):
    height, width = gray_image.shape
    gx = np.zeros_like(gray_image, dtype=np.float32)
    gy = np.zeros_like(gray_image, dtype=np.float32)

    gx[:, :-1] = np.diff(gray_image, n=1, axis=1)
    gy[:-1, :] = np.diff(gray_image, n=1, axis=0)

    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_orientation = (np.arctan2(gy, gx) * (180.0 / np.pi) + 180.0) % 180.0

    num_cells_x = width // cell_size
    num_cells_y = height // cell_size

    histogram = np.zeros((num_cells_y, num_cells_x, num_bins))

    for y in range(num_cells_y):
        for x in range(num_cells_x):
            cell_magnitude = gradient_magnitude[y * cell_size:(y + 1) * cell_size,
                             x * cell_size:(x + 1) * cell_size]
            cell_orientation = gradient_orientation[y * cell_size:(y + 1) * cell_size,
                               x * cell_size:(x + 1) * cell_size]
            
            hist = np.histogram(cell_orientation, bins=num_bins, range=(0, 180),
                                weights=cell_magnitude)[0]

            histogram[y, x, :] = hist / np.sqrt(np.sum(hist ** 2) + 1e-6)

    hog_descriptor = np.zeros((num_cells_y - block_size + 1, num_cells_x - block_size + 1,
                               block_size * block_size * num_bins))

    for y in range(num_cells_y - block_size + 1):
        for x in range(num_cells_x - block_size + 1):
            block_histogram = histogram[y:y + block_size, x:x + block_size, :].flatten()
            hog_descriptor[y, x, :] = block_histogram / np.sqrt(np.sum(block_histogram ** 2) + 1e-6)

    return hog_descriptor.flatten()



img = []
fname = []
for file in glob.glob(
    "G:/Faks/Sola/2.letnik/2.semester/rv/vaja4/images/*.jpg"
):
    tmp=os.path.splitext(os.path.basename(file))[0]
    fname.append(tmp[:3])
    img.append(cv2.imread(file))
    

vel_celice = int(input("vnestie vel_celice:"))
vel_blok = int(input("vnestie vel_blok:"))
razdelki = int(input("vnestie število razdelkov:"))
lowPercent = int(input("vnesite % učenja:"))

images = []
animals = []
for z, n in zip(img,fname):
    if z is not None:
        z = cv2.resize(z, (100, 100))
        images.append(z)
        if n=="cat":
            animals.append(0)
        else:
            animals.append(1)
    else:
        print(z)

labels = np.array(animals)

vect = []
i=0
for img in images:
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_image = lbp(gray_image)
    hog_descriptor = hog(gray_image, vel_celice, vel_blok, razdelki)
    feature_vector = np.concatenate((lbp_image.flatten(), hog_descriptor))
    vect.append(feature_vector)
    print(i)
    i=i+1

# Convert data to numpy arrays
vect = np.array(vect)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(vect, labels, test_size=((100-lowPercent)/100), random_state=80, stratify=labels)

# One-hot encode labels
num_classes = len(np.unique(labels))
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_encoded =  tf.keras.utils.to_categorical(y_test, num_classes)

# Define neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Update the number of output units
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print('Test accuracy:', test_acc)

# Use model for prediction
predictions = model.predict(vect)


"""
catsHOG = []
catsLBP = []
i=0
for x in images:
    gray_image = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    catsHOG.append(hog(gray_image, vel_celice, vel_blok, razdelki))
    catsLBP.append(lbp(gray_image))
    print(i)
    i=i+1
labels = np.array(animals)


# Assuming catsHOG and catsLBP are lists of HOG and LBP features for each image
combined_learn = []
for hog_feat, lbp_feat in zip(catsHOG, catsLBP):
    # Flatten the HOG and LBP features
    hog_feat_flat = hog_feat.flatten()
    lbp_feat_flat = lbp_feat.flatten()

    # Concatenate the flattened features
    combined_feat = np.hstack((hog_feat_flat, lbp_feat_flat))
    combined_learn.append(combined_feat)


# Pad the shorter array with zeros (if necessary)
max_length = max(len(combined_learn), len(labels))
if len(combined_learn) < max_length:
    combined_learn += [np.zeros_like(combined_learn[0])] * (
        max_length - len(combined_learn)
    )
elif len(labels) < max_length:
    labels = np.hstack((labels, [None] * (max_length - len(labels))))

# Replace missing values (NaN) with the column mean
combined_learn_df = pd.DataFrame(combined_learn)
combined_learn_df.fillna(combined_learn_df.mean(), inplace=True)
combined_learn_clean = combined_learn_df.values.tolist()

# Split the data and train the classifier
X_train, X_test, y_train, y_test = train_test_split(
    combined_learn_clean, labels, test_size=((100-lowPercent) / 100), random_state=90, stratify=labels
)
# knn=SVC(kernel='poly', C=5, probability=True)
# knn = KNeighborsClassifier(n_neighbors=8)
knn = RandomForestClassifier(n_estimators=1000, random_state=80)

knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
dump(knn, "model7.joblib")
"""