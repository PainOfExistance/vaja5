import cv2
import numpy as np
import glob
from numba import njit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def razdeli_podatke(images, lowPercent):
    learn = []
    test = []
    dist =  int(len(images) / 2)
    catPercent = int(dist * (lowPercent/100))
    dogPercent = int(dist * (lowPercent/100))
    
    for element in images[0:catPercent:1]:
        learn.append(element)

    for element in images[catPercent:dist:1]:
        test.append(element)


    for element in images[dist:dist+dogPercent:1]:
        learn.append(element)

    for element in images[dist+dogPercent:len(images):1]:
        test.append(element)

    return learn, test

def lbp(image):
    height, width = image.shape
    lbp_image = np.zeros((height - 2, width - 2), dtype=np.uint8)
    
    center = image[1:-1, 1:-1]
    
    # Create a 3D array to store the 8 neighbors for each pixel
    neighbors = np.stack([image[:-2, :-2], image[:-2, 1:-1], image[:-2, 2:],
                          image[1:-1, :-2],                 image[1:-1, 2:],
                          image[2:, :-2],   image[2:, 1:-1], image[2:, 2:]], axis=-1)
    
    # Calculate binary pattern
    binary_pattern = (neighbors >= center[..., np.newaxis]).astype(int)
    
    # Calculate LBP values using vectorized operations
    powers_of_two = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    lbp_image = np.sum(binary_pattern * powers_of_two, axis=-1)
    
    return lbp_image

def hog(gray_image, vel_celice, vel_blok, razdelki):
    gx = cv2.Sobel(gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    gy = cv2.Sobel(gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    M = np.sqrt(gx**2 + gy**2)
    S = np.arctan2(gy, gx) * (180 / np.pi) % 180

    bins = []
    y_range = range(0, S.shape[0] - vel_celice * vel_blok + 1, vel_celice)
    x_range = range(0, S.shape[1] - vel_celice * vel_blok + 1, vel_celice)
    
    for y in y_range:
        for x in x_range:
            block_bins = np.zeros(razdelki)
            subarea_M = M[y:y + vel_celice * vel_blok, x:x + vel_celice * vel_blok]
            subarea_S = S[y:y + vel_celice * vel_blok, x:x + vel_celice * vel_blok]
            z_range = range(0, vel_celice * vel_blok, vel_celice)
            
            for z in z_range:
                for u in z_range:
                    cell_M = subarea_M[z:z + vel_celice, u:u + vel_celice]
                    cell_S = subarea_S[z:z + vel_celice, u:u + vel_celice]
                    hist, _ = np.histogram(cell_S, bins=razdelki, range=(0, 180), weights=cell_M)
                    block_bins += hist
            
            # Normalize the block histogram using L2-norm
            l2_norm = np.sqrt(np.sum(block_bins ** 2))
            normalized_block_bins = block_bins / l2_norm
            bins.append(normalized_block_bins)

    # Flatten the list of block histograms
    flattened_histograms = np.hstack(bins)
    return flattened_histograms

img = [cv2.imread(file) for file in glob.glob("G:/Faks/Sola/2.letnik/2.semester/rv/vaja4/images/*.jpg")]
vel_celice = int(input("vnestie vel_celice:"))
vel_blok = int(input("vnestie vel_blok:"))
razdelki = int(input("vnestie število razdelkov:"))
lowPercent = int(input("vnesite % učenja:"))

images=[]
for z in img:
    if(z is not None):
        z=cv2.resize(z,(100,100))
        images.append(z)
    else:
        print(z)

        """
learn, test=razdeli_podatke(images, lowPercent)
catsHOG=[]
catsLBP=[]
dogsHOG=[]
dogsLBP=[]

catsHOGtest=[]
catsLBPtest=[]
dogsHOGtest=[]
dogsLBPtest=[]

for x in learn[0:int(len(learn)/2):1]:
    gray_image= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    catsHOG.append(hog(gray_image, vel_celice, vel_blok, razdelki))
    catsLBP.append(lbp(gray_image))

for x in learn[int(len(test)/2)+1:int(len(learn)):1]:
    gray_image= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    dogsHOG.append(hog(gray_image, vel_celice, vel_blok, razdelki))
    dogsLBP.append(lbp(gray_image))


for x in test[0:int(len(test)/2):1]:
    gray_image= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    catsHOGtest.append(hog(gray_image, vel_celice, vel_blok, razdelki))
    catsLBPtest.append(lbp(gray_image))

for x in test[int(len(test)/2)+1:int(len(test)):1]:
    gray_image= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    dogsHOGtest.append(hog(gray_image, vel_celice, vel_blok, razdelki))
    dogsLBPtest.append(lbp(gray_image))


for x in dogsHOG:
    catsHOG.append(x)

for x in dogsLBP:
    catsLBP.append(x)

for x in dogsHOGtest:
    catsHOG.append(x)

for x in dogsLBPtest:
    catsLBP.append(x)
    """

catsHOG=[]
catsLBP=[]
for x in images:
    gray_image= cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    catsHOG.append(hog(gray_image, vel_celice, vel_blok, razdelki))
    catsLBP.append(lbp(gray_image))


# Assuming catsHOG and catsLBP are lists of HOG and LBP features for each image
combined_learn = []
for hog_feat, lbp_feat in zip(catsHOG, catsLBP):
    # Flatten the HOG and LBP features
    hog_feat_flat = hog_feat.flatten()
    lbp_feat_flat = lbp_feat.flatten()

    # Concatenate the flattened features
    combined_feat = np.hstack((hog_feat_flat, lbp_feat_flat))
    combined_learn.append(combined_feat)

# Create labels
animals=[]
for i in range(int(len(images))):
    if(i<500):
        animals.append("cat")
    else:
        animals.append("dog")
labels = np.array(animals)

# Pad the shorter array with zeros (if necessary)
max_length = max(len(combined_learn), len(labels))
if len(combined_learn) < max_length:
    combined_learn += [np.zeros_like(combined_learn[0])] * (max_length - len(combined_learn))
elif len(labels) < max_length:
    labels = np.hstack((labels, [None] * (max_length - len(labels))))

# Replace missing values (NaN) with the column mean
combined_learn_df = pd.DataFrame(combined_learn)
combined_learn_df.fillna(combined_learn_df.mean(), inplace=True)
combined_learn_clean = combined_learn_df.values.tolist()

# Split the data and train the classifier
X_train, X_test, y_train, y_test = train_test_split(combined_learn_clean, labels, test_size=(lowPercent/100), random_state=2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

cv2.imshow("Edge Detection", images[0]);
cv2.waitKey(0);
cv2.destroyAllWindows();



"""
# Assuming catsHOG and catsLBP are lists of HOG and LBP features for each image
combined_learn = []
for hog_feat, lbp_feat in zip(catsHOG, catsLBP):
    # Flatten the HOG and LBP features
    hog_feat_flat = hog_feat.flatten()
    lbp_feat_flat = lbp_feat.flatten()

    # Concatenate the flattened features
    combined_feat = np.hstack((hog_feat_flat, lbp_feat_flat))
    combined_learn.append(combined_feat)

# Create labels
animals = ["cat"] * (len(images) // 2) + ["dog"] * (len(images) // 2)
labels = np.array(animals)

# Pad the shorter array with zeros (if necessary)
max_length = max(len(combined_learn), len(labels))
if len(combined_learn) < max_length:
    combined_learn += [np.zeros_like(combined_learn[0])] * (max_length - len(combined_learn))
elif len(labels) < max_length:
    labels = np.hstack((labels, [None] * (max_length - len(labels))))

# Replace missing values (NaN) with the column mean
combined_learn_df = pd.DataFrame(combined_learn)
combined_learn_df.fillna(combined_learn_df.mean(), inplace=True)
combined_learn_clean = combined_learn_df.values.tolist()

# Split the data and train the classifier
X_train, X_test, y_train, y_test = train_test_split(combined_learn_clean, labels, test_size=(lowPercent/100), random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
"""