import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Function to load train images from a folder and assign labels
def load_train_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(subfolder)
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels

# Function to load test images from a folder and assign labels
def load_test_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 100x100)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(subfolder)
                        print('Labels \n', labels)
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels

def train_knn_model(images, labels, num_neighbors):
    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(images.reshape(images.shape[0], -1), labels)
    return knn

def main():
    # Folder paths
    dataset_folder = 'dataset'
    test_folder = 'test'

    # Load images and labels from the 'dataset' folder and resize them to (200, 200)
    images, labels = load_train_images_from_folder(dataset_folder, target_shape=(200, 200))
    
    # Reshape the images and convert them to grayscale
    image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]
    
    # Convert the list of 1D arrays to a 2D numpy array
    image_data = np.array(image_data)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(image_data)

    # Antrenarea modelului KNN
    num_neighbors = 3  # Poți ajusta acest număr la nevoie
    knn_model = train_knn_model(scaled_data, labels, num_neighbors)
    
    # Evaluarea modelului pe setul de date de antrenare
    train_predictions = knn_model.predict(scaled_data.reshape(scaled_data.shape[0], -1))

    # Calcularea acuratetei pe setul de date de antrenare
    accuracy_train = accuracy_score(labels, train_predictions)
    print(f"\nAcuratete pe setul de date de antrenare: {accuracy_train:.2f}")

    # Matricea de confuzie pe setul de date de antrenare
    conf_matrix = confusion_matrix(labels, train_predictions)
    print("\nMatricea de confuzie:")
    print(conf_matrix)

    # Raportul de clasificare pe setul de date de antrenare
    class_report = classification_report(labels, train_predictions, target_names=['Cana', 'Pahar'])
    print("\nRaportul de clasificare:")
    print(class_report)

    # Testarea modelului pe setul de date de test
    test_images, test_labels = load_test_images_from_folder(test_folder, target_shape=(200, 200))
    print('# TEST files:', len(test_images))

    test_prediction = []
    plt.figure(figsize=(12, 6))
    
    # Apply KNN to TEST images
    for i, test_image in enumerate(test_images):
        test_image_data = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).flatten()
        test_image_data = np.array(test_image_data).reshape(1, -1)  # Reshape for a single sample
        scaled_test_data = scaler.transform(test_image_data)
        test_prediction.append(knn_model.predict(scaled_test_data))  # Predict classes
        if test_prediction[i] == 'cana':
            print(f"Test Image {i + 1} - ==Cana==")
        else:
            print(f"Test Image {i + 1} - ==Pahar==")

        # Vizualizare imagine cu predictia
        plt.subplot(2, len(test_images) // 2, i + 1)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        if test_prediction[i] == 'cana':
            plt.title(f"Cana")
        else:
            plt.title(f"Pahar")
        plt.axis('off')

    plt.show()

    # Calcularea acurateței pe setul de date de testare
    accuracy_test = accuracy_score(test_labels, test_prediction)
    print(f"\nAcuratete pe setul de date de testare: {accuracy_test:.2f}")

    # Matricea de confuzie pe setul de date de testare
    conf_matrix = confusion_matrix(test_labels, test_prediction)
    print("\nMatricea de confuzie:")
    print(conf_matrix)

    # Raportul de clasificare pe setul de date de testare
    class_report = classification_report(test_labels, test_prediction, target_names=['Cana', 'Pahar'])
    print("\nRaportul de clasificare:")
    print(class_report)

if __name__ == "__main__":
    main()
