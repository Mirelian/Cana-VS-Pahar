<a name="br1"></a> 

Clasiﬁcare Imagini (Cana vs. Pahar)

Proiectul consta intr-un sistem de clasiﬁcare a imaginilor intre doua clase: 'Cana' și 'Pahar'.

Scopul este antrenarea unui model de invatare automata care sa poata disꢀnge intre aceste doua

obiecte comune. Logica principala prin care am incercat sa antrenez algoritmul este de a detecta

manerul de la o cana, absenta sa indicand clasiﬁcarea drept pahar.

Setul de date a fost colectat manual din surse publice si organizat in 2 subfoldere

corespunzatoare celor doua clase in folderul ”dataset”. M-am asigurat in selectare ca nu exista mai mult

de o cana / pahar per imagine pentru simpliﬁcare procesului. Am folosit un script .bat pentru a le

redenumi de la 1 la n, dupa am folosit un Random-Number-Generator pentru a selecta datele de testare,

organizate in acelasi fel in folderul ”test”.

Primul pas in preprocesarea este redimensionarea imaginilor asꢁel incat sa ﬁe consistente una

cu cealalta si pentru scaderea resurselor necesare procesarii. Am ales 200x200 prin testare unde am

vazut ca ca dimensiuni mai mari nu cresc semniﬁcaꢀv calitatea rezultatelor. Dupa am converꢀt valorile la

grayscale deoarece nu am considerat obligatorie culoare in detecꢀe, algoritmul ﬁind antrenat sa se uite

la characterisꢀcile ﬁzice ale obiectelor, cu alte cuvinte manerul. In ﬁnal am converꢀt valorile in matrice

2D numpy si le-am normalizat pentru procesare.

Am uꢀlizat algoritmul K-Nearest Neighbors (KNN) pentru clasiﬁcare datorita usurintei de

implementare si am ales arbitrar n\_neighbors=3.

**Biblioteci Python Uꢀlizate**

·

·

·

·

·

os pentru manipularea sistemului de ﬁșiere

cv2 (OpenCV) pentru manipularea imaginilor

numpy pentru manipularea matricelor

scikit-learn pentru implementarea algoritmului KNN și evaluarea modelului

matplotlib pentru vizualizarea imaginilor și rezultatelor

import os

import cv2

import numpy as np

from sklearn.neighbors import KNeighborsClassiﬁer

from sklearn.preprocessing import StandardScaler



<a name="br2"></a> 

from sklearn.metrics import accuracy\_score

from sklearn.metrics import confusion\_matrix

from sklearn.metrics import classiﬁcaꢀon\_report

import matplotlib.pyplot as plt

\# Funcꢀon to load train images from a folder and assign labels

def load\_train\_images\_from\_folder(folder, target\_shape=None):

images = []

labels = []

for subfolder in os.listdir(folder):

subfolder\_path = os.path.join(folder, subfolder)

if os.path.isdir(subfolder\_path):

for ﬁlename in os.listdir(subfolder\_path):

if ﬁlename.endswith('.jpg'):

img = cv2.imread(os.path.join(subfolder\_path, ﬁlename))

if img is not None:

\# Resize the image to a consistent shape (e.g., 100x100)

if target\_shape is not None:

img = cv2.resize(img, target\_shape)

images.append(img)

labels.append(subfolder)

else:

print(f"Warning: Unable to load {ﬁlename}")

return images, labels

\# Funcꢀon to load test images from a folder and assign labels

def load\_test\_images\_from\_folder(folder, target\_shape=None):

images = []

labels = []



<a name="br3"></a> 

for subfolder in os.listdir(folder):

subfolder\_path = os.path.join(folder, subfolder)

if os.path.isdir(subfolder\_path):

for ﬁlename in os.listdir(subfolder\_path):

if ﬁlename.endswith('.jpg'):

img = cv2.imread(os.path.join(subfolder\_path, ﬁlename))

if img is not None:

\# Resize the image to a consistent shape (e.g., 100x100)

if target\_shape is not None:

img = cv2.resize(img, target\_shape)

images.append(img)

labels.append(subfolder)

print('Labels \n', labels)

else:

print(f"Warning: Unable to load {ﬁlename}")

return images, labels

def train\_knn\_model(images, labels, num\_neighbors):

knn = KNeighborsClassiﬁer(n\_neighbors=num\_neighbors)

knn.ﬁt(images.reshape(images.shape[0], -1), labels)

return knn

def main():

\# Folder paths

dataset\_folder = 'dataset'

test\_folder = 'test'

\# Load images and labels from the 'dataset' folder and resize them to (200, 200)

images, labels = load\_train\_images\_from\_folder(dataset\_folder, target\_shape=(200, 200))



<a name="br4"></a> 

\# Reshape the images and convert them to grayscale

image\_data = [cv2.cvtColor(image, cv2.COLOR\_BGR2GRAY).ﬂaꢂen() for image in images]

\# Convert the list of 1D arrays to a 2D numpy array

image\_data = np.array(image\_data)

\# Scale the data

scaler = StandardScaler()

scaled\_data = scaler.ﬁt\_transform(image\_data)

\# Antrenarea modelului KNN

num\_neighbors = 3 # Poți ajusta acest număr la nevoie

knn\_model = train\_knn\_model(scaled\_data, labels, num\_neighbors)

\# Evaluarea modelului pe setul de date de antrenare

train\_predicꢀons = knn\_model.predict(scaled\_data.reshape(scaled\_data.shape[0], -1))

\# Calcularea acuratetei pe setul de date de antrenare

accuracy\_train = accuracy\_score(labels, train\_predicꢀons)

print(f"\nAcuratete pe setul de date de antrenare: {accuracy\_train:.2f}")

\# Matricea de confuzie pe setul de date de antrenare

conf\_matrix = confusion\_matrix(labels, train\_predicꢀons)

print("\nMatricea de confuzie:")

print(conf\_matrix)

\# Raportul de clasiﬁcare pe setul de date de antrenare

class\_report = classiﬁcaꢀon\_report(labels, train\_predicꢀons, target\_names=['Cana', 'Pahar'])



<a name="br5"></a> 

print("\nRaportul de clasiﬁcare:")

print(class\_report)

\# Testarea modelului pe setul de date de test

test\_images, test\_labels = load\_test\_images\_from\_folder(test\_folder, target\_shape=(200, 200))

print('# TEST ﬁles:', len(test\_images))

test\_predicꢀon = []

plt.ﬁgure(ﬁgsize=(12, 6))

\# Apply KNN to TEST images

for i, test\_image in enumerate(test\_images):

test\_image\_data = cv2.cvtColor(test\_image, cv2.COLOR\_BGR2GRAY).ﬂaꢂen()

test\_image\_data = np.array(test\_image\_data).reshape(1, -1) # Reshape for a single sample

scaled\_test\_data = scaler.transform(test\_image\_data)

test\_predicꢀon.append(knn\_model.predict(scaled\_test\_data)) # Predict classes

if test\_predicꢀon[i] == 'cana':

print(f"Test Image {i + 1} - ==Cana==")

else:

print(f"Test Image {i + 1} - ==Pahar==")

\# Vizualizare imagine cu predicꢀa

plt.subplot(2, len(test\_images) // 2, i + 1)

plt.imshow(cv2.cvtColor(test\_image, cv2.COLOR\_BGR2RGB))

if test\_predicꢀon[i] == 'cana':

plt.ꢀtle(f"Cana")

else:

plt.ꢀtle(f"Pahar")

plt.axis('oﬀ')



<a name="br6"></a> 

plt.show()

\# Calcularea acurateței pe setul de date de testare

accuracy\_test = accuracy\_score(test\_labels, test\_predicꢀon)

print(f"\nAcuratete pe setul de date de testare: {accuracy\_test:.2f}")

\# Matricea de confuzie pe setul de date de testare

conf\_matrix = confusion\_matrix(test\_labels, test\_predicꢀon)

print("\nMatricea de confuzie:")

print(conf\_matrix)

\# Raportul de clasiﬁcare pe setul de date de testare

class\_report = classiﬁcaꢀon\_report(test\_labels, test\_predicꢀon, target\_names=['Cana', 'Pahar'])

print("\nRaportul de clasiﬁcare:")

print(class\_report)

if \_\_name\_\_ == "\_\_main\_\_":

main()



<a name="br7"></a> 

Prima data am testat algoritmul pe setul de date de antrenare pentru a vedea daca exista deviaꢀi

in adaptare. Din ce am ciꢀt legat de KNN algoritmul poate avea tendința de a memora setul de date de

antrenare, dar rezultatele imperfecte de 90% arata ca algoritmul este cel puꢀn capabil de a generaliza

imaginile.

Putem vedea de asemenea o inclinare catre TN (pahare detectate drept cani) dubla fata de FP.

(cani detectate drept pahare).

Pe setul de date de testare au aparut doua clasiﬁcari gresite de ꢀp TN, conform cu ceea ce am

vazut la testarea precedenta. Scorurile f1 ale celor doua clase do obiecte sunt destul de apropiate unul

de celalalt asꢁel putem spune ca sunt in marginea de erroare idenꢀce, algoritmul neavand preferinte

catre precizie sau recall. In rest, acuratetea de 90% este opꢀma, dar nu ideala.

Daca ne uitam la imaginile alese, vedem ca paharele cu label-ul ’Cana’ au in imagine obiecte in

plus care pot ﬁ interpretate de catre algoritm drept maner, deci o categorisire de ’Cana’. De asemenea

am observat ca exista o diversitate mare de ꢀpuri de pahare in setul de date ce ar ﬁ putut incurce

algoritmul in deﬁnirea proprie a unui ’Pahar’.



<a name="br8"></a> 

