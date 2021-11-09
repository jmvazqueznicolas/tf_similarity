import tensorflow as tf
import numpy as np
import tensorflow_similarity as tfsim
import cv2
import threading

#from tabulate import tabulate
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from tensorflow_similarity.utils import tf_cap_memory
from tensorflow_similarity.layers import MetricEmbedding # row wise L2 norm
from tensorflow_similarity.losses import MultiSimilarityLoss  # specialized similarity loss
from tensorflow_similarity.models import SimilarityModel # TF model with additional features
from tensorflow_similarity.samplers import MultiShotMemorySampler  # sample data 
from tensorflow_similarity.samplers import select_examples  # select n example per class
from tensorflow_similarity.visualization import viz_neigbors_imgs  # neigboors vizualisation
from tensorflow_similarity.visualization import confusion_matrix  # matching performance

detener_hilos = False
flag1 = False
frame1 = np.array([])
width = 1280 
height = 720

# Hilo de detección en fotografia
def cam_1(w, h):
    global flag1
    global frame1
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # Toma de capturas para el autoenfoque
    for value in range(50):
        ret, frame1_lec = cap.read()
    # Inicia la busqueda continua de los rostros en la credencial
    while True:
        ret, frame1_lec = cap.read()
        dim = (480, 320)
        dim = (480, 320)
        frame1 = cv2.resize(frame1_lec, dim, interpolation = cv2.INTER_AREA)
        frame1 = frame1[:, :, ::-1]
        flag1 = True
        if detener_hilos == True:
            cap.release()
            break

import os
folder = './imagenes_test'
clases = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]


""" Se crea el dataset """
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'imagenes_train',
  image_size=(240, 240),
  class_names=clases,
  batch_size = 1,
  )

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'imagenes_test',
  image_size=(240, 240),
  batch_size = 1,
  )


# Vectores para guardar el dataset
x_train = []
y_train = []

x_test = []
y_test = []

dataset_train = train_ds.enumerate()
dataset_test = test_ds.enumerate()

for element in dataset_train.as_numpy_iterator():
  x_train.append(element[1][0])
  y_train.append(element[1][1])


for element in dataset_test.as_numpy_iterator():
  x_test.append(element[1][0])
  y_test.append(element[1][1])


# Se dejan como enteros 
x_train = np.asarray(x_train).astype(int)
y_train = np.asarray(y_train).astype(int)
x_test = np.asarray(x_test).astype(int)
y_test = np.asarray(y_test).astype(int)


# Reshape del dataset al formato de la red neuronal
x_train = np.reshape(x_train, (216,240,240,3))
y_train = np.reshape(y_train, (216))
x_test = np.reshape(x_test, (54,240,240,3))
y_test = np.reshape(y_test, (54))


# reload the model
reloaded_model = load_model("tf_similarity_model")
# reload the index
reloaded_model.load_index("tf_similarity_model")
#check the index is back
reloaded_model.index_summary()


# used to label in images in the viz_neighbors_imgs plots
# note we added a 11th classes for unknown
labels = ["0", "1",  "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", 
"22", "23", "24", "25", "26", "Unknown"]

labels = sorted(clases)

CLASSES = [i for i in range(27)]

# re-run to test on other examples
num_neighboors = 3

# select
x_display, y_display = select_examples(x_test, y_test, CLASSES, 1)

#cam1 = threading.Thread(target=cam_1, args=(width,height,))
#cam1.start()

#while True:
#  if (flag1==True):
frame1 = cv2.imread('leche.jpg')
im_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
resized = cv2.resize(im_rgb, (240, 240), interpolation = cv2.INTER_AREA)
resized = resized.reshape(1, 240, 240, 3)
# lookup the nearest neighbors
nns = reloaded_model.lookup(resized, k=num_neighboors)

# display
for idx in np.argsort(y_display):
    viz_neigbors_imgs(x_display[idx], y_display[idx], nns[idx], 
                      class_mapping=labels, fig_size=(16, 2), cmap='Greys')


"""
    for coincidencias in nns:
      for coincidencia in coincidencias:
        #print(labels[coincidencia.label], end='    ')
        coinci_img = np.asarray(coincidencia.data).astype(np.uint8)
        coinci_img = cv2.cvtColor(coinci_img, cv2.COLOR_RGB2BGR)
        print("La distancia es", coincidencia.distance)
        cv2.imshow('Coincidencias', coinci_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detener_hilos = True
            break

"""