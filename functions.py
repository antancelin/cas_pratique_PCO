import cv2
import numpy as np
import json
import joblib
import tensorflow as tf

# Chargement des encodeurs entraînés pour transformer les images en vecteurs numériques
encodeur_classifier = joblib.load('joblib/encodeur/encodeur_classifier.joblib')
encodeur_left = joblib.load('joblib/encodeur/encodeur_left.joblib')
encodeur_right = joblib.load('joblib/encodeur/encodeur_right.joblib')


# Chargement des modèles entraînés
model_classifier = tf.keras.models.load_model('models/classifier')
model_classifier.trainable = False
model_left = tf.keras.models.load_model('models/left_eye')
model_left.trainable = False
model_right = tf.keras.models.load_model('models/right_eye')
model_right.trainable = False


# Prétraitement de l'image : redimensionnemennt et standardization
def preprocess_img(img,new_dim=(240,320)):
    new_img=cv2.resize(img, (new_dim[1],new_dim[0]), interpolation = cv2.INTER_AREA)
    # Normalisation en divisant par 255
    last_img = new_img / 255.0
    return last_img


# Recherche des informations de l'employé correspondant à l'ID prédit via les informations stockées dans le JSON
def employee_ID(user_prediction):
    with open('employees_info.json', 'r') as json_file:
        data = json.load(json_file)
    info = data[str(user_prediction)]
    nom = info['nom']
    annee_embauche = info['annee_embauche']
    genre = info['genre']
    poste = info['poste']
    return nom,annee_embauche,genre,poste


# Réalisation de la prédiction pour savoir s'il s'agit de l'oeil gauche ou droit
def detect_classifier(image):
    probs=model_classifier.predict(np.array([image]))
    prediction = np.argmax(probs)             
    decode_prediction = encodeur_classifier.inverse_transform([prediction])
    return decode_prediction
