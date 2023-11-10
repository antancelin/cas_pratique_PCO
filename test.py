import streamlit as st
from PIL import Image
from functions import preprocess_img, employee_ID, detect_classifier
import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

class ImagePredictionApp:

    def __init__(self):
        # Chargement des modèles
        self.load_model_classifier()
        self.load_model_left()
        self.load_model_right()

        # Chargement des encodeurs
        self.encodeur_classifier = joblib.load('joblib/encodeur/encodeur_classifier.joblib')
        self.encodeur_left = joblib.load('joblib/encodeur/encodeur_left.joblib')
        self.encodeur_right = joblib.load('joblib/encodeur/encodeur_right.joblib')

    def load_model_classifier(self):
        # Chargement du modèle classifier en utilisant la bibliothèque ML
        self.model_classifier = tf.keras.models.load_model("models/classifier")
        self.model_classifier.trainable = False

    def load_model_left(self):
        # Chargement du modèle oeil gauche en utilisant la bibliothèque ML
        self.model_left = tf.keras.models.load_model("models/left_eye")
        self.model_left.trainable = False

    def load_model_right(self):
        # Chargement du modèle oeil droit en utilisant la bibliothèque ML
        self.model_right = tf.keras.models.load_model("models/right_eye")
        self.model_right.trainable = False

    def display_user(self, user_prediction):
        nom, annee_embauche, genre, poste = employee_ID(user_prediction)
        st.write(f"Nom: {nom}")
        st.write(f"Embauché(e) en {annee_embauche}")
        st.write(f"Genre: {genre}")
        st.write(f"Poste: {poste}")

    def predict_image(self, image):
        # Faites la prédiction pour savoir s'il s'agit de l'oeil gauche ou droit
        eye_type = detect_classifier(image)

        # Utilisez le modèle approprié pour faire la prédiction
        if eye_type == 0:
            prediction = self.model_right.predict(np.array([image]))
            fiability = tf.keras.models.Model.predict(self.model_right, np.array([image]))[0][np.argmax(prediction)]
            fiability *= 100
            fiability = format(fiability, ".2f")
            user_prediction = np.argmax(prediction)
            decode_user_prediction = self.encodeur_right.inverse_transform([user_prediction])  # Inversion encodage
            self.display_user(decode_user_prediction[0])
        else:
            prediction = self.model_left.predict(np.array([image]))
            fiability = tf.keras.models.Model.predict(self.model_left, np.array([image]))[0][np.argmax(prediction)]
            fiability *= 100
            fiability = format(fiability, ".2f")
            user_prediction = np.argmax(prediction)
            decode_user_prediction = self.encodeur_left.inverse_transform([user_prediction])  # Inversion encodage
            self.display_user(decode_user_prediction[0])

        st.write(f"{fiability}% de fiabilité qu'il s'agisse de :")
        st.write(f"Oeil utilisé: {eye_type}")

if __name__ == "__main__":
    app = ImagePredictionApp()

    st.title("Application d'Authentification d'Iris")

    uploaded_file = st.file_uploader("Sélectionner une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image sélectionnée", use_column_width=True)

        # Convertir l'image en tableau numpy pour le traitement
        image_np = np.array(image)

        # Prétraiter l'image
        image_prep = preprocess_img(image_np)

        # Bouton de prédiction
        if st.button("Lancer la Prédiction"):
            app.predict_image(image_prep)

    st.button("Effacer")