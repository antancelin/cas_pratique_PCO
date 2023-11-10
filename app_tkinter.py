import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from functions import preprocess_img, employee_ID, detect_classifier

import cv2
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image


class ImagePredictionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Application d'Authentification d'Iris")


        # Chargement des modèles
        self.load_model_classifier()
        self.load_model_left()
        self.load_model_right()


        # Chargement des encodeurs
        self.encodeur_classifier = joblib.load('joblib/encodeur/encodeur_classifier.joblib')
        self.encodeur_left = joblib.load('joblib/encodeur/encodeur_left.joblib')
        self.encodeur_right = joblib.load('joblib/encodeur/encodeur_right.joblib')
        

        self.select_button = tk.Button(root, text="Sélectionner une image", command=self.load_image)
        self.select_button.pack()
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.pack()
        self.imagedisp_label = tk.Label(root)
        self.imagedisp_label.pack()
        self.predict_button = tk.Button(root, text="Lancer la Prédiction", command=self.predict_image)
        self.predict_button.pack()

        self.prediction_label_eye = tk.Label(root, text="")
        self.prediction_label_eye.pack()
        
        self.prediction_label = tk.Label(root, text="")
        self.prediction_label.pack()

        self.prediction_label_name = tk.Label(root, text="")
        self.prediction_label_name.pack()

        self.prediction_label_annee = tk.Label(root, text="")
        self.prediction_label_annee.pack()

        self.prediction_label_genre = tk.Label(root, text="")
        self.prediction_label_genre.pack()

        self.prediction_label_poste = tk.Label(root, text="")
        self.prediction_label_poste.pack()

        self.clear_button = tk.Button(root, text="Effacer", command=self.clear_image)
        self.clear_button.pack()

        self.close_button = tk.Button(root, text="Fermer", command=root.destroy)
        self.close_button.pack()

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

        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image_prep=preprocess_img(self.image)  # Redimensionnez l'image pour l'affichage via la fonction
            self.imagepath_label.config(text=f"Image : {file_path}")
            self.photo = ImageTk.PhotoImage(Image.open(file_path))
            self.imagedisp_label.config(image=self.photo)


    def clear_image(self):
        self.image = None
        self.image_prep = None
        self.prediction_label.config(text="")
        self.prediction_label_eye.config(text="")
        # self.prediction_label_id.config(text="")
        self.prediction_label_name.config(text="")
        self.prediction_label_annee.config(text="")
        self.prediction_label_genre.config(text="")
        self.prediction_label_poste.config(text="")
        self.imagedisp_label.config(image="")
        self.imagepath_label.config(text="")

    
    def display_user(self,user_prediction):
        nom,annee_embauche,genre,poste = employee_ID(user_prediction)
        self.prediction_label_name.config(text=f"{nom}")
        self.prediction_label_annee.config(text=f"Embauché(e) en {annee_embauche}")
        self.prediction_label_genre.config(text=f"Genre : {genre}")
        self.prediction_label_poste.config(text=f"Sur un poste de {poste}")

    
    def predict_image(self):
        if hasattr(self, 'image'):
            # Faites la prédiction pour savoir s'il s'agit de l'oeil gauche ou droit
            eye_type = detect_classifier(self.image_prep)

            # Utilisez le modèle approprié pour faire la prédiction
            if eye_type == 0:
                prediction = self.model_right.predict(np.array([self.image_prep]))
                fiability = tf.keras.models.Model.predict(self.model_right, np.array([self.image_prep]))[0][np.argmax(prediction)]
                fiability *= 100
                fiability = format(fiability, ".2f")
                user_prediction = np.argmax(prediction)
                decode_user_prediction = self.encodeur_right.inverse_transform([user_prediction]) # Inversion encodage
                self.display_user(decode_user_prediction[0])

            else:
                prediction = self.model_left.predict(np.array([self.image_prep]))
                fiability = tf.keras.models.Model.predict(self.model_left, np.array([self.image_prep]))[0][np.argmax(prediction)]
                fiability *= 100
                fiability = format(fiability, ".2f")
                user_prediction = np.argmax(prediction)
                decode_user_prediction = self.encodeur_left.inverse_transform([user_prediction]) # Inversion encodage
                self.display_user(decode_user_prediction[0])

            # self.prediction_label.config(text=f"Prédiction du modèle : {prediction} avec {fiability}% de fiabilité.")
            self.prediction_label.config(text=f"{fiability}% de fiabilité qu'il s'agisse de :")

            # Affichage de l'oeil utilisé en utilisant la variable 'eye_type'
            if eye_type == 'right':
                self.prediction_label_eye.config(text="Oeil : droit")
            elif eye_type == 'left':
                self.prediction_label_eye.config(text="Oeil : gauche")

        else:
            self.prediction_label.config(text="Aucune image sélectionnée")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictionApp(root)
    root.mainloop()

    # Centre les éléments de l'interface graphique
    root.geometry("300x200")
    for widget in root.winfo_children():
        widget.pack(anchor="center")