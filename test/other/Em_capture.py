import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

cap = None
run = False

def iniciar_camara():
    global cap, run
    run = True
    cap = cv2.VideoCapture(0)
    mostrar_camara()

def mostrar_camara():
    global cap, run
    if not run or cap is None:
        return

    ret, frame = cap.read()
    if ret:
        # Convertir a escala de grises
        h, w, c = frame.shape
        black = np.zeros((h, w, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(black, contours, -1, (255, 255, 255), 1)

        # Invertir colores
        black = cv2.bitwise_not(black)

        img = Image.fromarray(black)
        img_tk = ImageTk.PhotoImage(image=img)

        # Mostrar camara
        camera_label.img_tk = img_tk
        camera_label.configure(image=img_tk)

    # Actualizar el frame
    camera_label.after(10, mostrar_camara)

def tomar_foto():
    global cap
    if cap is not None and run:
        ret, frame = cap.read()
        if ret:
            h, w, c = frame.shape
            black = np.zeros((h, w, 3), dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(black, contours, -1, (255, 255, 255), 1)
            black = cv2.bitwise_not(black)

            # Guardar la imagen
            cv2.imwrite('./img1.png', black)
            message_label.config(text="Imagen guardada")

# Crear la interfaz
root = tk.Tk()
root.title("Cámara Contornos")
root.geometry("950x750")
root.configure(bg="#f0f0f0")

title = tk.Label(
    root, text="Cámara Contornos", font=("Arial", 16, "bold"), bg="#f0f0f0"
)
title.pack(pady=10)

# Mostrar la cámara
camera_label = tk.Label(root, bg="black")
camera_label.pack(pady=10)

# Botón Tomar foto
capture_button = tk.Button(
    root, text="Capturar Imagen", font=("Arial", 12), bg="lightgreen", fg="black", command=tomar_foto
)
capture_button.pack(pady=5)

message_label = tk.Label(root, text="", font=("Arial", 12), bg="#f0f0f0", fg="black")
message_label.pack(pady=10)

iniciar_camara()

root.mainloop()