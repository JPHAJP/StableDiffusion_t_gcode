import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox


def cord_a_gcode(x, y, z=None, feedrate=100):
    if z is not None:
        return f"G1 X{x:.2f} Y{y:.2f} Z{z:.2f} F{feedrate}"
    else:
        return f"G1 X{x:.2f} Y{y:.2f} F{feedrate}"


def png_a_gcode(input_file, output_file):
    # Cargar la imagen (escala de grises)
    img = Image.open(input_file).convert("L")
    img_array = np.array(img)

    # Mejorar calidad
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    # Binarización
    binary_img = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2
    )

    # Encontrar contornos
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gcode = []
    gcode.append("G21 ; Configurar unidades en milímetros")
    gcode.append("G90 ; Modo absoluto")
    gcode.append("G0 Z20 ; Levantar herramienta inicial")

    for contour in contours:
        # Suavizar los contornos
        epsilon = 0.002 * cv2.arcLength(contour, True)  # Más detalle
        approx_curve = cv2.approxPolyDP(contour, epsilon, True)

        # Levantar herramienta
        gcode.append("G0 Z20 ; Levantar herramienta")

        # Mover al inicio 
        start_x, start_y = approx_curve[0][0]
        gcode.append(f"G0 X{start_x:.2f} Y{binary_img.shape[0] - start_y:.2f} ; Mover al inicio")

        # Bajar herramienta 
        gcode.append("G1 Z-1.0 F100 ; Bajar herramienta")

        # Generar G-code
        for point in approx_curve:
            x, y = point[0]
            gcode.append(cord_a_gcode(x, binary_img.shape[0] - y))

        # Levantar herramienta al final
        gcode.append("G0 Z20 ; Levantar herramienta")


    # Guardar el archivo G-code
    try:
        with open(output_file, "w") as f:
            f.write("\n".join(gcode))
    except Exception as e:
        raise Exception(f"No se pudo guardar el archivo")


def selec_png():

    input_file = filedialog.askopenfilename(
        title="Seleccionar archivo PNG",
        filetypes=[("Archivos PNG", "*.png")]
    )
    if not input_file:
        messagebox.showinfo("Error", "No se seleccionó ningún archivo.")
        return

    output_file = filedialog.asksaveasfilename(
        title="Guardar archivo G-code",
        defaultextension=".nc",
        filetypes=[("Archivos G-code", "*.nc")]
    )
    if not output_file:
        messagebox.showinfo("Error", "No se guardó el archivo.")
        return

    try:
        png_a_gcode(input_file, output_file)
        messagebox.showinfo("", f"G-code guardado")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error")


def cerrar_programa():
    root.destroy()


# Crear interfaz
root = tk.Tk()
root.title("Convertidor de PNG a G-code")
root.geometry("600x300")

frame = tk.Frame(root, bg="white")
frame.pack(expand=True, fill="both")

label = tk.Label(frame, text="Convertidor de PNG a G-code", font=("Arial", 16), bg="white", fg="black")
label.pack(pady=20)

# Botón para seleccionar archivo PNG
boton_selec = tk.Button(
    frame, text="Convertir PNG a G-code",
    font=("Arial", 12), bg="#ADD8E6", fg="black",
    command=selec_png
)
boton_selec.pack(pady=10)

# Cerrar el programa
boton_cerrar = tk.Button(
    frame, text="Cerrar programa",
    font=("Arial", 12), bg="#20B2AA", fg="black",
    command=cerrar_programa
)
boton_cerrar.pack(pady=10)

# Iniciar el programa
root.mainloop()