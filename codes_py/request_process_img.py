import requests
import base64
import io
import numpy as np
import cv2
from PIL import Image, ImageFilter
import os

def generate_image(prompt: str) -> Image.Image:
    """
    Genera una imagen a partir del prompt usando la API de Stable Diffusion.
    Guarda la imagen original como 'output.png' y la muestra.
    
    Parámetro:
        prompt (str): Texto descriptivo para generar la imagen.
    
    Retorna:
        Image.Image: La imagen generada.
    """
    api_url = "http://localhost:8000/"
    
    payload = {
        "modelInputs": {
            "prompt": prompt,
            "num_inference_steps": 30,
            "guidance_scale": 5.5,
            "width": 400,
            "height": 400
        },
        "callInputs": {
            "MODEL_ID": "runwayml/stable-diffusion-v1-5",
            "PIPELINE": "StableDiffusionPipeline",
            "SCHEDULER": "LMSDiscreteScheduler",
            "safety_checker": True
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("Enviando solicitud a la API...")
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if "image_base64" in data:
            img_base64 = data["image_base64"]
            img_bytes = base64.b64decode(img_base64)
            image = Image.open(io.BytesIO(img_bytes))
            
            image.save("output.png")
            print("Imagen original guardada como 'output.png'")
            #image.show()
            
            return image
        else:
            raise ValueError("La respuesta no contiene la clave 'image_base64'.")
    else:
        raise Exception(f"Error en la solicitud: {response.status_code}\n{response.text}")

def load_image(file_path: str) -> Image.Image:
    """
    Carga una imagen desde la ruta especificada.
    
    Parámetro:
        file_path (str): Ruta del archivo de imagen.
    
    Retorna:
        Image.Image: La imagen cargada.
    """
    try:
        image = Image.open(file_path)
        print(f"Imagen cargada desde {file_path}")
        #image.show()
        return image
    except Exception as e:
        raise Exception(f"No se pudo cargar la imagen: {e}")

def process_image(
    input_image_path, 
    output_path="output_processed.png", 
    # Parámetros de preprocesamiento
    blur_kernel_size=3,                # Tamaño del kernel para el desenfoque gaussiano (impar)
    clahe_clip_limit=1.5,              # Límite de contraste para CLAHE
    clahe_grid_size=(8, 8),            # Tamaño de cuadrícula para CLAHE
    
    # Parámetros de detección de bordes
    canny_threshold_min=100,            # Umbral mínimo para Canny
    canny_threshold_max=200,           # Umbral máximo para Canny
    
    # Parámetros de operaciones morfológicas
    morph_kernel_size=2,               # Tamaño del kernel para operaciones morfológicas
    morph_iterations=1,                # Número de iteraciones para operaciones morfológicas
    
    # Parámetros de filtrado de contornos
    min_contour_area=5,                # Área mínima de contorno para filtrar ruido
    contour_thickness=2,               # Grosor de línea al dibujar contornos
    
    # Opciones adicionales
    combine_with_original=True,        # Combinar con bordes Canny originales
    invert_output=True                # Invertir la salida (bordes negros en fondo blanco)
):
    """
    Procesa una imagen para detectar bordes con parámetros totalmente configurables.
    
    Parámetros:
        input_image_path (str o Image.Image): Ruta de la imagen o objeto Image a procesar.
        output_path (str): Ruta donde guardar la imagen procesada.
        
        # Parámetros de preprocesamiento
        blur_kernel_size (int): Tamaño del kernel para el desenfoque gaussiano (debe ser impar).
        clahe_clip_limit (float): Límite de recorte para CLAHE (1.0-4.0).
        clahe_grid_size (tuple): Tamaño de la cuadrícula para CLAHE.
        
        # Parámetros de detección de bordes
        canny_threshold_min (int): Umbral mínimo para la detección de bordes (0-255).
        canny_threshold_max (int): Umbral máximo para la detección de bordes (0-255).
        
        # Parámetros de operaciones morfológicas
        morph_kernel_size (int): Tamaño del kernel para operaciones morfológicas.
        morph_iterations (int): Número de iteraciones para operaciones morfológicas.
        
        # Parámetros de filtrado de contornos
        min_contour_area (float): Área mínima de contorno para filtrar ruido.
        contour_thickness (int): Grosor de línea al dibujar contornos.
        
        # Opciones adicionales
        combine_with_original (bool): Si es True, combina con los bordes Canny originales.
        invert_output (bool): Si es True, invierte la salida (bordes negros en fondo blanco).
        
    Retorna:
        Image.Image: La imagen procesada con los bordes detectados.
    """
    # Asegurar que el tamaño del kernel de desenfoque sea impar
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    # Verificar si la entrada es una ruta o un objeto Image
    if isinstance(input_image_path, str):
        image_cv = cv2.imread(input_image_path)
    else:
        image_pil = input_image_path
        image_np = np.array(image_pil)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
    
    # 1. Preprocesamiento - reducción de ruido 
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    # 2. Mejora de contraste con CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
    equalized = clahe.apply(blurred)
    
    # 3. Detección de bordes con Canny
    edges = cv2.Canny(equalized, canny_threshold_min, canny_threshold_max)
    
    # 4. Operaciones morfológicas para conectar bordes cercanos
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
    # 5. Filtrado de ruido basado en área de contorno
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una máscara para el resultado
    filtered_mask = np.zeros_like(closed_edges)
    
    # Filtrar contornos según el área mínima
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(filtered_mask, [contour], -1, (255,255,255), contour_thickness)
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(filtered_mask, [approx_curve], -1, (255, 255, 255), contour_thickness)

    
    # 6. Combinar con la imagen original de Canny si se solicita
    if combine_with_original:
        result = cv2.bitwise_or(filtered_mask, edges)
    else:
        result = filtered_mask
    
    # 7. Invertir la salida si se solicita
    if invert_output:
        result = cv2.bitwise_not(result)
    
    #kernel = np.ones((3, 3), np.uint8)
     # Erosion: The pixel value is set to the minimum value within the kernel's neighborhood
    result = cv2.erode(result, kernel, iterations=1)
     # Dilation: The pixel value is set to the maximum value within the kernel's neighborhood
    result = cv2.dilate(result, kernel, iterations=1)
   

    # Convertir resultado a formato PIL para guardar
    result_pil = Image.fromarray(result)
    
    # Guardar la imagen resultante
    result_pil.save(output_path)
    print(f"Imagen procesada guardada como '{output_path}'")
    return result_pil

if __name__ == '__main__':
    modo = input("¿Deseas generar una imagen (g) o cargar una imagen existente (c)? ").strip().lower()
    if modo == 'g':
        prompt_text = "A dashhound dog running in a field of flowers."
        img = generate_image(prompt_text)
    elif modo == 'c':
        #file_path = input("Ingresa la ruta de la imagen a cargar: ").strip()
        #img = load_image(file_path)
        pass
    else:
        raise Exception("Opción no válida. Elige 'g' para generar o 'c' para cargar.")
    
    # Procesar la imagen (ya sea generada o cargada)
    process_image('img/output5.png', "bordes_simplificados.png", blur_kernel_size=3, min_contour_area=5, combine_with_original=True)