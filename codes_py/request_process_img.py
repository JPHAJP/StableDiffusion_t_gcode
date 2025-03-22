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

def process_image(input_image_path, output_path="output_processed.png", threshold_min=100, threshold_max=200):
    """
    Procesa una imagen para detectar bordes de forma avanzada, obteniendo líneas continuas.
    Invierte los colores al final (bordes blancos sobre fondo negro).
    
    Parámetros:
        input_image_path (str o Image.Image): Ruta de la imagen o objeto Image a procesar.
        output_path (str): Ruta donde guardar la imagen procesada.
        threshold_min (int): Umbral mínimo para la detección de bordes (0-255).
        threshold_max (int): Umbral máximo para la detección de bordes (0-255).
        
    Retorna:
        Image.Image: La imagen procesada con los bordes detectados (invertida).
    """
    # Verificar si la entrada es una ruta o un objeto Image
    if isinstance(input_image_path, str):
        # Cargar la imagen con OpenCV (mejor para procesamiento)
        image_cv = cv2.imread(input_image_path)
    else:
        # Convertir objeto PIL Image a formato OpenCV
        image_pil = input_image_path
        image_np = np.array(image_pil)
        # Convertir RGB a BGR (formato OpenCV)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_cv = image_np
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Ecualización del histograma para mejorar contraste
    equalized = cv2.equalizeHist(blurred)
    
    # Detector de bordes Canny (mejor para líneas continuas)
    edges = cv2.Canny(equalized, threshold_min, threshold_max)
    
    # Dilatación para rellenar pequeños huecos en las líneas
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Erosión para adelgazar las líneas dilatadas
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    # Transformación morfológica para mejorar la continuidad
    closing = cv2.morphologyEx(eroded_edges, cv2.MORPH_CLOSE, kernel)
    
    # Aplicar umbral adaptativo para mejorar la definición
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Combinar los resultados con una operación bitwise OR
    combined = cv2.bitwise_or(closing, adaptive_threshold)
    
    # Invertir los colores para tener bordes blancos sobre fondo negro
    inverted = cv2.bitwise_not(combined)
    
    # Convertir resultado a formato PIL para guardar
    result_pil = Image.fromarray(inverted)
    
    # Guardar la imagen resultante
    result_pil.save(output_path)
    print(f"Imagen procesada guardada como '{output_path}'")
    
    return result_pil

def process_image_alternative(input_image_path, output_path="output_simplified.png", simplify_level=10):
    """
    Método alternativo para simplificar una imagen y extraer solo los contornos principales.
    Utiliza cuantización de color y detección de contornos para simplificar.
    
    Parámetros:
        input_image_path (str o Image.Image): Ruta de la imagen o objeto Image a procesar.
        output_path (str): Ruta donde guardar la imagen procesada.
        simplify_level (int): Nivel de simplificación (1-20, menor = más detalles).
        
    Retorna:
        Image.Image: La imagen procesada con contornos simplificados.
    """
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
    
    # Fuerte desenfoque para eliminar detalles
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Cuantización de la imagen (reducir número de niveles de gris)
    num_levels = max(2, 256 // simplify_level)
    quantized = np.floor(blurred / (256 / num_levels)) * (256 / num_levels)
    quantized = quantized.astype(np.uint8)
    
    # Detectar bordes con Laplaciano (más simple que Canny)
    laplacian = cv2.Laplacian(quantized, cv2.CV_8U, ksize=3)
    
    # Umbralización adaptativa
    thresholded = cv2.adaptiveThreshold(
        laplacian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Eliminar ruido con operaciones morfológicas
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # Detección de contornos para encontrar solo los principales
    contours, _ = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una imagen en blanco para dibujar solo los contornos importantes
    contour_img = np.zeros_like(gray)
    
    # Filtrar contornos por área para mantener solo los más grandes/importantes
    min_area = (gray.shape[0] * gray.shape[1]) / (100 * simplify_level)
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Dibujar los contornos importantes
    cv2.drawContours(contour_img, significant_contours, -1, 255, 1)
    
    # Invertir para tener líneas blancas sobre fondo negro
    inverted = cv2.bitwise_not(contour_img)
    
    # Convertir resultado a formato PIL para guardar
    result_pil = Image.fromarray(inverted)
    
    # Guardar la imagen resultante
    result_pil.save(output_path)
    print(f"Imagen alternativa simplificada guardada como '{output_path}'")
    
    return result_pil

if __name__ == '__main__':
    modo = input("¿Deseas generar una imagen (g) o cargar una imagen existente (c)? ").strip().lower()
    if modo == 'g':
        prompt_text = "A dashhound dog running in a field of flowers."
        img = generate_image(prompt_text)
    elif modo == 'c':
        file_path = input("Ingresa la ruta de la imagen a cargar: ").strip()
        img = load_image(file_path)
    else:
        raise Exception("Opción no válida. Elige 'g' para generar o 'c' para cargar.")
    
    # Procesar la imagen (ya sea generada o cargada)
    process_image(img, "bordes_simplificados.png")
    process_image_alternative(img, "contornos_principales.png", simplify_level=1)