import requests
import base64
import io
from PIL import Image, ImageFilter

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

def process_image(image: Image.Image) -> Image.Image:
    """
    Procesa la imagen para obtener un dibujo de líneas simple:
    - Convierte la imagen a escala de grises.
    - Aplica un desenfoque previo para suavizar la imagen.
    - Detecta los bordes con FIND_EDGES.
    - Aplica un desenfoque posterior para suavizar los bordes.
    - Reduce la resolución a la mitad en cada dimensión.
    
    Guarda la imagen procesada como 'output_processed.png' y la muestra.
    
    Parámetro:
        image (Image.Image): La imagen a procesar.
        
    Retorna:
        Image.Image: La imagen procesada.
    """
    gray_image = image.convert("L")
    pre_smoothed = gray_image.filter(ImageFilter.GaussianBlur(radius=1))
    edges = pre_smoothed.filter(ImageFilter.FIND_EDGES)
    smooth_edges = edges.filter(ImageFilter.GaussianBlur(radius=1))
    
    new_size = (smooth_edges.width // 1, smooth_edges.height // 1)
    processed_image = smooth_edges.resize(new_size, resample=Image.NEAREST)
    
    processed_image.save("output_processed.png")
    print("Imagen procesada guardada como 'output_processed.png'")
    #processed_image.show()
    
    return processed_image

if __name__ == '__main__':
    modo = input("¿Deseas generar una imagen (g) o cargar una imagen existente (c)? ").strip().lower()
    if modo == 'g':
        prompt_text = "A beatifull sunflower, big and nice with a shiny sun, animated, logo style, simple."
        img = generate_image(prompt_text)
    elif modo == 'c':
        file_path = input("Ingresa la ruta de la imagen a cargar: ").strip()
        img = load_image(file_path)
    else:
        raise Exception("Opción no válida. Elige 'g' para generar o 'c' para cargar.")
    
    # Procesar la imagen (ya sea generada o cargada)
    process_image(img)

#https://github.com/kiri-art/docker-diffusers-api/blob/dev/README.md
#https://github.com/Stypox/image-to-gcode
#https://github.com/jherrm/gcode-viewer

#export

#sudo docker build -t gadicc/diffusers-api .
#sudo docker run --gpus all -p 8000:8000 -e HF_AUTH_TOKEN gadicc/diffusers-api
#python3 image-to-gcode/image_to_gcode.py --input image.png --output graph.nc --threshold 100 --scale 0.02