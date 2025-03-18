import requests
import base64
from PIL import Image
import io

def main():
    # URL de la API
    api_url = "http://localhost:8000/"

    # Definir el payload de la petición
    payload = {
        "modelInputs": {
            "prompt": "Una casa estilo zaha hadid en un paisaje futurista con un cielo de colores brillantes",
            "num_inference_steps": 60,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        },
        "callInputs": {
            "MODEL_ID": "runwayml/stable-diffusion-v1-5",
            "PIPELINE": "StableDiffusionPipeline",
            "SCHEDULER": "LMSDiscreteScheduler",
            "safety_checker": False
        }
    }

    headers = {"Content-Type": "application/json"}

    print("Enviando solicitud a la API...")
    response = requests.post(api_url, json=payload, headers=headers)
    print (response)

    if response.status_code == 200:
        data = response.json()
        # Se espera que la respuesta tenga la imagen en "image_base64"
        if "image_base64" in data:
            img_base64 = data["image_base64"]
            # Decodificar la cadena base64 a bytes
            img_bytes = base64.b64decode(img_base64)
            # Abrir la imagen con PIL
            image = Image.open(io.BytesIO(img_bytes))
            print(f"Dimensiones de la imagen recibida: {image.size}")
            # Mostrar la imagen
            image.show()

            # Preguntar si se desea guardar la imagen
            filename = "output.jpg"
            image.save(filename)
            print(f"Imagen guardada como {filename}")
        else:
            print("La respuesta no contiene la clave 'image_base64'.")
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print(response.text)


    

if __name__ == '__main__':
    main()
