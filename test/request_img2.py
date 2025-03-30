import requests
import base64
import json

# Endpoint de la API
url = "http://localhost:7860/sdapi/v1/txt2img"

# Parámetros para generar la imagen
payload = {
    "prompt": "a futuristic cyberpunk city street at night, wet asphalt reflecting vibrant neon lights, glowing purple and blue hues, cyberpunk aesthetic, high-tech buildings with holographic advertisements, misty atmosphere, sci-fi cars, pedestrians with glowing outfits, cinematic lighting, ultra-detailed, 4K, dynamic perspective, realistic reflections, rain-drenched urban environment, deep shadows and vivid highlights",
    "negative_prompt": "low resolution, blurry, pixelated, distorted, ugly, low detail, bad anatomy, poorly drawn buildings, artifacts, watermark, text, duplicate, out of frame",
    "steps": 30,
    "cfg_scale": 7,
    "width": 512,
    "height": 512,
    "sampler_index": "Euler",
    "seed": -1,
    "batch_size": 1,
    "n_iter": 1,
    "send_images": True,
    "save_images": False
}

# Realiza el POST
response = requests.post(url, json=payload)

# Verifica si fue exitoso
if response.status_code == 200:
    data = response.json()

    # Decodifica la imagen base64 y la guarda
    image_base64 = data['images'][0]
    image_data = base64.b64decode(image_base64)
    with open("output.png", "wb") as f:
        f.write(image_data)
    print("✅ Imagen guardada como 'output.png'")

    # Imprime la información detallada
    info_raw = data.get('info')
    if info_raw:
        try:
            info_json = json.loads(info_raw)
            print("\n📄 Información de la imagen generada:\n")
            for key, value in info_json.items():
                print(f"{key}: {value}")
        except json.JSONDecodeError:
            print("⚠️ No se pudo decodificar la info como JSON.")
            print(info_raw)
    else:
        print("⚠️ No se recibió información adicional.")

else:
    print(f"❌ Error al generar imagen: {response.status_code}")
    print(response.text)
