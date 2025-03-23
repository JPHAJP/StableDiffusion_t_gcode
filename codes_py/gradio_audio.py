import gradio as gr
import speech_recognition as sr
import ollama
import os
import tempfile

def speech_to_text(audio_path):
    """Convierte audio a texto usando Google Speech Recognition"""
    recognizer = sr.Recognizer()
    
    try:
        # Cargar el archivo de audio
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
        # Reconocer usando Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
    
    except Exception as e:
        return f"Error en reconocimiento de voz: {str(e)}"

def generate_sd_prompt(text, target_language="en"):
    """Genera un prompt optimizado para Stable Diffusion usando Ollama"""
    
    # Instrucciones para Ollama - crear un prompt en inglés con estilo animado
    system_prompt = """
    Act as a prompt expert for Stable Diffusion.
        Convert the user description into a prompt optimized for generating images with Stable Diffusion.

        Rules:
        1. The prompt MUST be in English, regardless of the input language.
        2. Focus on generating an animated/cartoon-style image.
        3. Keep the description simple but detailed.
        4. Include aspects such as: art style, lighting, colors, perspective.
        5. Final format: Just the prompt with no additional explanation.
        6. Use no more than 75 words.
        7. Never say you are generating a prompt.
        8. Just write the prompt as if you are describing a scene to an artist.
    """
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Make a description to make an image of: {text}"
        }
    ]
    
    try:
        # Llamada a Ollama para generar el prompt
        response = ollama.chat(model='llama3.1:latest', messages=messages)
        return response['message']['content'].strip()
    
    except Exception as e:
        return f"Error al generar prompt con Ollama: {str(e)}"

def process_audio(audio):
    """Función principal que procesa el audio y devuelve el resultado"""
    # Guardar temporalmente el audio
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
    
    # Asegurarse de que audio sea bytes o un archivo
    if isinstance(audio, str):
        temp_audio_path = audio
    else:
        with open(temp_audio_path, "wb") as f:
            f.write(audio)
    
    # Conversión de voz a texto
    text = speech_to_text(temp_audio_path)
    
    # Generar prompt con Ollama
    sd_prompt = generate_sd_prompt(text)
    
    return text, sd_prompt

# Interfaz Gradio
with gr.Blocks(title="Generador de Prompts para Stable Diffusion") as demo:
    gr.Markdown("# 🎤 De Voz a Prompt para Stable Diffusion")
    gr.Markdown("Habla y convierte tu descripción en un prompt optimizado para generar imágenes con Stable Diffusion")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath",
                label="Entrada de Audio"
            )
            process_btn = gr.Button("Procesar Audio", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="Texto Reconocido")
            prompt_output = gr.Textbox(label="Prompt para Stable Diffusion (en inglés)")
    
    process_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[text_output, prompt_output]
    )

if __name__ == "__main__":
    demo.launch()