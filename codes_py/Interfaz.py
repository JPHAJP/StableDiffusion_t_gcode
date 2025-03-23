import gradio as gr
from PIL import Image
import cv2
import time
import math
import speech_recognition as sr
import ollama
import os
import tempfile

from request_process_img import generate_image, load_image, process_image
from img_t_gcode2 import EdgesToGcode, convert_image_to_gcode
from gcode_t_img import plot_gcode
from gcode_t_ur import main as send_gcode_to_ur

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

def enhance_prompt(text, target_language="en"):
    """Genera un prompt optimizado para Stable Diffusion usando Ollama"""
    if not text or text.strip() == "":
        return ""
        
    # Instrucciones para Ollama - crear un prompt en inglés con estilo animado
    system_prompt = """
    Act as a prompt expert for Stable Diffusion.
        Convert the user description into a prompt optimized for generating images with Stable Diffusion.

        Rules:
        1. The prompt MUST be in English, regardless of the input language.
        2. Focus on generating an animated image.
        3. Keep the description simple but detailed.
        4. Include aspects such as: art style, lighting, colors, perspective.
        5. Final format: Just the prompt with no additional explanation.
        6. Use no more than 75 words.
        7. Never say you are generating a prompt.
        8. Just write the prompt as if you are describing a scene to an artist.
        9. Focus on making the image interesting but simple.
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
    """Procesa el audio y devuelve el texto reconocido"""
    if audio is None:
        return "", ""
        
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
    
    # Generar prompt mejorado con Ollama
    enhanced_prompt = enhance_prompt(text)
    
    return text, enhanced_prompt

def check_gcode_conditions(gcode_file, max_segment=700, max_lines=10000):
    """
    Verifica que:
    - Ninguna línea de movimiento (G0 o G1) supere max_segment (700 mm)
    - El número total de líneas no exceda max_lines (10000)
    """
    try:
        with open(gcode_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.startswith("G0") or l.startswith("G1")]
    except Exception as e:
        print("Error al leer el G-code:", e)
        return False

    if len(lines) > max_lines:
        print(f"Número de líneas ({len(lines)}) supera el máximo permitido ({max_lines}).")
        return False

    prev_x, prev_y = None, None
    for line in lines:
        try:
            parts = line.split()
            x_val, y_val = None, None
            for part in parts:
                if part.startswith("X"):
                    x_val = float(part[1:])
                elif part.startswith("Y"):
                    y_val = float(part[1:])
            if x_val is not None and y_val is not None:
                if prev_x is not None and prev_y is not None:
                    dist = math.hypot(x_val - prev_x, y_val - prev_y)
                    if dist > max_segment:
                        print(f"Segmento de {dist:.2f} mm excede el máximo permitido de {max_segment} mm.")
                        return False
                prev_x, prev_y = x_val, y_val
        except Exception as e:
            print("Error al parsear línea:", line, e)
            continue

    return True

def process_unified(final_prompt, uploaded_image, blur_kernel_size, min_contour_area, clahe_clip_limit, combine_with_original, progress=gr.Progress(track_tqdm=False)):
    """
    Función unificada que procesa la entrada según el prompt final o imagen cargada.
    """
    progress(0, desc="Iniciando proceso...")
    start_time = time.time()
    
    # Determinar qué entrada usar
    original = None
    image_source = "ninguna"
    
    # Si hay una imagen cargada, usar esa
    if uploaded_image is not None:
        original = uploaded_image
        image_source = "imagen cargada"
    # Si no hay imagen pero hay prompt, generar imagen
    elif final_prompt and final_prompt.strip():
        try:
            original = generate_image(final_prompt)
            image_source = "prompt"
        except Exception as e:
            return None, None, None, f"Error al generar imagen: {str(e)}"
    # Si no hay imagen ni prompt, mostrar error
    else:
        return None, None, None, "Debes ingresar un prompt o subir una imagen para continuar."
    
    # Redimensionar para no exceder 500x500 y guardar
    original.thumbnail((500, 500))
    original.save("original.png")
    
    yield (original, None, None, f"Fuente: {image_source}\nPrompt usado: {final_prompt}\nImagen cargada/generada")
    progress(0.2, desc="Imagen cargada/generada")

    # Procesar la imagen utilizando los parámetros ingresados o por defecto
    processed = process_image(
        original, 
        output_path="processed.png",
        blur_kernel_size=blur_kernel_size,
        min_contour_area=min_contour_area,
        clahe_clip_limit=clahe_clip_limit,
        combine_with_original=combine_with_original
    )
    yield (original, processed, None, f"Fuente: {image_source}\nPrompt usado: {final_prompt}\nImagen procesada para detección de bordes")
    progress(0.4, desc="Imagen procesada para detección de bordes")

    # Generar G-code a partir de la imagen procesada
    convert_image_to_gcode(
        image_input="processed.png",
        output_gcode="out.nc",
        edges_mode="black",
        threshold=32,
        scale=1.0,
        simplify=0.8,
        dot_output=None
    )
    
    # Verificar condiciones del G-code
    if not check_gcode_conditions("out.nc", max_segment=700, max_lines=10000):
        warning = f"Fuente: {image_source}\nPrompt usado: {final_prompt}\nEl G-code generado no cumple con las restricciones: verifique el número de líneas o el tamaño de segmento."
        yield (original, processed, None, warning)
        return

    # Si el G-code cumple las condiciones, se visualiza y se muestra el estado final
    try:
        plot_gcode("out.nc", "gcode_plot.png")
        gcode_plot = Image.open("gcode_plot.png")
        
        with open("out.nc", "r") as f:
            lines = [l.strip() for l in f.readlines() if l.startswith("G0") or l.startswith("G1")]
            num_lines = len(lines)
        
        total_time = time.time() - start_time
        final_status = (f"Fuente: {image_source}\n"
                        f"Prompt usado: {final_prompt}\n"
                        f"✅ G-code generado exitosamente en {total_time:.2f} segundos.\n"
                        f"Líneas: {num_lines}/10000")
        progress(1.0, desc="Proceso completado")
        yield (original, processed, gcode_plot, final_status)
        
    except Exception as e:
        yield (original, processed, None, f"Error al visualizar G-code: {str(e)}")

def send_gcode():
    try:
        send_gcode_to_ur()
        return "G-code enviado al robot UR exitosamente."
    except Exception as e:
        return f"Error al enviar G-code: {e}"

def update_text_prompt(text):
    """Mejora el prompt de texto y lo muestra"""
    if not text or text.strip() == "":
        return ""
    return enhance_prompt(text)

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Interfaz para generar/procesar imágenes y enviar G-code a UR")

        # Variables de estado para almacenar los prompts mejorados
        final_prompt = gr.State("")
        
        # Pestañas para los tres tipos de entrada
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs() as input_tabs:
                    with gr.TabItem("Texto") as text_tab:
                        with gr.Row():
                            with gr.Column():
                                text_input = gr.Textbox(
                                    label="Prompt original",
                                    placeholder="Ingresa un prompt aquí..."
                                )
                                text_enhance_btn = gr.Button("Mejorar Prompt")
                                
                            with gr.Column():
                                text_enhanced = gr.Textbox(
                                    label="Prompt mejorado",
                                    placeholder="Aquí aparecerá el prompt mejorado..."
                                )
                                text_use_btn = gr.Button("Usar este prompt")
                    
                    with gr.TabItem("Audio") as audio_tab:
                        with gr.Row():
                            with gr.Column():
                                audio_input = gr.Audio(
                                    sources=["microphone", "upload"], 
                                    type="filepath",
                                    label="Entrada de Audio"
                                )
                                audio_process_btn = gr.Button("Procesar Audio")
                                
                            with gr.Column():
                                audio_text = gr.Textbox(
                                    label="Texto reconocido",
                                    placeholder="Aquí aparecerá el texto reconocido..."
                                )
                                audio_enhanced = gr.Textbox(
                                    label="Prompt mejorado",
                                    placeholder="Aquí aparecerá el prompt mejorado..."
                                )
                                audio_use_btn = gr.Button("Usar este prompt")
                        
                    with gr.TabItem("Imagen") as image_tab:
                        image_input = gr.Image(
                            label="Carga una imagen",
                            type="pil"
                        )
                
                # Prompt final que será utilizado (visible para el usuario)
                prompt_display = gr.Textbox(
                    label="Prompt que será utilizado",
                    placeholder="El prompt seleccionado aparecerá aquí...",
                    interactive=False
                )
                
                # Parámetros de procesamiento (fuera de las pestañas)
                gr.Markdown("### Parámetros de procesamiento")
                blur_kernel_size = gr.Number(label="Tamaño del kernel de desenfoque", value=3)
                min_contour_area = gr.Number(label="Área mínima del contorno", value=5)
                clahe_clip_limit = gr.Number(label="Límite de clip para CLAHE", value=1.5)
                combine_with_original = gr.Checkbox(label="Combinar con original", value=True)
                
                # Botón único para procesar
                process_button = gr.Button("Procesar", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                original_output = gr.Image(label="Imagen original / generada")
                processed_output = gr.Image(label="Imagen procesada (bordes)")
                gcode_output = gr.Image(label="Visualización del G-code")
                status_output = gr.Textbox(label="Estado del proceso", lines=5)
        
        # Eventos para la entrada de texto
        text_enhance_btn.click(
            fn=update_text_prompt,
            inputs=text_input,
            outputs=text_enhanced
        )
        
        text_use_btn.click(
            fn=lambda x: x,
            inputs=text_enhanced,
            outputs=prompt_display
        ).then(
            fn=lambda x: x,
            inputs=text_enhanced,
            outputs=final_prompt
        )
        
        # Eventos para la entrada de audio
        audio_process_btn.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[audio_text, audio_enhanced]
        )
        
        audio_use_btn.click(
            fn=lambda x: x,
            inputs=audio_enhanced,
            outputs=prompt_display
        ).then(
            fn=lambda x: x,
            inputs=audio_enhanced,
            outputs=final_prompt
        )
        
        # Un solo botón de procesamiento para todas las entradas
        process_button.click(
            fn=process_unified,
            inputs=[
                final_prompt, image_input,
                blur_kernel_size, min_contour_area, 
                clahe_clip_limit, combine_with_original
            ],
            outputs=[original_output, processed_output, gcode_output, status_output],
            queue=True
        )

        gr.Markdown("### Enviar G-code al robot UR")
        send_button = gr.Button("Enviar G-code")
        send_status = gr.Textbox(label="Estado de envío")
        send_button.click(fn=send_gcode, inputs=[], outputs=send_status)

        demo.queue().launch(share=True)

if __name__ == "__main__":
    main()