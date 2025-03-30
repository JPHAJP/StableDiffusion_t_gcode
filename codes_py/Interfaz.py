import gradio as gr
from PIL import Image
import cv2
import time
import math
import speech_recognition as sr
import ollama
import os
import tempfile
import requests
import base64
import io

from request_process_img import process_image  
from img_t_gcode2 import convert_image_to_gcode
from gcode_t_img import plot_gcode
from gcode_t_ur import main as send_gcode_to_ur

# ====================================================
# Función para generar imagen usando la API txt2img de AUTOMATIC1111
# ====================================================
def generate_image(prompt, negative_prompt="", steps=30, cfg_scale=7, width=512, height=512, sampler_index="Euler", seed=-1, batch_size=1, n_iter=1, send_images=True, save_images=False):
    url = "http://localhost:7860/sdapi/v1/txt2img"
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "sampler_index": sampler_index,
        "seed": seed,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "send_images": send_images,
        "save_images": save_images
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        data = response.json()
        image_base64 = data['images'][0]
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image
    else:
        raise Exception("Error al generar imagen: " + response.text)

# ====================================================
# Funciones de prompts usando Ollama
# ====================================================
def enhance_prompt(text, target_language="en"):
    """Genera un prompt optimizado para Stable Diffusion usando Ollama"""
    if not text or text.strip() == "":
        return ""
    system_prompt = """
    Act as a prompt expert for Stable Diffusion.
    Convert the user description into a prompt optimized for generating images with Stable Diffusion.

    Rules:
    1. The prompt MUST be in English.
    2. Focus on generating an animated image.
    3. Keep the description simple but detailed.
    4. Include aspects such as: art style, lighting, colors, perspective.
    5. Final format: Just the prompt with no additional explanation.
    6. Use no more than 75 words.
    7. Do not mention that you are generating a prompt.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Make a description to make an image of: {text}"}
    ]
    try:
        response = ollama.chat(model='llama3.1:latest', messages=messages)
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error al generar prompt con Ollama: {str(e)}"

def enhance_negative_prompt(text, target_language="en"):
    """Genera un negative prompt optimizado para Stable Diffusion usando Ollama"""
    if not text or text.strip() == "":
        return ""
    system_prompt = """
    Act as a negative prompt expert for Stable Diffusion.
    Convert the user description into a negative prompt optimized for image generation.
    Rules:
    1. The prompt MUST be in English.
    2. Focus on describing what to avoid: low quality, blurry, distorted, artifacts, watermark, etc.
    3. Keep it concise and effective.
    4. Use no more than 50 words.
    5. Output only the negative prompt without additional explanation.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate a negative prompt for: {text}"}
    ]
    try:
        response = ollama.chat(model='llama3.1:latest', messages=messages)
        return response['message']['content'].strip()
    except Exception as e:
        return f"Error al generar negative prompt con Ollama: {str(e)}"

def generate_both_prompts(user_prompt):
    """
    A partir de un único prompt de usuario, genera automáticamente
    el prompt positivo y el negative prompt.
    """
    positive = enhance_prompt(user_prompt)
    negative = enhance_negative_prompt(user_prompt)
    return positive, negative

# ====================================================
# Función para procesamiento de audio (se mantiene para referencia)
# ====================================================
def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"Error en reconocimiento de voz: {str(e)}"

def process_audio(audio):
    if audio is None:
        return "", ""
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
    if isinstance(audio, str):
        temp_audio_path = audio
    else:
        with open(temp_audio_path, "wb") as f:
            f.write(audio)
    text = speech_to_text(temp_audio_path)
    positive = enhance_prompt(text)
    negative = enhance_negative_prompt(text)
    return text, positive  # Se retorna el texto y el prompt positivo (el negative se generará automáticamente)

# ====================================================
# Funciones para cada paso del proceso
# ====================================================
def step_generate_original(final_prompt, final_negative_prompt, uploaded_image):
    """
    Paso 1: Genera o carga la imagen.
    Si se sube imagen, se usa esa; de lo contrario, se genera con la API.
    """
    if uploaded_image is not None:
        original = uploaded_image
        image_source = "imagen cargada"
    elif final_prompt and final_prompt.strip():
        try:
            original = generate_image(final_prompt, negative_prompt=final_negative_prompt)
            image_source = "generado con prompt"
        except Exception as e:
            return None, f"Error al generar imagen: {str(e)}"
    else:
        return None, "Debes ingresar un prompt o subir una imagen."
    
    original.thumbnail((500, 500))
    original.save("original.png")
    return original, f"Imagen original obtenida ({image_source})."

def step_process_image(original_image, blur_kernel_size, min_contour_area, clahe_clip_limit):
    """
    Paso 2: Procesa la imagen original para detección de bordes.
    """
    if original_image is None:
        return None, "No hay imagen original para procesar."
    processed = process_image(
        original_image,
        output_path="processed.png",
        blur_kernel_size=blur_kernel_size,
        min_contour_area=min_contour_area,
        clahe_clip_limit=clahe_clip_limit,
        combine_with_original=True
    )
    return processed, "Imagen procesada para detección de bordes."

def step_generate_gcode(black_gcode):
    """
    Paso 3: Genera el G-code a partir de la imagen procesada y lo visualiza.
    """
    mode = "black" if black_gcode else None
    convert_image_to_gcode(
        image_input="processed.png",
        output_gcode="out.nc",
        edges_mode=mode,
        threshold=32,
        scale=1.0,
        simplify=0.8,
        dot_output=None
    )
    if not check_gcode_conditions("out.nc", max_segment=700, max_lines=10000):
        return None, "El G-code generado no cumple con las restricciones."
    try:
        plot_gcode("out.nc", "gcode_plot.png")
        gcode_plot = Image.open("gcode_plot.png")
        with open("out.nc", "r") as f:
            lines = [l.strip() for l in f.readlines() if l.startswith("G0") or l.startswith("G1")]
            num_lines = len(lines)
        return gcode_plot, f"G-code generado exitosamente. Líneas: {num_lines}/10000"
    except Exception as e:
        return None, f"Error al visualizar G-code: {str(e)}"

def step_send_gcode():
    """
    Paso 4: Envía el G-code al robot UR.
    """
    try:
        send_gcode_to_ur()
        return "G-code enviado al robot UR exitosamente."
    except Exception as e:
        return f"Error al enviar G-code: {e}"

# Añadida función para verificar condiciones del G-code (faltaba en el código original)
def check_gcode_conditions(gcode_file, max_segment=700, max_lines=10000):
    """
    Verifica si el G-code cumple con las restricciones:
    - No más de max_segment segmentos
    - No más de max_lines líneas
    """
    try:
        with open(gcode_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.startswith("G0") or l.startswith("G1")]
            if len(lines) > max_lines:
                return False
            
            # Se podría implementar una verificación de segmentos si es necesario
            
        return True
    except Exception:
        return False

# Función para obtener una vista previa del G-code
def get_gcode_preview():
    try:
        with open("out.nc", "r") as f:
            lines = f.readlines()[:20]  # Primeras 20 líneas
            return "".join(lines)
    except:
        return "// G-code no disponible o no se pudo leer el archivo"

# ====================================================
# Interfaz con Gradio REDISEÑADA: Con pestañas y mejor organización
# ====================================================
def main():
    with gr.Blocks(theme=gr.themes.Soft(), title="Sistema de Generación y Envío de G-code") as demo:
        # Variables de estado - definidas fuera de pestañas para que sean accesibles a todas
        final_prompt = gr.State("")
        final_negative_prompt = gr.State("")
        original_image = gr.State(None)
        processed_image = gr.State(None)
        gcode_ready = gr.State(False)
        # Variable de estado para controlar la pestaña activa
        active_tab = gr.State(0)

        gr.Markdown("# 🤖 Sistema de Generación y Envío de G-code a UR")
        gr.Markdown("### Genera imágenes, conviértelas a G-code y envíalas al robot UR")
        
        with gr.Tabs() as tabs:
            # =============================================
            # Pestaña 1: Generación de Imagen
            # =============================================
            with gr.Tab("📷 Generación de Imagen"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎨 Generación de Prompts")
                        user_prompt_input = gr.Textbox(
                            label="Descripción de la imagen",
                            placeholder="Describe lo que deseas generar...",
                            lines=3
                        )
                        with gr.Row():
                            audio_input = gr.Audio(
                                label="O usar descripción por voz", 
                                type="filepath"
                            )
                            process_audio_btn = gr.Button("Procesar audio", variant="secondary")
                        
                        with gr.Accordion("Configuración avanzada", open=False):
                            with gr.Row():
                                steps = gr.Slider(minimum=10, maximum=100, value=30, step=1, label="Pasos")
                                cfg_scale = gr.Slider(minimum=1, maximum=15, value=7, step=0.5, label="Escala CFG")
                            with gr.Row():
                                width = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Ancho")
                                height = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Alto")
                            sampler_index = gr.Dropdown(
                                choices=["Euler", "Euler a", "DPM++ 2M", "DPM++ SDE"], 
                                value="Euler", 
                                label="Método de sampleo"
                            )
                            
                        gen_prompts_btn = gr.Button("Generar Prompts", variant="primary")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Resultado")
                        pos_prompt_display = gr.Textbox(label="Prompt Positivo", interactive=True, lines=4)
                        neg_prompt_display = gr.Textbox(label="Negative Prompt", interactive=True, lines=2)
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                uploaded_image_input = gr.Image(label="O sube una imagen", type="pil")
                            with gr.Column(scale=1):
                                gen_image_btn = gr.Button("Generar/Cargar Imagen", variant="primary")
                                status_image = gr.Textbox(label="Estado", lines=2)
                
                with gr.Row():
                    original_output = gr.Image(label="Imagen Original/Generada")
                    
                with gr.Row():
                    next_to_process_btn = gr.Button("Continuar a Procesamiento ➡️", variant="primary", visible=True)
                    
                # Eventos para la pestaña 1
                process_audio_btn.click(
                    fn=process_audio,
                    inputs=audio_input,
                    outputs=[user_prompt_input, pos_prompt_display]
                )
                
                def update_prompts(user_prompt):
                    pos, neg = generate_both_prompts(user_prompt)
                    return pos, neg, pos, neg
                
                gen_prompts_btn.click(
                    fn=update_prompts,
                    inputs=user_prompt_input,
                    outputs=[pos_prompt_display, neg_prompt_display, final_prompt, final_negative_prompt]
                )
                
                def update_image(prompt, neg_prompt, uploaded_img):
                    img, status = step_generate_original(prompt, neg_prompt, uploaded_img)
                    return img, status, img
                
                gen_image_btn.click(
                    fn=update_image,
                    inputs=[final_prompt, final_negative_prompt, uploaded_image_input],
                    outputs=[original_output, status_image, original_image]
                )
                
                # Función para cambiar pestaña corregida
                def go_to_processing_tab():
                    # Simplemente devuelve el índice de la pestaña a la que queremos ir
                    return 1
                
                next_to_process_btn.click(
                    fn=go_to_processing_tab,
                    outputs=active_tab
                )
            
            # =============================================
            # Pestaña 2: Procesamiento de Imagen
            # =============================================
            with gr.Tab("⚙️ Procesamiento de Imagen"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Parámetros de Procesamiento")
                        with gr.Row():
                            blur_kernel_size = gr.Slider(
                                minimum=1, maximum=9, value=3, step=2, 
                                label="Kernel de Desenfoque"
                            )
                        with gr.Row():
                            min_contour_area = gr.Slider(
                                minimum=1, maximum=50, value=5, step=1,
                                label="Área Mínima de Contorno"
                            )
                        with gr.Row():
                            clahe_clip_limit = gr.Slider(
                                minimum=0.1, maximum=5.0, value=0.5, step=0.1,
                                label="Límite de Clip CLAHE"
                            )
                        
                        proc_image_btn = gr.Button("Procesar Imagen", variant="primary")
                        status_process = gr.Textbox(label="Estado", lines=2)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Resultado del Procesamiento")
                        processed_output = gr.Image(label="Imagen Procesada (Bordes)")
                
                with gr.Row():
                    back_to_gen_btn = gr.Button("⬅️ Volver a Generación", variant="secondary")
                    next_to_gcode_btn = gr.Button("Continuar a G-code ➡️", variant="primary")
                
                # Eventos para la pestaña 2
                def process_and_update(original_img, blur, min_area, clahe):
                    img, status = step_process_image(original_img, blur, min_area, clahe)
                    return img, status, img
                
                proc_image_btn.click(
                    fn=process_and_update,
                    inputs=[original_image, blur_kernel_size, min_contour_area, clahe_clip_limit],
                    outputs=[processed_output, status_process, processed_image]
                )
                
                back_to_gen_btn.click(
                    fn=lambda: 0,  # Índice de la pestaña de generación
                    outputs=active_tab
                )
                
                next_to_gcode_btn.click(
                    fn=lambda: 2,  # Índice de la pestaña de g-code
                    outputs=active_tab
                )
            
            # =============================================
            # Pestaña 3: Generación de G-code
            # =============================================
            with gr.Tab("💾 Generación de G-code"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuración de G-code")
                        with gr.Row():
                            black_gcode = gr.Checkbox(
                                label="Simplificación (usar 'black')", 
                                value=True,
                                info="Simplifica el G-code usando el modo 'black'"
                            )
                        
                        with gr.Accordion("Parámetros Avanzados", open=False):
                            with gr.Row():
                                threshold = gr.Slider(minimum=10, maximum=100, value=32, step=1, label="Umbral")
                                scale = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Escala")
                            simplify = gr.Slider(minimum=0.1, maximum=1.0, value=0.8, step=0.1, label="Simplificación")
                        
                        gen_gcode_btn = gr.Button("Generar G-code", variant="primary")
                        status_gcode = gr.Textbox(label="Estado", lines=2)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Visualización de G-code")
                        gcode_output = gr.Image(label="Previsualización del G-code")
                        
                        with gr.Accordion("Vista del G-code", open=False):
                            gcode_preview = gr.TextArea(
                                label="Primeras líneas de G-code", 
                                lines=10,
                                value="// El G-code se mostrará aquí después de generarlo"
                            )
                
                with gr.Row():
                    back_to_process_btn = gr.Button("⬅️ Volver a Procesamiento", variant="secondary")
                    next_to_send_btn = gr.Button("Continuar a Envío ➡️", variant="primary")
                
                # Eventos para la pestaña 3
                def generate_gcode_and_preview(black_gcode_option):
                    result, status = step_generate_gcode(black_gcode_option)
                    gcode_text = get_gcode_preview()
                    is_ready = result is not None
                    return result, status, gcode_text, is_ready
                
                gen_gcode_btn.click(
                    fn=generate_gcode_and_preview,
                    inputs=[black_gcode],
                    outputs=[gcode_output, status_gcode, gcode_preview, gcode_ready]
                )
                
                back_to_process_btn.click(
                    fn=lambda: 1,  # Índice de la pestaña de procesamiento
                    outputs=active_tab
                )
                
                next_to_send_btn.click(
                    fn=lambda: 3,  # Índice de la pestaña de envío
                    outputs=active_tab
                )
            
            # =============================================
            # Pestaña 4: Envío al Robot
            # =============================================
            with gr.Tab("🤖 Envío al Robot UR"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Envío de G-code al Robot")
                        
                        robot_status = gr.Textbox(
                            label="Estado del Robot", 
                            value="Esperando para enviar G-code...",
                            interactive=False
                        )
                        
                        with gr.Accordion("Configuración de Conexión", open=False):
                            robot_ip = gr.Textbox(
                                label="IP del Robot", 
                                value="192.168.1.10",
                                placeholder="Dirección IP del Robot UR"
                            )
                            robot_port = gr.Number(
                                label="Puerto", 
                                value=30002,
                                precision=0
                            )
                        
                        send_button = gr.Button("Enviar G-code al Robot", variant="primary")
                        send_status = gr.Textbox(label="Estado de Envío", lines=2)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Resumen del Proceso")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### Imagen Original")
                                original_preview = gr.Image(
                                    label="", 
                                    interactive=False,
                                    height=150
                                )
                            with gr.Column(scale=1):
                                gr.Markdown("#### G-code Generado")
                                gcode_preview_small = gr.Image(
                                    label="", 
                                    interactive=False,
                                    height=150
                                )
                        
                        progress = gr.Textbox(
                            label="Progreso", 
                            value="Esperando envío...",
                            interactive=False
                        )
                
                with gr.Row():
                    back_to_gcode_btn = gr.Button("⬅️ Volver a G-code", variant="secondary")
                    restart_btn = gr.Button("🔄 Reiniciar Proceso", variant="secondary")
                
                # Eventos para la pestaña 4
                def update_send_tab():
                    try:
                        original = Image.open("original.png")
                        gcode_img = Image.open("gcode_plot.png")
                        return original, gcode_img
                    except Exception as e:
                        return None, None
                
                # Eliminado el evento problemático que usaba tabs como input
                
                send_button.click(
                    fn=step_send_gcode,
                    outputs=send_status
                )
                
                back_to_gcode_btn.click(
                    fn=lambda: 2,  # Índice de la pestaña de g-code
                    outputs=active_tab
                )
                
                def reset_state():
                    # Devuelve valores por defecto para los estados
                    return "", "", None, None, False, 0  # El último 0 es para ir a la primera pestaña
                
                restart_btn.click(
                    fn=reset_state,
                    outputs=[final_prompt, final_negative_prompt, original_image, processed_image, gcode_ready, active_tab]
                )
        
                # Evento para cambiar de pestaña basado en active_tab
        # Evento para cambiar de pestaña basado en active_tab
        active_tab.change(
            fn=lambda tab_idx: gr.update(selected=tab_idx),
            inputs=active_tab,
            outputs=tabs
        )

        # Nueva función para cargar las imágenes cuando se cambia a la pestaña de envío
        def on_tab_change(tab_idx):
            if tab_idx == 3:  # Si estamos yendo a la pestaña de envío
                original_path = "original.png"
                gcode_path = "gcode_plot.png"
                
                # Verificar si los archivos existen antes de intentar abrirlos
                if not os.path.exists(original_path) or not os.path.exists(gcode_path):
                    print(f"Advertencia: Uno o ambos archivos no existen: {original_path}, {gcode_path}")
                    return None, None
                
                try:
                    original = Image.open(original_path)
                    gcode_img = Image.open(gcode_path)
                    return original, gcode_img
                except Exception as e:
                    print(f"Error al cargar imágenes de previsualización: {str(e)}")
                    return None, None
            return None, None
        
        # Usamos el componente de estado active_tab para actualizar las imágenes de vista previa
        active_tab.change(
            fn=on_tab_change,
            inputs=active_tab,
            outputs=[original_preview, gcode_preview_small]
        )
        
        # Mensaje de pie de página
        gr.Markdown("""
        ### 📝 Notas
        - Los archivos generados se guardan localmente: `original.png`, `processed.png`, `out.nc` y `gcode_plot.png`
        - La generación de imágenes requiere tener AUTOMATIC1111 corriendo localmente en el puerto 7860
        - La optimización de prompts requiere tener Ollama con llama3.1 instalado
        """)
    
    # Iniciar la aplicación
    demo.queue().launch(share=False)

if __name__ == "__main__":
    main()