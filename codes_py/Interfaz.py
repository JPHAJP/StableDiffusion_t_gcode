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
def generate_image(prompt, negative_prompt="", steps=40, cfg_scale=7, width=812, height=812, sampler_index="Euler", seed=-1, batch_size=1, n_iter=1, send_images=True, save_images=False):
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


def check_gcode_exists():
    """Verifica si el archivo out.nc existe y está listo para ser enviado"""
    if os.path.exists("out.nc"):
        try:
            with open("out.nc", "r") as f:
                lines = [l.strip() for l in f.readlines() if l.startswith("G0") or l.startswith("G1")]
                if len(lines) > 0:
                    return True
        except:
            pass
    return False

def step_send_gcode(robot_ip, robot_port):
    """
    Paso 4: Envía el G-code al robot UR.
    Usa un generador para enviar actualizaciones incrementales a Gradio.
    """
    if not os.path.exists("out.nc"):
        yield "Error: No se encontró el archivo out.nc. Primero genera el G-code."
        return
    
    # Contar el número total de líneas para mostrar el progreso
    try:
        with open("out.nc", "r") as f:
            total_lines = len([l for l in f.readlines() if l.strip() and (l.startswith("G0") or l.startswith("G1"))])
        yield f"Archivo G-code cargado: {total_lines} líneas de código detectadas."
    except Exception as e:
        yield f"Error al leer el archivo G-code: {str(e)}"
        return
    
    progress_messages = []
    current_line = 0
    
    def update_progress(message):
        nonlocal current_line
        progress_messages.append(message)
        
        # Actualizar contador de líneas si el mensaje contiene información de línea
        if "Línea" in message and "/" in message:
            try:
                line_info = message.split("Línea ")[1].split("/")[0]
                current_line = int(line_info.strip())
            except:
                pass
        
        # Crear texto de progreso con contador de líneas y porcentaje
        progress_percent = round((current_line / total_lines) * 100, 1) if total_lines > 0 else 0
        progress_header = f"Progreso: {current_line}/{total_lines} líneas ({progress_percent}%)\n"
        progress_header += "=" * int(50 * progress_percent / 100) + ">" + " " * (50 - int(50 * progress_percent / 100)) + "\n\n"
        
        # Mostrar las últimas líneas de mensajes
        progress_text = progress_header + "\n".join(progress_messages[-15:])
        return progress_text
    
    try:
        yield "Inicializando conexión con el robot..."
        
        # Define una función de callback para recibir actualizaciones de progreso
        def progress_callback(msg):
            return update_progress(msg)
        
        # Importar dinámicamente para evitar problemas de scope
        from gcode_t_ur import NCtoURConverter
        
        # Crear instancia con la IP y callback proporcionados
        converter = NCtoURConverter(robot_ip=robot_ip, progress_callback=progress_callback)
        
        yield update_progress("Intentando conectar con el robot...")
        
        # Inicializar robot
        if not converter.initialize_robot():
            yield update_progress("Error: No se pudo conectar con el robot. Verifica la IP y que el robot esté encendido.")
            return
        
        yield update_progress("Conexión establecida. Moviendo a posición inicial...")
        
        # Mover a posición inicial
        converter.go_home()
        yield update_progress("Robot en posición inicial. Procesando G-code...")
        
        # Procesar G-code
        if not converter.process_nc_file(file_path="out.nc"):
            yield update_progress("Error: No se pudo procesar el archivo G-code.")
            return
        
        yield update_progress("G-code procesado. Volviendo a posición inicial...")
        
        # Volver a posición inicial
        converter.go_home()
        
        yield update_progress("¡Proceso completado con éxito! El robot ha ejecutado el G-code.")
    except Exception as e:
        yield update_progress(f"Error durante la ejecución: {str(e)}")

        
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
        # Variables de estado
        final_prompt = gr.State("")
        final_negative_prompt = gr.State("")
        original_image = gr.State(None)
        processed_image = gr.State(None)
        gcode_ready = gr.State(False)
        
        gr.Markdown("# 🤖 Sistema de Generación y Envío de G-code a UR")
        gr.Markdown("### Genera imágenes, conviértelas a G-code y envíalas al robot UR")
        
        with gr.Tabs():
            # =============================================
            # Pestaña 1: Generación de Imagen
            # =============================================
            with gr.TabItem("📷 Generación de Imagen"):
                # --- (Contenido de la pestaña de generación de imagen) ---
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
                    
                gr.Markdown("*Para continuar, selecciona la pestaña '⚙️ Procesamiento de Imagen' en la parte superior.*")
                
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
            
            # =============================================
            # Pestaña 2: Procesamiento de Imagen
            # =============================================
            with gr.TabItem("⚙️ Procesamiento de Imagen"):
                # --- (Contenido de la pestaña de procesamiento) ---
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Parámetros de Procesamiento")
                        with gr.Row():
                            blur_kernel_size = gr.Slider(minimum=1, maximum=20, value=3, step=2, label="Kernel de Desenfoque")
                        with gr.Row():
                            min_contour_area = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Área Mínima de Contorno")
                        with gr.Row():
                            clahe_clip_limit = gr.Slider(minimum=0.01, maximum=10.0, value=0.5, step=0.01, label="Límite de Clip CLAHE")
                        
                        proc_image_btn = gr.Button("Procesar Imagen", variant="primary")
                        status_process = gr.Textbox(label="Estado", lines=2)
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ Resultado del Procesamiento")
                        processed_output = gr.Image(label="Imagen Procesada (Bordes)")
                
                with gr.Row():
                    gr.Markdown("*Para volver a 'Generación de Imagen' o continuar a 'Generación de G-code', utiliza las pestañas de arriba.*")
                
                def process_and_update(original_img, blur, min_area, clahe):
                    img, status = step_process_image(original_img, blur, min_area, clahe)
                    return img, status, img
                
                proc_image_btn.click(
                    fn=process_and_update,
                    inputs=[original_image, blur_kernel_size, min_contour_area, clahe_clip_limit],
                    outputs=[processed_output, status_process, processed_image]
                )
            
            # =============================================
            # Pestaña 3: Generación de G-code
            # =============================================
            with gr.TabItem("💾 Generación de G-code"):
                # --- (Contenido de la pestaña de G-code) ---
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
                    gr.Markdown("*Para volver a 'Procesamiento de Imagen' o continuar a 'Envío al Robot', usa las pestañas de arriba.*")
                
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
            
            # =============================================
            # Pestaña 4: Envío al Robot UR
            # =============================================
            # Agrega esta función junto a las otras funciones de procesamiento, por ejemplo, después de step_send_gcode
            def load_images():
                try:
                    original = Image.open("original.png")
                except Exception as e:
                    original = None
                try:
                    gcode_img = Image.open("gcode_plot.png")
                except Exception as e:
                    gcode_img = None
                return original, gcode_img

            # ...

            # En la pestaña 4: Envío al Robot UR
            with gr.TabItem("🤖 Envío al Robot UR"):
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
                                value="192.168.1.1",
                                placeholder="Dirección IP del Robot UR"
                            )
                            robot_port = gr.Number(
                                label="Puerto", 
                                value=30002,
                                precision=0
                            )
                        send_button = gr.Button("Enviar G-code al Robot", variant="primary")
                        # Cambiamos Text por Textbox con más líneas para ver el progreso
                        progress_log = gr.Textbox(
                            label="Progreso de la Ejecución", 
                            lines=10,  # Más líneas para ver una historia de progreso
                            value="El progreso se mostrará aquí...",
                            interactive=False
                        )
                        
                        # Usamos el modo 'queue' para actualizaciones incrementales
                        send_button.click(
                            fn=step_send_gcode,
                            inputs=[robot_ip, robot_port],
                            outputs=[progress_log]
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Resumen del Proceso")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### Imagen Original")
                                original_preview = gr.Image(label="", interactive=False, height=150)
                            with gr.Column(scale=1):
                                gr.Markdown("#### G-code Generado")
                                gcode_preview_small = gr.Image(label="", interactive=False, height=150)
                        
                        # Se agrega el botón para cargar imágenes
                        load_images_btn = gr.Button("Cargar Imágenes", variant="secondary")
                        progress = gr.Textbox(label="Progreso", value="Esperando envío...", interactive=False)
                        
                        # Evento para el botón: actualiza las imágenes en el resumen
                        load_images_btn.click(
                            fn=load_images,
                            outputs=[original_preview, gcode_preview_small]
                        )

            
        
        gr.Markdown("""
        ### 📝 Notas
        - Los archivos generados se guardan localmente: `original.png`, `processed.png`, `out.nc` y `gcode_plot.png`
        - La generación de imágenes requiere tener AUTOMATIC1111 corriendo localmente en el puerto 7860
        - La optimización de prompts requiere tener Ollama con llama3.1 instalado
        """)
    
    demo.queue().launch(share=False)

if __name__ == "__main__":
    main()