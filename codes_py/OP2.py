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
import threading
import queue
import socket
import qrcode
import subprocess  # Añadido para ejecutar el comando externo


from request_process_img import process_image  
# Eliminamos la importación de img_t_gcode2 ya que no la usaremos
# from img_t_gcode2 import convert_image_to_gcode
from gcode_t_img import plot_gcode
from gcode_t_ur import NCtoURConverter


def get_server_ip():
    """
    Obtiene la IP local del servidor para conexiones LAN.
    Se conecta a un servidor público (8.8.8.8) para determinar la IP de salida.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def generate_qr_code(url):
    """
    Genera un código QR para la URL proporcionada y lo devuelve como una imagen PIL.
    Convierte la imagen a un formato compatible con Gradio.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    # Crea la imagen QR
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convertir a formato PIL.Image.Image que Gradio puede manejar
    if not isinstance(img, Image.Image):
        # Si es un objeto PilImage especial de qrcode, convertirlo a PIL.Image
        img = img.get_image()
    
    # Guardar temporalmente la imagen y cargarla para asegurar compatibilidad
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    pil_img = Image.open(buffer)
    
    return pil_img

def update_qr_with_url(url):
    """Actualiza el código QR con la URL proporcionada"""
    if url and url.startswith("http"):
        qr_img = generate_qr_code(url)
        return qr_img, f"QR generado para: {url}"
    return None, "Ingresa una URL válida que comience con http:// o https://"

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
    3. Keep the description very simple.
    4. Final format: Just the prompt with no additional explanation.
    5. Use no more than 25 words.
    6. Do not mention that you are generating a prompt.
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
    2. Focus on describing what to avoid: low quality, blurry, distorted, watermark, etc.
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

# FUNCIÓN MODIFICADA: Reemplazamos la llamada a la función por la ejecución del comando externo
def step_generate_gcode(use_simplify, scale_value):
    """
    Paso 3: Genera el G-code a partir de la imagen procesada usando un comando externo
    y lo visualiza.
    """
    if not os.path.exists("processed.png"):
        return None, "No se encontró la imagen procesada. Primero procesa una imagen."
    
    # Construir el comando con los parámetros
    cmd = ["python", "codes_py/svgtry.py", "processed.png"]
    
    # Añadir bandera --simplify si está habilitada
    if use_simplify:
        cmd.append("--simplify")
    
    # Añadir el parámetro de escala
    cmd.extend(["--scale", str(scale_value), "-o", "out.nc"])
    
    try:
        # Ejecutar el comando
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Verificar si el comando se ejecutó correctamente
        if result.returncode != 0:
            return None, f"Error al generar G-code: {result.stderr}"
        
        # Verificar si se generó el archivo de salida
        if not os.path.exists("out.nc"):
            return None, "El script no generó el archivo out.nc."
        
        # Verificar condiciones del G-code generado
        if not check_gcode_conditions("out.nc", max_segment=700, max_lines=10000):
            return None, "El G-code generado no cumple con las restricciones."
            
        # Visualizar el G-code
        plot_gcode("out.nc", "gcode_plot.png")
        gcode_plot = Image.open("gcode_plot.png")
        
        # Contar líneas en el G-code
        with open("out.nc", "r") as f:
            lines = [l.strip() for l in f.readlines() if l.startswith("G0") or l.startswith("G1")]
            num_lines = len(lines)
        
        return gcode_plot, f"G-code generado exitosamente. Líneas: {num_lines}/10000"
    
    except subprocess.CalledProcessError as e:
        return None, f"Error al ejecutar el script: {e.stderr}"
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
            gcode_lines = [l for l in f.readlines() if l.strip() and (l.startswith("G0") or l.startswith("G1"))]
            total_lines = len(gcode_lines)
        yield f"Archivo G-code cargado: {total_lines} líneas de código detectadas."
    except Exception as e:
        yield f"Error al leer el archivo G-code: {str(e)}"
        return
    
    # Cola de mensajes para comunicación entre hilos
    message_queue = queue.Queue()
    
    # Variable para seguimiento de progreso
    progress_data = {
        "current_line": 0,
        "total_lines": total_lines,
        "messages": [],
        "status": "iniciando"
    }
    
    # Función de callback para recibir actualizaciones desde el robot
    def progress_callback(message):
        message_queue.put(message)
        return message  # Necesario retornar para la clase NCtoURConverter
    
    # Función para el hilo de ejecución del robot
    def robot_execution_thread():
        try:
            converter = NCtoURConverter(robot_ip=robot_ip, progress_callback=progress_callback)
            
            # Poner mensaje en la cola
            message_queue.put("Intentando conectar con el robot...")
            
            # Inicializar robot
            if not converter.initialize_robot():
                message_queue.put("ERROR: No se pudo conectar con el robot. Verifica la IP y que el robot esté encendido.")
                progress_data["status"] = "error"
                return
            
            message_queue.put("Conexión establecida. Moviendo a posición inicial...")
            
            # Mover a posición inicial
            converter.go_home()
            message_queue.put("Robot en posición inicial. Procesando G-code...")
            
            # Procesar G-code
            if not converter.process_nc_file(file_path="out.nc"):
                message_queue.put("ERROR: No se pudo procesar el archivo G-code.")
                progress_data["status"] = "error"
                return
            
            message_queue.put("G-code procesado. Volviendo a posición inicial...")
            
            # Volver a posición inicial
            converter.go_home()
            
            message_queue.put("¡Proceso completado con éxito! El robot ha ejecutado el G-code.")
            progress_data["status"] = "completado"
            
        except Exception as e:
            message_queue.put(f"ERROR durante la ejecución: {str(e)}")
            progress_data["status"] = "error"
    
    # Iniciar el hilo de ejecución
    thread = threading.Thread(target=robot_execution_thread)
    thread.daemon = True
    thread.start()
    
    # Bucle principal para enviar actualizaciones a Gradio en tiempo real
    last_update_time = time.time()
    
    # Generamos una primera actualización
    yield "Iniciando proceso de ejecución de G-code..."
    
    # Bucle para capturar mensajes y actualizar la interfaz
    while thread.is_alive() or not message_queue.empty():
        # Esperar un poco para no saturar la interfaz
        time.sleep(0.1)
        
        new_messages = []
        # Recoger todos los mensajes disponibles desde la última vez
        while not message_queue.empty():
            msg = message_queue.get()
            new_messages.append(msg)
            progress_data["messages"].append(msg)
            
            # Actualizar el contador de líneas si el mensaje contiene información de línea
            if "Línea" in msg and "/" in msg:
                try:
                    line_info = msg.split("Línea ")[1].split("/")[0]
                    progress_data["current_line"] = int(line_info.strip())
                except:
                    pass
        
        # Si hay nuevos mensajes o ha pasado suficiente tiempo, actualizar la interfaz
        current_time = time.time()
        if new_messages or (current_time - last_update_time) > 1:
            # Crear texto de progreso con contador de líneas y porcentaje
            current = progress_data["current_line"]
            total = progress_data["total_lines"]
            progress_percent = round((current / total) * 100, 1) if total > 0 else 0
            
            # Cabecera con barra de progreso
            progress_header = f"Progreso: {current}/{total} líneas ({progress_percent}%)\n"
            progress_bar = "=" * int(50 * progress_percent / 100) + ">" + " " * (50 - int(50 * progress_percent / 100))
            progress_header += progress_bar + "\n\n"
            
            # Mostrar las últimas líneas de mensajes (limitado a 15 para no saturar)
            messages_to_show = progress_data["messages"][-15:] if len(progress_data["messages"]) > 15 else progress_data["messages"]
            progress_text = progress_header + "\n".join(messages_to_show)
            
            # Actualizar el tiempo de la última actualización
            last_update_time = current_time
            
            # Yield para actualizar la interfaz de Gradio
            yield progress_text
    
    # Mensaje final una vez que el thread ha terminado
    if progress_data["status"] == "error":
        yield "❌ Error en la ejecución del G-code. Revisa los mensajes anteriores para más detalles."
    else:
        yield "✅ Ejecución de G-code completada correctamente."


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
    # Obtiene la IP del servidor para mostrar información de red local
    server_ip = get_server_ip()
    server_port = 7861  # Puerto predeterminado de Gradio
    local_url = f"http://{server_ip}:{server_port}"
    
    # Inicia la interfaz de Gradio
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
            # Pestaña 3: Generación de G-code - MODIFICADA
            # =============================================
            with gr.TabItem("💾 Generación de G-code"):
                # --- (Contenido de la pestaña de G-code) ---
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuración de G-code")
                        with gr.Row():
                            use_simplify = gr.Checkbox(
                                label="Usar simplificación", 
                                value=True,
                                info="Añade la bandera --simplify al comando"
                            )
                        with gr.Row():
                            scale_value = gr.Slider(
                                minimum=0.01, 
                                maximum=1.0, 
                                value=0.1, 
                                step=0.01, 
                                label="Escala"
                            )
                        
                        gen_gcode_btn = gr.Button("Generar G-code", variant="primary")
                        status_gcode = gr.Textbox(label="Estado", lines=2)
                        
                        # Muestra el comando que se ejecutará
                        command_preview = gr.Textbox(
                            label="Comando a ejecutar", 
                            value="El comando se mostrará aquí",
                            interactive=False
                        )
                    
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
                
                # Función para actualizar la vista previa del comando
                def update_command_preview(use_simplify, scale):
                    cmd = f"python codes_py/svgtry.py processed.png"
                    if use_simplify:
                        cmd += " --simplify"
                    cmd += f" --scale {scale} -o out.nc"
                    return cmd
                
                # Evento para actualizar la vista previa del comando
                use_simplify.change(
                    fn=update_command_preview,
                    inputs=[use_simplify, scale_value],
                    outputs=command_preview
                )
                
                scale_value.change(
                    fn=update_command_preview,
                    inputs=[use_simplify, scale_value],
                    outputs=command_preview
                )
                
                # Función para generar G-code con los nuevos parámetros
                def generate_gcode_and_preview(use_simplify_option, scale):
                    # Actualizar la vista previa del comando
                    cmd_preview = update_command_preview(use_simplify_option, scale)
                    
                    # Generar el G-code
                    result, status = step_generate_gcode(use_simplify_option, scale)
                    
                    # Obtener vista previa del G-code generado
                    gcode_text = get_gcode_preview()
                    
                    # Verificar si se generó correctamente
                    is_ready = result is not None
                    
                    return result, status, gcode_text, is_ready, cmd_preview
                
                gen_gcode_btn.click(
                    fn=generate_gcode_and_preview,
                    inputs=[use_simplify, scale_value],
                    outputs=[gcode_output, status_gcode, gcode_preview, gcode_ready, command_preview]
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
                        with gr.Accordion("Configuración de Conexión", open=True):
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
                        
                        # Verificar si el G-code está listo antes de enviar
                        def check_and_send(ip, port):
                            if check_gcode_exists():
                                return step_send_gcode(ip, port)
                            else:
                                return "Error: No hay G-code generado. Por favor, genera primero el G-code en la pestaña anterior."
                        
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
                            fn=check_and_send,
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
            
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""
                ### 📝 Notas
                - Los archivos generados se guardan localmente: `original.png`, `processed.png`, `out.nc` y `gcode_plot.png`
                - La generación de imágenes requiere tener AUTOMATIC1111 corriendo localmente en el puerto 7860
                - La optimización de prompts requiere tener Ollama con llama3.1 instalado
                - El script de generación de G-code utiliza `codes_py/svgtry.py` para convertir la imagen procesada
                """)
            with gr.Column(scale=2):
                gr.Markdown(f"### 📱 Acceso remoto")
                gr.Markdown(f"URL local: {local_url}")
                
                # Entrada para la URL de Gradio compartida
                gr.Markdown("**Cuando Gradio muestre la URL pública, cópiala aquí:**")
                gradio_url = gr.Textbox(
                    label="URL compartida de Gradio",
                    placeholder="https://xxx-xxx-xxx.gradio.live",
                    interactive=True
                )
                generate_qr_btn = gr.Button("Generar código QR")
                qr_image = gr.Image(label="Escanea para acceder remotamente")
                status_text = gr.Textbox(label="Estado", interactive=False)
                
                # Evento para generar el QR
                generate_qr_btn.click(
                    fn=update_qr_with_url,
                    inputs=[gradio_url],
                    outputs=[qr_image, status_text]
                )
    
    # Iniciar Gradio con interfaz compartida
    demo.queue().launch(share=False, server_port=7861)

if __name__ == "__main__":
    main()