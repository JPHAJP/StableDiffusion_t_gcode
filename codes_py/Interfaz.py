import gradio as gr
from PIL import Image
import cv2
import time
import math

from request_process_img import generate_image, load_image, process_image
from img_t_gcode2 import EdgesToGcode, convert_image_to_gcode
from gcode_t_img import plot_gcode
from gcode_t_ur import main as send_gcode_to_ur

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

def process_input(prompt, image_file, blur_kernel_size, min_contour_area, clahe_clip_limit, combine_with_original, progress=gr.Progress(track_tqdm=False)):
    """
    Función generadora que:
    1) Muestra barra de progreso y un mensaje de estado.
    2) Carga o genera la imagen, la procesa para detectar bordes con los parámetros ingresados y genera el G-code.
    
    Si el G-code no cumple las restricciones, se muestra una advertencia.
    """
    progress(0, desc="Iniciando proceso...")
    start_time = time.time()

    # Paso 1: cargar o generar la imagen
    if image_file is not None:
        original = image_file
    elif prompt and prompt.strip():
        original = generate_image(prompt)
    else:
        raise ValueError("Debes ingresar un prompt o subir una imagen para continuar.")

    # Redimensionar para no exceder 500x500 y guardar
    original.thumbnail((500, 500))
    original.save("original.png")
    
    yield (original, None, None, "Imagen cargada/generada")
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
    yield (original, processed, None, "Imagen procesada para detección de bordes")
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
        warning = "El G-code generado no cumple con las restricciones: verifique el número de líneas o el tamaño de segmento."
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
        final_status = (f"✅ G-code generado exitosamente en {total_time:.2f} segundos.\n"
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

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Interfaz para generar/procesar imágenes y enviar G-code a UR")

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt para generar imagen",
                    placeholder="Ingresa un prompt aquí..."
                )
                image_input = gr.Image(
                    label="O carga una imagen",
                    type="pil"
                )
                blur_kernel_size_input = gr.Number(label="Tamaño del kernel de desenfoque", value=3)
                min_contour_area_input = gr.Number(label="Área mínima del contorno", value=5)
                clahe_clip_limit_input = gr.Number(label="Límite de clip para CLAHE", value=1.5)
                combine_with_original_input = gr.Checkbox(label="Combinar con original", value=True)
                process_button = gr.Button("Procesar")
            
            with gr.Column():
                original_output = gr.Image(label="Imagen original / generada")
                processed_output = gr.Image(label="Imagen procesada (bordes)")
                gcode_output = gr.Image(label="Visualización del G-code")
                status_output = gr.Textbox(label="Estado del proceso")
        
        process_button.click(
            fn=process_input,
            inputs=[
                prompt_input, image_input,
                blur_kernel_size_input, min_contour_area_input, 
                clahe_clip_limit_input, combine_with_original_input
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
