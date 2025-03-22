import gradio as gr
from PIL import Image
import cv2
import time

from request_process_img import generate_image, load_image, process_image
from img_t_gcode2 import EdgesToGcode
from gcode_t_img import plot_gcode
from gcode_t_ur import main as send_gcode_to_ur
from img_t_gcode2 import convert_image_to_gcode


def process_input(prompt, image_file, progress=gr.Progress(track_tqdm=False)):
    """
    Función generadora que:
    1) Muestra barra de progreso en la interfaz.
    2) Hace yields parciales para actualizar las imágenes en cada paso.
    """
    # Inicia la barra de progreso en 0%
    progress(0, desc="Iniciando proceso...")
    #guarda el tiempo de inicio
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
    
    # Actualiza interfaz con la imagen original
    yield (original, None, None)
    progress(0.3, desc="Imagen cargada/generada")

    # Paso 2: procesar la imagen (detección de bordes)
    processed = process_image(original, output_path="processed.png")
    
    # Muestra imagen procesada
    yield (original, processed, None)
    progress(0.6, desc="Imagen procesada (bordes)")

    # Paso 3: convertir a binario y generar G-code
    # Aquí llamas a la nueva función, indicando -e black (edges_mode="black")
    convert_image_to_gcode(
        image_input="processed.png",   # o la PIL.Image si quieres
        output_gcode="out.nc",
        edges_mode="black",            # equivale a --edges black
        threshold=32,                  # equivale a -t 32
        scale=1.0,                     # equivale a -s 1.0
        simplify=0.5,                  # equivale a --simplify 0.5
        dot_output=None                # si quieres un .dot, pon algo como "graph.dot"
    )

    # De este modo, se generará "out.nc" usando la misma lógica que antes.
    # Luego, si quieres visualizar el G-code:
    plot_gcode("out.nc", "gcode_plot.png")
    gcode_plot = Image.open("gcode_plot.png")

    # Muestra todo listo
    yield (original, processed, gcode_plot)
    progress(1.0, desc="Completado")

    # Calcula tiempo total y muestra
    total_time = time.time() - start_time
    print(f"Proceso completado en {total_time:.2f} segundos.")

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
                process_button = gr.Button("Procesar")
            
            with gr.Column():
                original_output = gr.Image(label="Imagen original / generada")
                processed_output = gr.Image(label="Imagen procesada (bordes)")
                gcode_output = gr.Image(label="Visualización del G-code")
        
        # Usa queue=True para que se muestren los yields y la barra de progreso
        process_button.click(
            fn=process_input,
            inputs=[prompt_input, image_input],
            outputs=[original_output, processed_output, gcode_output],
            queue=True
        )

        gr.Markdown("### Enviar G-code al robot UR")
        send_button = gr.Button("Enviar G-code")
        send_status = gr.Textbox(label="Estado de envío")
        send_button.click(fn=send_gcode, inputs=[], outputs=send_status)

        demo.queue().launch()


if __name__ == "__main__":
    main()