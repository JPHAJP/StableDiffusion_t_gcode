import re
import matplotlib.pyplot as plt

def parse_gcode_line(line, current_pos):
    """
    Analiza una línea del G-code.
    Si la línea comienza con G0 o G1, busca los parámetros X e Y.
    Retorna la instrucción ('G0' o 'G1') y la nueva posición.
    Si no se especifica alguna coordenada, se mantiene la actual.
    """
    cmd = None
    # Detectar el tipo de movimiento
    if line.startswith("G0") or line.startswith("G00"):
        cmd = "G0"
    elif line.startswith("G1") or line.startswith("G01"):
        cmd = "G1"
    
    if cmd:
        # Buscar coordenadas X y Y usando expresiones regulares
        x_match = re.search(r'X([-+]?[0-9]*\.?[0-9]+)', line)
        y_match = re.search(r'Y([-+]?[0-9]*\.?[0-9]+)', line)
        x = float(x_match.group(1)) if x_match else current_pos[0]
        y = float(y_match.group(1)) if y_match else current_pos[1]
        return cmd, (x, y)
    return None, current_pos

def plot_gcode(gcode_file, output_png):
    """
    Lee un archivo G-code y genera un gráfico PNG.
    Se dibujan:
      - Movimientos G0 (saltos) en amarillo.
      - Movimientos G1 (trabajo) en azul.
    """
    current_pos = (0, 0)
    segments_g0 = []  # Lista de segmentos para G0
    segments_g1 = []  # Lista de segmentos para G1

    # Leer el archivo de G-code
    with open(gcode_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):  # Ignorar líneas vacías o comentarios
                continue
            cmd, new_pos = parse_gcode_line(line, current_pos)
            if cmd:
                # Almacenar el segmento desde la posición actual hasta la nueva
                if cmd == "G0":
                    segments_g0.append((current_pos, new_pos))
                elif cmd == "G1":
                    segments_g1.append((current_pos, new_pos))
                current_pos = new_pos

    # Crear la figura
    plt.figure(figsize=(8, 8))
    
    # Dibujar segmentos G0 en amarillo
    for seg in segments_g0:
        x_values = [seg[0][0], seg[1][0]]
        y_values = [seg[0][1], seg[1][1]]
        plt.plot(x_values, y_values, color="yellow", linewidth=2, label="G0")
    
    # Dibujar segmentos G1 en azul
    for seg in segments_g1:
        x_values = [seg[0][0], seg[1][0]]
        y_values = [seg[0][1], seg[1][1]]
        plt.plot(x_values, y_values, color="blue", linewidth=2, label="G1")
    
    # Evitar leyendas duplicadas
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("G-code Plotter")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Gráfico guardado en: {output_png}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python gcode_plotter.py archivo_gcode.txt salida.png")
        sys.exit(1)
    gcode_file = sys.argv[1]
    output_png = sys.argv[2]
    plot_gcode(gcode_file, output_png)
