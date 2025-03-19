import re
import numpy as np
import rtde_control
import rtde_receive

# Inicialización del robot
def inicializar_robot():
    try:
        ip = "192.168.1.1"
        control = rtde_control.RTDEControlInterface(ip)
        receive = rtde_receive.RTDEReceiveInterface(ip)
        return control, receive
    except Exception as e:
        print(f"Error al inicializar el robot: {e}")
        return None, None

# Función para parsear un código NC
def parse_nc_code(nc_code):
    """
    Parsea un código NC para extraer las coordenadas X, Y.
    Este ejemplo asume que las coordenadas están en el formato G1 X... Y...
    
    Parámetro:
    nc_code (str): Línea de código NC.
    
    Retorna:
    tuple: (X, Y) o None si no se encuentran coordenadas.
    """
    match = re.search(r'G1.*X([-\d.]+).*Y([-\d.]+)', nc_code)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return (x, y)
    return None

# Función para mover el robot a una posición específica
def move_robot_to_position(x, y, control, receive, z_offset=-0.05):
    """
    Mueve el robot a la posición (x, y) en el plano XY, con un offset negativo en Z.
    La herramienta sigue el plano XY_BASE del robot.
    
    Parámetros:
    x (float): Coordenada X en milímetros.
    y (float): Coordenada Y en milímetros.
    control (object): Interfaz de control RTDE.
    receive (object): Interfaz de recepción RTDE.
    z_offset (float): Desplazamiento negativo en Z (por defecto -0.05m).
    """
    # Establecer Z con el offset negativo
    z = z_offset
    
    # Verificar si la posición está dentro del alcance del robot
    def is_point_within_reach(point):
        distance = np.linalg.norm(point)
        UR5E_MAX_REACH = 0.9  # Alcance máximo del UR5
        return distance <= UR5E_MAX_REACH and distance >= 0.3

    if not is_point_within_reach([x, y, z]):
        print("Punto fuera de alcance")
        return

    # Obtener la orientación actual del TCP
    xr, yr, zr, rxr, ryr, rzr = receive.getActualTCPPose()

    # Mover el robot usando el movimiento lineal (con el offset negativo en Z)
    control.moveL([x, y, z, rxr, ryr, rzr], 0.5, 0.5)

# Función principal para leer el código NC y mover el robot
def main():
    control, receive = inicializar_robot()
    if control is None:
        return
    
    # Ejemplo de código NC
    nc_code = [
        "G1 X-0.1 Y-0.2",
        "G1 X-0.15 Y-0.15",
        "G1 X-0.2 Y-0.1"
    ]
    
    for line in nc_code:
        coordinates = parse_nc_code(line)
        if coordinates:
            x, y = coordinates
            print(f"Moviendo a: X={x}, Y={y}, Z con offset aplicado.")
            move_robot_to_position(x, y, control, receive)

if __name__ == "__main__":
    main()
