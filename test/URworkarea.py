import numpy as np
import time

import rtde_control
import rtde_receive
import rtde_io

def inicializar_robot():
    try:
        ip = "192.168.1.1"
        control = rtde_control.RTDEControlInterface(ip)
        receive = rtde_receive.RTDEReceiveInterface(ip)
        io = rtde_io.RTDEIOInterface(ip)
        return control, receive, io
    except Exception as e:
        print("Error al inicializar el robot.")
        print(e)
        time.sleep(1)
        return None, None, None

def gohome(control):
    # Función para mover el robot a la posición "Home"
    # Coordenadas articulares para el home
    home_joint_angles_deg = [-51.9, -71.85, -112.7, -85.96, 90, 38]
    # Convertir la lista de ángulos a radianes
    home_joint_angles_rad = np.radians(home_joint_angles_deg)
    # Mover el robot a la posición "Home" usando control.moveJ
    # Velocidad = 1 rad/s, Aceleración = 1 rad/s^2
    control.moveJ(home_joint_angles_rad, 1, 1)

def move_robot(xtransfn, ytransfn, ofzn, control, receive):
    def is_point_within_reach(point):
        """
        Verifica si un punto está dentro del alcance del UR5-e.
        
        Parámetro:
        point (tuple): Una tupla con las coordenadas (x, y, z) del punto en milímetros.
        
        Retorna:
        bool: True si el punto está dentro del alcance, False de lo contrario.
        """
        # Calcula la distancia euclidiana desde la base (0, 0, 0) al punto
        distance = np.linalg.norm(point)
    
        # Verifica si la distancia está dentro del rango máximo
        UR5E_MAX_REACH = .85
        UR5E_MIN_REACH = .30
        return distance <= UR5E_MAX_REACH and distance >= UR5E_MIN_REACH
    is_point_on_work=is_point_within_reach([xtransfn, ytransfn, ofzn])
    if not is_point_on_work:
        print("Punto fuera de alcance")
        return
    xr, yr, zr, rxr, ryr, rzr = receive.getActualTCPPose()
    #destinationf = [xtransfn, ytransfn, ofzn]
    control.moveL([xtransfn, ytransfn, ofzn, rxr, ryr, rzr], .5, .5, asynchronous=True)
    # Normalizamos para poder hacer comparativas
    #destinationf = np.around(destinationf, decimals=2)
    return

def main():
    # Inicializar el robot
    control = None
    while control is None:
        control, receive, io = inicializar_robot()
    
    # Mover el robot a la posición "Home"
    gohome(control)
    time.sleep(2)

    while True:
        # Input de coordenadas x, y, z
        xtransfn = float(input("Ingrese la coordenada x: "))
        ytransfn = float(input("Ingrese la coordenada y: "))
        ofzn = float(input("Ingrese la coordenada z: "))
        # Convertir las coordenadas de mm a metros
        xtransfn = xtransfn/1000
        ytransfn = ytransfn/1000
        ofzn = ofzn/1000
        # Mover el robot a la posición deseada
        move_robot(xtransfn, ytransfn, ofzn, control, receive)
if __name__ == "__main__":
    main()



#TODO Set the pad with the blend radius (DOCUMENTATION EXAMPLE)
    # velocity = 0.5
    # acceleration = 0.5
    # blend_1 = 0.0
    # blend_2 = 0.02
    # blend_3 = 0.0
    # path_pose1 = [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04, velocity, acceleration, blend_1]
    # path_pose2 = [-0.143, -0.51, 0.21, -0.001, 3.12, 0.04, velocity, acceleration, blend_2]
    # path_pose3 = [-0.32, -0.61, 0.31, -0.001, 3.12, 0.04, velocity, acceleration, blend_3]
    # path = [path_pose1, path_pose2, path_pose3]

    # # Send a linear path with blending in between - (currently uses separate script)
    # control.moveL(path)
    # control.stopScript()

## TODO Check the blend radius function for dinamic blend radius
# def blenradius(corner_radius, tool_radius):
#     """
#     Calculate the blend radius for a given corner radius and tool radius.
#     This function calculates the blend radius for a given corner radius and a given tool radius.
#     The blend radius is the radius of the circle that connects the corner and the tool radius.
#     The blend radius should be less than or equal to the corner radius.
#     The blend radius should be less than or equal to the tool radius.
#     The blend radius should be greater than or equal to zero.
    
#     Parameters:
#     corner_radius (float): The corner radius in millimeters.
#     tool_radius (float): The tool radius in millimeters.
    
#     Returns:
#     float: The blend radius in millimeters.
#     """
#     # Calculate the blend radius
#     blend_radius = min(corner_radius, tool_radius)
#     return blend_radius