import numpy as np
import time
import re
import rtde_control
import rtde_receive
import rtde_io

class NCtoURConverter:
    def __init__(self, robot_ip="192.168.1.1", progress_callback=None):
        self.ip = robot_ip
        self.progress_callback = progress_callback  # Función de callback para el progreso
        self.control = None
        self.receive = None
        self.io = None
        self.drawing_plane_z = 0.002  # 2mm drawing plane
        self.line_offset_z = 0.01  # 10mm offset between lines
        self.blend_radius = 0.005  # 5mm blend radius para movimientos continuos
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.initial_rx = None
        self.initial_ry = None
        self.initial_rz = None
        self.move_speed = 0.5
        self.move_accel = 0.5
        # Add X and Y offset for positioning
        self.x_offset = -0.1
        self.y_offset = -0.2
        
    def initialize_robot(self):
        """Initialize connection to the UR robot"""
        try:
            self.control = rtde_control.RTDEControlInterface(self.ip)
            self.receive = rtde_receive.RTDEReceiveInterface(self.ip)
            self.io = rtde_io.RTDEIOInterface(self.ip)
            # Store initial TCP rotation
            tcp_pose = self.receive.getActualTCPPose()
            self.initial_rx = tcp_pose[3]
            self.initial_ry = tcp_pose[4]
            self.initial_rz = tcp_pose[5]
            return True
        except Exception as e:
            print(f"Error initializing robot: {e}")
            return False
    
    def go_home(self):
        """Move robot to home position"""
        # Home joint angles in degrees
        home_joint_angles_deg = [-51.9, -71.85, -112.7, -85.96, 90, 38]
        # Convert to radians
        home_joint_angles_rad = np.radians(home_joint_angles_deg)
        # Move robot to home position
        self.control.moveJ(home_joint_angles_rad, 1, 1)
        time.sleep(1)  # Add sleep to ensure movement completes
        # Update current position after moving home
        tcp_pose = self.receive.getActualTCPPose()
        self.current_x = tcp_pose[0]
        self.current_y = tcp_pose[1]
        self.current_z = tcp_pose[2]

    def is_point_within_reach(self, x, y, z):
        """Check if the point is within the robot's workspace"""
        # Point is already in meters
        point = np.array([x, y, z])
        
        # Calculate distance from base to point
        distance = np.linalg.norm(point)
        
        # UR5e workspace limits (in meters)
        UR5E_MAX_REACH = 0.85
        UR5E_MIN_REACH = 0.18
        
        return UR5E_MIN_REACH <= distance <= UR5E_MAX_REACH
    
    def move_robot(self, x, y, z, is_rapid=False):
        """Move the robot to a specified position while maintaining TCP orientation"""
        # Apply offset to x and y coordinates
        x = x + self.x_offset
        y = -y + self.y_offset
        
        # Handle cases where x or y is zero (maintain previous value)
        if x == self.x_offset and self.current_x is not None:
            x = self.current_x
        if y == self.y_offset and self.current_y is not None:
            y = self.current_y
            
        # Check if point is within workspace
        if not self.is_point_within_reach(x, y, z):
            print(f"Warning: Point ({x:.3f}, {y:.3f}, {z:.3f}) is outside workspace!")
            return False
        
        # Create pose with initial TCP orientation
        target_pose = [x, y, z, self.initial_rx, self.initial_ry, self.initial_rz]
        
        # Set speed and blend radius based on movement type
        speed = self.move_speed * 2 if is_rapid else self.move_speed
        accel = self.move_accel * 2 if is_rapid else self.move_accel
        
        # CORRECCIÓN: Usar blend radius solo para movimientos lineales (G1)
        # Para movimientos rápidos (G0), usar blend=0 para mayor precisión en el posicionamiento
        blend = 0 if is_rapid else self.blend_radius
        
        # Move the robot
        try:
            # Usar el formato correcto para moveL
            self.control.moveL(target_pose, speed, accel, blend)
            # Reduce el tiempo de espera para movimientos más fluidos
            time.sleep(0.05)
        except Exception as e:
            print(f"Movement error: {e}")
            return False
        
        # Update current position
        self.current_x = x
        self.current_y = y
        self.current_z = z
        
        return True
    
    def move_arc_by_linear_segments(self, end_x, end_y, center_i, center_j, z, is_clockwise=True):
        """Move the robot in an arc by approximating with multiple linear movements
        
        Args:
            end_x, end_y: End point of the arc
            center_i, center_j: Offset from current position to center of arc
            z: Z-coordinate for the movement
            is_clockwise: True for G2 (CW), False for G3 (CCW)
        """
        # Apply offset to coordinates
        end_x = end_x + self.x_offset
        end_y = -end_y + self.y_offset
        
        # Calculate arc center (considering the inverted y-axis)
        center_x = self.current_x + center_i
        center_y = self.current_y - center_j  # Negative due to inverted y-axis
        
        # Verify current position is set
        if self.current_x is None or self.current_y is None or self.current_z is None:
            print("Warning: Current position not set, cannot create arc")
            return False
        
        # Check if end point is within workspace
        if not self.is_point_within_reach(end_x, end_y, z):
            print(f"Warning: End point ({end_x:.3f}, {end_y:.3f}, {z:.3f}) is outside workspace!")
            return False
        
        try:
            # Calculate vectors from center to start and end points
            start_vec = [self.current_x - center_x, self.current_y - center_y]
            end_vec = [end_x - center_x, end_y - center_y]
            
            # Calculate radius as magnitude of vector from center to start
            radius = np.sqrt(start_vec[0]**2 + start_vec[1]**2)
            
            # Calculate start and end angles (relative to positive x-axis)
            start_angle = np.arctan2(start_vec[1], start_vec[0])
            end_angle = np.arctan2(end_vec[1], end_vec[0])
            
            # Ensure proper angle direction
            if is_clockwise:
                if end_angle > start_angle:
                    end_angle -= 2 * np.pi
            else:  # Counter-clockwise
                if end_angle < start_angle:
                    end_angle += 2 * np.pi
            
            # Calculate angle difference
            angle_diff = abs(end_angle - start_angle)
            
            # Determine number of segments (more segments for larger arcs)
            # Use at least 12 segments for a full circle, scaled by arc length
            segments = max(int(12 * angle_diff / (2 * np.pi)), 5)
            
            # Generate points along the arc
            for i in range(1, segments + 1):
                # Calculate angle for this segment
                segment_ratio = i / segments
                if is_clockwise:
                    angle = start_angle + (end_angle - start_angle) * segment_ratio
                else:
                    angle = start_angle + (end_angle - start_angle) * segment_ratio
                
                # Calculate point coordinates
                point_x = center_x + radius * np.cos(angle)
                point_y = center_y + radius * np.sin(angle)
                
                # Create pose
                pose = [point_x, point_y, z, self.initial_rx, self.initial_ry, self.initial_rz]
                
                # Check if point is within workspace
                if not self.is_point_within_reach(point_x, point_y, z):
                    print(f"Warning: Arc segment point ({point_x:.3f}, {point_y:.3f}, {z:.3f}) is outside workspace!")
                    return False
                
                # Para todos excepto el último segmento, usar blending
                blend = self.blend_radius if i < segments else 0
                
                # Use moveL for each segment - EXACTAMENTE como en gcode_t_ur.py original
                self.control.moveL(pose, self.move_speed, self.move_accel, blend)
                time.sleep(0.05)  # Short sleep between segments
            
            # Update current position
            self.current_x = end_x
            self.current_y = end_y
            self.current_z = z
            
            return True
        except Exception as e:
            print(f"Arc movement failed: {e}")
            return False
    
    def _execute_buffered_movements(self, points_buffer):
        """Execute a series of buffered movements as a continuous path"""
        if not points_buffer:
            return
        
        try:
            # Si es un solo punto, usar moveL normal
            if len(points_buffer) == 1:
                pose = points_buffer[0]
                self.control.moveL(pose, self.move_speed, self.move_accel, 0)
                # Actualizar posición actual
                self.current_x = pose[0]
                self.current_y = pose[1]
                self.current_z = pose[2]
                return
            
            # Para múltiples puntos, crear un path con blending entre puntos
            path = []
            for i, pose in enumerate(points_buffer):
                # El último punto tiene blend_radius=0 para asegurar llegar exactamente
                blend = 0 if i == len(points_buffer) - 1 else self.blend_radius
                path.append(pose + [self.move_speed, self.move_accel, blend])
            
            # Ejecutar el path completo
            self.control.moveL(path)
            
            # Actualizar posición actual al último punto
            last_pose = points_buffer[-1]
            self.current_x = last_pose[0]
            self.current_y = last_pose[1]
            self.current_z = last_pose[2]
            
        except Exception as e:
            print(f"Error executing buffered movements: {e}")
    
    def process_nc_file(self, file_path=None, nc_code=None):
        """Process NC code from a file or string"""
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    nc_lines = f.readlines()
                print(f"Successfully loaded NC file: {file_path}")
                if self.progress_callback:
                    self.progress_callback(f"Archivo NC cargado: {file_path}")
            except Exception as e:
                print(f"Error reading NC file: {e}")
                if self.progress_callback:
                    self.progress_callback(f"Error al leer archivo NC: {e}")
                return False
        elif nc_code:
            nc_lines = nc_code.strip().split('\n')
            print("Processing NC code from string input")
            if self.progress_callback:
                self.progress_callback("Procesando código NC desde entrada de texto")
        else:
            print("Error: No NC code provided")
            if self.progress_callback:
                self.progress_callback("Error: No se ha proporcionado código NC")
            return False
        
        
        
        # Get current TCP pose to initialize values
        tcp_pose = self.receive.getActualTCPPose()
        self.current_x = tcp_pose[0]
        self.current_y = tcp_pose[1]
        self.current_z = tcp_pose[2]
        
        # Filtrar solo las líneas de G-code válidas (G0, G1, G2 o G3)
        valid_line_pattern = r'G[0123]\s'
        valid_lines = [line.strip() for line in nc_lines if line.strip() and re.match(valid_line_pattern, line.strip())]
        total_valid_lines = len(valid_lines)
        
        print(f"Starting NC code processing with {total_valid_lines} valid G-code lines")
        if self.progress_callback:
            self.progress_callback(f"Iniciando procesamiento de código NC con {total_valid_lines} líneas G-code válidas")
        
        # Process each line of NC code
        processed_lines = 0
        progress_percentage_old = 0
        progress_percentage = 0
        linear_points_buffer = []  # Buffer for sequential G1 movements
        
        for i, line in enumerate(nc_lines):
            line = line.strip()
            if not line:
                continue
                    
            # Parse G-code command
            g_match = re.match(r'G([0123])\s', line)
            if not g_match:
                # Si hay puntos en el buffer y cambiamos de comando, los ejecutamos
                if linear_points_buffer:
                    self._execute_buffered_movements(linear_points_buffer)
                    linear_points_buffer = []
                print(f"Skipping unsupported command: {line}")
                continue
                    
            g_code = int(g_match.group(1))
            is_rapid = (g_code == 0)      # G0 is rapid movement
            is_linear = (g_code == 1)     # G1 is linear movement
            is_arc_cw = (g_code == 2)     # G2 is clockwise arc
            is_arc_ccw = (g_code == 3)    # G3 is counterclockwise arc
            
            # Si cambiamos de tipo de movimiento (G1->G0, G1->G2, etc), ejecutamos buffer pendiente
            if (not is_linear) and linear_points_buffer:
                self._execute_buffered_movements(linear_points_buffer)
                linear_points_buffer = []
                
            processed_lines += 1
            

            # --------------------------------- PROYECTO LEON ---------------------------------
            # Leer I/O si esta en true segir, sino detener hasta que sea true
            while not self.receive.getDigitalInState(0):
                print("Waiting for I/O signal to start processing...")
                if self.progress_callback:
                    self.progress_callback("Esperando señal de I/O para iniciar procesamiento...")
                time.sleep(1)  # Esperar medio segundo antes de volver a comprobar

            # Calcular porcentaje de progreso
            progress_percentage = (processed_lines / total_valid_lines) * 100
            if progress_percentage >= progress_percentage_old + 5:
                progress_percentage_old = progress_percentage
                progress_msg = f"Porcentaje({progress_percentage:.2f}%)"
                print(progress_msg)
                # if self.progress_callback:
                #     self.progress_callback(progress_msg)
                # Encender I/O para indicar progreso durante medio segundo
                self.io.setStandardDigitalOut(0, True)
                time.sleep(0.5)
                self.io.setStandardDigitalOut(0, False)
            
            if progress_percentage >= 100:
                for i in range(5):
                    self.io.setStandardDigitalOut(0, True)
                    time.sleep(0.5)
                    self.io.setStandardDigitalOut(0, False)

            # ---------------------------------------------------------------------------------
            
            
            # Parse X, Y coordinates
            x_match = re.search(r'X([-\d.]+)', line)
            y_match = re.search(r'Y([-\d.]+)', line)
            
            # For arc movements, also parse I, J (arc center offset)
            i_match = re.search(r'I([-\d.]+)', line) if (is_arc_cw or is_arc_ccw) else None
            j_match = re.search(r'J([-\d.]+)', line) if (is_arc_cw or is_arc_ccw) else None
            
            # For linear or rapid movements (G0, G1)
            if (is_rapid or is_linear) and x_match and y_match:
                # Convert from mm to meters
                x = float(x_match.group(1)) / 1000.0
                y = float(y_match.group(1)) / 1000.0
                
                # Set Z based on whether it's a drawing move or a rapid move
                z = self.drawing_plane_z if not is_rapid else self.drawing_plane_z + self.line_offset_z
                
                # Progress update
                movement_type = 'Movimiento rápido' if is_rapid else 'Movimiento lineal'
                progress_msg = f"Línea {processed_lines}/{total_valid_lines}: {movement_type} a X={x*1000:.2f}mm, Y={y*1000:.2f}mm"
                print(progress_msg)
                if self.progress_callback:
                    self.progress_callback(progress_msg)
                
                # Si es movimiento lineal (G1), acumulamos para path continuo
                if is_linear:
                    # Apply offset to x and y coordinates
                    x_adjusted = x + self.x_offset
                    y_adjusted = -y + self.y_offset
                    
                    # Create pose with initial TCP orientation
                    target_pose = [x_adjusted, y_adjusted, z, self.initial_rx, self.initial_ry, self.initial_rz]
                    linear_points_buffer.append(target_pose)
                    
                    # Si el buffer alcanza un máximo, lo ejecutamos
                    if len(linear_points_buffer) >= 10:
                        self._execute_buffered_movements(linear_points_buffer)
                        linear_points_buffer = []
                else:
                    # Para G0 (movimiento rápido), ejecutamos directamente
                    if not self.move_robot(x, y, z, is_rapid=True):
                        print(f"Warning: Movement to ({x:.3f}, {y:.3f}, {z:.3f}) failed!")
                        if self.progress_callback:
                            self.progress_callback(f"¡Advertencia! Movimiento a ({x*1000:.2f}mm, {y*1000:.2f}mm) falló")
            
            # For arc movements (G2, G3)
            elif (is_arc_cw or is_arc_ccw) and x_match and y_match and i_match and j_match:
                # Convert from mm to meters
                end_x = float(x_match.group(1)) / 1000.0
                end_y = float(y_match.group(1)) / 1000.0
                center_i = float(i_match.group(1)) / 1000.0  # I is X offset from current position to center
                center_j = float(j_match.group(1)) / 1000.0  # J is Y offset from current position to center
                
                # Z position for arc is always drawing plane (no rapid arcs in standard gcode)
                z = self.drawing_plane_z
                
                # Progress update
                arc_type = 'Arco horario (CW)' if is_arc_cw else 'Arco antihorario (CCW)'
                progress_msg = f"Línea {processed_lines}/{total_valid_lines}: {arc_type} a X={end_x*1000:.2f}mm, Y={end_y*1000:.2f}mm (I={center_i*1000:.2f}mm, J={center_j*1000:.2f}mm)"
                print(progress_msg)
                if self.progress_callback:
                    self.progress_callback(progress_msg)
                
                # Execute the arc movement using linear segment approximation
                # NOTA: Usar EXACTAMENTE el mismo algoritmo de arco del archivo original
                if not self.move_arc_by_linear_segments(end_x, end_y, center_i, center_j, z, is_clockwise=is_arc_cw):
                    print(f"Warning: Arc movement to ({end_x:.3f}, {end_y:.3f}, {z:.3f}) failed!")
                    if self.progress_callback:
                        self.progress_callback(f"¡Advertencia! Movimiento de arco a ({end_x*1000:.2f}mm, {end_y*1000:.2f}mm) falló")
            
            # Incomplete G2/G3 command
            elif (is_arc_cw or is_arc_ccw):
                print(f"Warning: Incomplete arc command: {line}")
                if self.progress_callback:
                    self.progress_callback(f"¡Advertencia! Comando de arco incompleto: {line}")
        
        # Procesar cualquier punto restante en el buffer
        if linear_points_buffer:
            self._execute_buffered_movements(linear_points_buffer)
        
        print("NC code processing complete")
        if self.progress_callback:
            self.progress_callback(f"Procesamiento de código NC completado. Se ejecutaron {processed_lines} líneas de {total_valid_lines}.")
        return True

def main(robot_ip="192.168.1.1", progress_callback=None):
    # Set default file path for NC code
    default_nc_file = "out.nc"
    
    # Create NC to UR converter
    converter = NCtoURConverter(robot_ip=robot_ip, progress_callback=progress_callback)
    
    # Initialize robot connection
    print("Initializing robot connection...")
    if progress_callback:
        progress_callback("Inicializando conexión con el robot...")
    
    while not converter.initialize_robot():
        print("Retrying connection to robot...")
        if progress_callback:
            progress_callback("Reintentando conexión con el robot...")
        time.sleep(1)
    
    # Move to home position
    print("Moving to home position...")
    if progress_callback:
        progress_callback("Moviendo a posición inicial...")
    converter.go_home()
    time.sleep(2)
    
    # Process NC code from file
    print(f"Processing NC code from file: {default_nc_file}")
    if progress_callback:
        progress_callback(f"Procesando código NC desde archivo: {default_nc_file}")
    
    # Check if file exists, otherwise use hardcoded NC code
    try:
        converter.process_nc_file(file_path=default_nc_file)
    except FileNotFoundError:
        print(f"File {default_nc_file} not found. Using hardcoded NC code.")
        if progress_callback:
            progress_callback(f"Archivo {default_nc_file} no encontrado. Usando código NC predefinido.")
        # Hardcoded NC code as fallback - now including arc examples
        nc_code = """G0 X60.00 Y-2.00
G1 X63.00 Y-1.00
G1 X64.00 Y-4.00
G2 X67.00 Y-2.00 I1.50 J2.50
G3 X70.00 Y-5.00 I0.00 J-3.00
"""
        converter.process_nc_file(nc_code=nc_code)
    
    # Return to home position when done
    print("Execution complete. Returning to home position...")
    if progress_callback:
        progress_callback("Ejecución completada. Volviendo a posición inicial...")
    converter.go_home()
    print("Program finished")
    if progress_callback:
        progress_callback("Programa finalizado")
    return True

if __name__ == "__main__":
    main()