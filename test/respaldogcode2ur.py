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
        # Resto del código sin cambios...
        #self.ip = robot_ip
        self.control = None
        self.receive = None
        self.io = None
        self.drawing_plane_z = 0.002  # 16.5mm drawing plane
        self.line_offset_z = .01  # 25mm offset between lines
        self.blend_radius = 0.005     # 5mm blend radius
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.initial_rx = None
        self.initial_ry = None
        self.initial_rz = None
        self.move_speed = 0.5
        self.move_accel = 0.5
        # Add X and Y offset of -0.3 meters
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
        blend = 0 if is_rapid else self.blend_radius
        
        # Move the robot - removed asynchronous flag
        if is_rapid:
            # For G0 (rapid movement), we'll use moveL without blending
            self.control.moveL(target_pose, speed, accel)
            # Add sleep to ensure movement completes
            time.sleep(0.1)
        else:
            # For G1 (linear movement), use proper path format with blending
            path_pose = target_pose + [speed, accel, blend]
            self.control.moveL([path_pose])
            # Add sleep to ensure movement completes
            time.sleep(0.1)
        
        # Update current position
        self.current_x = x
        self.current_y = y
        self.current_z = z
        
        return True
    
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
        
        # Filtrar solo las líneas de G-code válidas (G0 o G1)
        valid_lines = [line.strip() for line in nc_lines if line.strip() and re.match(r'G[01]\s', line.strip())]
        total_valid_lines = len(valid_lines)
        
        print(f"Starting NC code processing with {total_valid_lines} valid G-code lines")
        if self.progress_callback:
            self.progress_callback(f"Iniciando procesamiento de código NC con {total_valid_lines} líneas G-code válidas")
        
        # Process each line of NC code
        processed_lines = 0
        for i, line in enumerate(nc_lines):
            line = line.strip()
            if not line:
                continue
                    
            # Parse G-code command
            g_match = re.match(r'G([01])\s', line)
            if not g_match:
                print(f"Skipping unsupported command: {line}")
                continue
                    
            g_code = int(g_match.group(1))
            is_rapid = (g_code == 0)  # G0 is rapid movement
            processed_lines += 1
                
            # Parse X, Y coordinates
            x_match = re.search(r'X([-\d.]+)', line)
            y_match = re.search(r'Y([-\d.]+)', line)
            
            if x_match and y_match:
                # Convert from mm to meters - handle conversion properly
                x = float(x_match.group(1)) / 1000.0
                y = float(y_match.group(1)) / 1000.0
                
                # Set Z based on whether it's a drawing move or a rapid move
                z = self.drawing_plane_z if not is_rapid else self.drawing_plane_z + self.line_offset_z
                
                # Progress update - Actualizamos en CADA línea para mejor seguimiento
                progress_msg = f"Línea {processed_lines}/{total_valid_lines}: {'Movimiento rápido' if is_rapid else 'Movimiento lineal'} a X={x*1000:.2f}mm, Y={y*1000:.2f}mm"
                print(progress_msg)
                if self.progress_callback:
                    self.progress_callback(progress_msg)
                
                # Execute the movement
                if not self.move_robot(x, y, z, is_rapid):
                    print(f"Warning: Movement to ({x:.3f}, {y:.3f}, {z:.3f}) failed!")
                    if self.progress_callback:
                        self.progress_callback(f"¡Advertencia! Movimiento a ({x*1000:.2f}mm, {y*1000:.2f}mm) falló")
        
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
        progress_callback("Initializing robot connection...")
    
    while not converter.initialize_robot():
        print("Retrying connection to robot...")
        if progress_callback:
            progress_callback("Retrying connection to robot...")
        time.sleep(1)
    
    # Move to home position
    print("Moving to home position...")
    if progress_callback:
        progress_callback("Moving to home position...")
    converter.go_home()
    time.sleep(2)
    
    # Process NC code from file
    print(f"Processing NC code from file: {default_nc_file}")
    if progress_callback:
        progress_callback(f"Processing NC code from file: {default_nc_file}")
    
    # Check if file exists, otherwise use hardcoded NC code
    try:
        converter.process_nc_file(file_path=default_nc_file)
    except FileNotFoundError:
        print(f"File {default_nc_file} not found. Using hardcoded NC code.")
        if progress_callback:
            progress_callback(f"File {default_nc_file} not found. Using hardcoded NC code.")
        # Hardcoded NC code as fallback
        nc_code = """G0 X60.00 Y-2.00
G1 X63.00 Y-1.00
G1 X64.00 Y-4.00
"""
        converter.process_nc_file(nc_code=nc_code)
    
    # Return to home position when done
    print("Execution complete. Returning to home position...")
    if progress_callback:
        progress_callback("Execution complete. Returning to home position...")
    converter.go_home()
    print("Program finished")
    if progress_callback:
        progress_callback("Program finished")
    return True



# def main():
#     # Set default file path for NC code
#     default_nc_file = "out.nc"
    
#     # Create NC to UR converter
#     converter = NCtoURConverter(robot_ip="192.168.1.1")
    
#     # Initialize robot connection
#     print("Initializing robot connection...")
#     while not converter.initialize_robot():
#         print("Retrying connection to robot...")
#         time.sleep(1)
    
#     # Move to home position
#     print("Moving to home position...")
#     converter.go_home()
#     time.sleep(2)
    
#     # Process NC code from file
#     print(f"Processing NC code from file: {default_nc_file}")
#     # Check if file exists, otherwise use hardcoded NC code
#     try:
#         converter.process_nc_file(file_path=default_nc_file)
#     except FileNotFoundError:
#         print(f"File {default_nc_file} not found. Using hardcoded NC code.")
#         # Hardcoded NC code as fallback
#         nc_code = """G0 X60.00 Y-2.00
# G1 X63.00 Y-1.00
# G1 X64.00 Y-4.00
# """
#         converter.process_nc_file(nc_code=nc_code)
    
#     # Return to home position when done
#     print("Execution complete. Returning to home position...")
#     converter.go_home()
#     print("Program finished")

if __name__ == "__main__":
    main()