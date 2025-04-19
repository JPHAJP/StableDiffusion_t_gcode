import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import math

def parse_gcode_line(line, current_pos):
    """
    Analyzes a G-code line.
    Handles G0, G1 (linear movements) and G2, G3 (circular arcs).
    Returns the command type and movement parameters.
    """
    cmd = None
    
    # Detect movement type
    if line.startswith("G0") or line.startswith("G00"):
        cmd = "G0"
    elif line.startswith("G1") or line.startswith("G01"):
        cmd = "G1"
    elif line.startswith("G2") or line.startswith("G02"):
        cmd = "G2"  # Clockwise arc
    elif line.startswith("G3") or line.startswith("G03"):
        cmd = "G3"  # Counterclockwise arc
    
    if not cmd:
        return None, current_pos, {}
    
    # Search for coordinates using regular expressions
    x_match = re.search(r'X([-+]?[0-9]*\.?[0-9]+)', line)
    y_match = re.search(r'Y([-+]?[0-9]*\.?[0-9]+)', line)
    
    # New position
    x = float(x_match.group(1)) if x_match else current_pos[0]
    y = float(y_match.group(1)) if y_match else current_pos[1]
    new_pos = (x, y)
    
    # For arcs, we also need center offset or radius
    params = {}
    if cmd in ["G2", "G3"]:
        # Check for IJK center format
        i_match = re.search(r'I([-+]?[0-9]*\.?[0-9]+)', line)
        j_match = re.search(r'J([-+]?[0-9]*\.?[0-9]+)', line)
        
        # Check for R radius format
        r_match = re.search(r'R([-+]?[0-9]*\.?[0-9]+)', line)
        
        if i_match and j_match:
            # IJK format - relative center coordinates
            i = float(i_match.group(1))
            j = float(j_match.group(1))
            params['center'] = (current_pos[0] + i, current_pos[1] + j)
        elif r_match:
            # R format - radius value
            params['radius'] = float(r_match.group(1))
            
    return cmd, new_pos, params

def calculate_arc_points(start, end, center=None, radius=None, is_clockwise=True, resolution=36):
    """
    Calculate points along an arc from start to end.
    Can use either center point or radius.
    """
    # If neither center nor radius is provided, return a straight line
    if center is None and radius is None:
        return [start, end]
        
    if center:
        # Calculate with center point
        center_x, center_y = center
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate radius from center and start
        radius = math.sqrt((start_x - center_x)**2 + (start_y - center_y)**2)
        
        # Calculate angles
        start_angle = math.atan2(start_y - center_y, start_x - center_x)
        end_angle = math.atan2(end_y - center_y, end_x - center_x)
        
        # Ensure proper angle direction
        if is_clockwise:
            if end_angle > start_angle:
                end_angle -= 2 * math.pi
        else:
            if end_angle < start_angle:
                end_angle += 2 * math.pi
        
    elif radius is not None:  # Ensure radius is provided
        # Calculating with radius
        start_x, start_y = start
        end_x, end_y = end
        
        # Distance between points
        chord = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Cannot create arc if chord > 2*radius
        if chord > 2 * abs(radius):
            # Fallback to straight line
            return [start, end]
            
        # Calculate center point
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # Distance from midpoint to center
        h = math.sqrt(radius**2 - (chord/2)**2)
        
        # Unit vector from start to end
        dx = end_x - start_x
        dy = end_y - start_y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        if dist < 1e-10:  # Very small distance
            return [start, end]
            
        dx, dy = dx/dist, dy/dist
        
        # Perpendicular unit vector
        if (is_clockwise and radius > 0) or (not is_clockwise and radius < 0):
            center_x = mid_x - h * dy
            center_y = mid_y + h * dx
        else:
            center_x = mid_x + h * dy
            center_y = mid_y - h * dx
            
        center = (center_x, center_y)
        
        # Recalculate angles
        start_angle = math.atan2(start_y - center_y, start_x - center_x)
        end_angle = math.atan2(end_y - center_y, end_x - center_x)
        
        # Adjust angles for direction
        if is_clockwise:
            if end_angle > start_angle:
                end_angle -= 2 * math.pi
        else:
            if end_angle < start_angle:
                end_angle += 2 * math.pi
    else:
        # If we somehow got here with no valid parameters, return a straight line
        return [start, end]
    
    # Generate points along the arc
    if is_clockwise:
        theta = np.linspace(start_angle, end_angle, resolution)
    else:
        theta = np.linspace(start_angle, end_angle, resolution)
    
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    
    return list(zip(x, y))

def plot_gcode(gcode_file, output_png):
    """
    Reads a G-code file and generates a PNG plot.
    Draws:
      - G0 movements (jumps) in yellow
      - G1 movements (work) in blue
      - G2 arcs (clockwise) in red
      - G3 arcs (counterclockwise) in green
    """
    current_pos = (0, 0)
    segments_g0 = []    # List of G0 segments (jumps)
    segments_g1 = []    # List of G1 segments (linear work)
    segments_g2 = []    # List of G2 segments (clockwise arcs)
    segments_g3 = []    # List of G3 segments (counterclockwise arcs)

    # Read the G-code file
    with open(gcode_file, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                if not line or line.startswith(";"):  # Ignore empty lines or comments
                    continue
                    
                cmd, new_pos, params = parse_gcode_line(line, current_pos)
                
                if cmd:
                    if cmd == "G0":
                        segments_g0.append((current_pos, new_pos))
                    elif cmd == "G1":
                        segments_g1.append((current_pos, new_pos))
                    elif cmd == "G2":
                        # Handle clockwise arc
                        points = calculate_arc_points(
                            current_pos, new_pos, 
                            center=params.get('center'), 
                            radius=params.get('radius'),
                            is_clockwise=True
                        )
                        segments_g2.append(points)
                    elif cmd == "G3":
                        # Handle counterclockwise arc
                        points = calculate_arc_points(
                            current_pos, new_pos, 
                            center=params.get('center'), 
                            radius=params.get('radius'),
                            is_clockwise=False
                        )
                        segments_g3.append(points)
                    
                    current_pos = new_pos
            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Error details: {e}")
                # Continue processing other lines

    # Create the figure
    plt.figure(figsize=(10, 10))
    
    # Draw G0 segments in yellow
    for seg in segments_g0:
        x_values = [seg[0][0], seg[1][0]]
        y_values = [seg[0][1], seg[1][1]]
        plt.plot(x_values, y_values, color="yellow", linewidth=1.5, label="G0")
    
    # Draw G1 segments in blue
    for seg in segments_g1:
        x_values = [seg[0][0], seg[1][0]]
        y_values = [seg[0][1], seg[1][1]]
        plt.plot(x_values, y_values, color="blue", linewidth=2, label="G1")
    
    # Draw G2 arcs in red
    for points in segments_g2:
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        plt.plot(x_values, y_values, color="red", linewidth=2, label="G2")
    
    # Draw G3 arcs in green
    for points in segments_g3:
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        plt.plot(x_values, y_values, color="green", linewidth=2, label="G3")
    
    # Avoid duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("G-code Visualization")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_png}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python gcode_plotter.py gcode_file.txt output.png")
        sys.exit(1)
    gcode_file = sys.argv[1]
    output_png = sys.argv[2]
    plot_gcode(gcode_file, output_png)