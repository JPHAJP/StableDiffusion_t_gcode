import numpy as np
from scipy import ndimage
import imageio
from PIL import Image, ImageFilter
import argparse
#import imageio.v2 as imageio

# Definición de constantes directamente en el código
circumferences = [
    # r=0
    [(0,0)],
    # r=1
    [(1,0),(0,1),(-1,0),(0,-1)],
    # r=2
    [(2,0),(2,1),(1,2),(0,2),(-1,2),(-2,1),(-2,0),(-2,-1),(-1,-2),(0,-2),(1,-2),(2,-1)],
    # r=3
    [(3,0),(3,1),(2,2),(1,3),(0,3),(-1,3),(-2,2),(-3,1),(-3,0),(-3,-1),(-2,-2),(-1,-3),(0,-3),(1,-3),(2,-2),(3,-1)],
    # r=4
    [(4,0),(4,1),(4,2),(3,3),(2,4),(1,4),(0,4),(-1,4),(-2,4),(-3,3),(-4,2),(-4,1),(-4,0),(-4,-1),(-4,-2),(-3,-3),(-2,-4),(-1,-4),(0,-4),(1,-4),(2,-4),(3,-3),(4,-2),(4,-1)],
    # r=5
    [(5,0),(5,1),(5,2),(4,3),(3,4),(2,5),(1,5),(0,5),(-1,5),(-2,5),(-3,4),(-4,3),(-5,2),(-5,1),(-5,0),(-5,-1),(-5,-2),(-4,-3),(-3,-4),(-2,-5),(-1,-5),(0,-5),(1,-5),(2,-5),(3,-4),(4,-3),(5,-2),(5,-1)],
    # r=6
    [(6,0),(6,1),(6,2),(5,3),(5,4),(4,5),(3,5),(2,6),(1,6),(0,6),(-1,6),(-2,6),(-3,5),(-4,5),(-5,4),(-5,3),(-6,2),(-6,1),(-6,0),(-6,-1),(-6,-2),(-5,-3),(-5,-4),(-4,-5),(-3,-5),(-2,-6),(-1,-6),(0,-6),(1,-6),(2,-6),(3,-5),(4,-5),(5,-4),(5,-3),(6,-2),(6,-1)],
    # r=7
    [(7,0),(7,1),(7,2),(6,3),(6,4),(5,5),(4,6),(3,6),(2,7),(1,7),(0,7),(-1,7),(-2,7),(-3,6),(-4,6),(-5,5),(-6,4),(-6,3),(-7,2),(-7,1),(-7,0),(-7,-1),(-7,-2),(-6,-3),(-6,-4),(-5,-5),(-4,-6),(-3,-6),(-2,-7),(-1,-7),(0,-7),(1,-7),(2,-7),(3,-6),(4,-6),(5,-5),(6,-4),(6,-3),(7,-2),(7,-1)],
    # r=8
    [(8,0),(8,1),(8,2),(7,3),(7,4),(6,5),(5,6),(4,7),(3,7),(2,8),(1,8),(0,8),(-1,8),(-2,8),(-3,7),(-4,7),(-5,6),(-6,5),(-7,4),(-7,3),(-8,2),(-8,1),(-8,0),(-8,-1),(-8,-2),(-7,-3),(-7,-4),(-6,-5),(-5,-6),(-4,-7),(-3,-7),(-2,-8),(-1,-8),(0,-8),(1,-8),(2,-8),(3,-7),(4,-7),(5,-6),(6,-5),(7,-4),(7,-3),(8,-2),(8,-1)],
    # r=9
    [(9,0),(9,1),(9,2),(9,3),(8,4),(8,5),(7,6),(6,7),(5,8),(4,8),(3,9),(2,9),(1,9),(0,9),(-1,9),(-2,9),(-3,9),(-4,8),(-5,8),(-6,7),(-7,6),(-8,5),(-8,4),(-9,3),(-9,2),(-9,1),(-9,0),(-9,-1),(-9,-2),(-9,-3),(-8,-4),(-8,-5),(-7,-6),(-6,-7),(-5,-8),(-4,-8),(-3,-9),(-2,-9),(-1,-9),(0,-9),(1,-9),(2,-9),(3,-9),(4,-8),(5,-8),(6,-7),(7,-6),(8,-5),(8,-4),(9,-3),(9,-2),(9,-1)],
    # r=10
    [(10,0),(10,1),(10,2),(10,3),(9,4),(9,5),(8,6),(7,7),(6,8),(5,9),(4,9),(3,10),(2,10),(1,10),(0,10),(-1,10),(-2,10),(-3,10),(-4,9),(-5,9),(-6,8),(-7,7),(-8,6),(-9,5),(-9,4),(-10,3),(-10,2),(-10,1),(-10,0),(-10,-1),(-10,-2),(-10,-3),(-9,-4),(-9,-5),(-8,-6),(-7,-7),(-6,-8),(-5,-9),(-4,-9),(-3,-10),(-2,-10),(-1,-10),(0,-10),(1,-10),(2,-10),(3,-10),(4,-9),(5,-9),(6,-8),(7,-7),(8,-6),(9,-5),(9,-4),(10,-3),(10,-2),(10,-1)]
]


class CircularRange:
    def __init__(self, begin, end, value):
        self.begin, self.end, self.value = begin, end, value

    def __repr__(self):
        return f"[{self.begin},{self.end})->{self.value}"

    def halfway(self):
        return int((self.begin + self.end) / 2)

class Graph:
    class Node:
        def __init__(self, point, index):
            # Intercambiamos para que x corresponda a la columna y y a la fila
            self.x, self.y = -point[1], -point[0]
            self.index = index
            self.connections = {}

        def __repr__(self):
            return f"({self.x},{self.y})"

        def _addConnection(self, to):
            self.connections[to] = False # i.e. not already used in gcode generation

        def toDotFormat(self):
            return (f"{self.index} [pos=\"{self.x},{self.y}!\", label=\"{self.index}\\n{self.x},{self.y}\"]\n" +
                "".join(f"{self.index}--{conn}\n" for conn in self.connections if self.index < conn))


    def __init__(self):
        self.nodes = []

    def __getitem__(self, index):
        return self.nodes[index]

    def __repr__(self):
        return repr(self.nodes)


    def addNode(self, point):
        index = len(self.nodes)
        self.nodes.append(Graph.Node(point, index))
        return index

    def addConnection(self, a, b):
        self.nodes[a]._addConnection(b)
        self.nodes[b]._addConnection(a)

    def distance(self, a, b):
        return np.hypot(self[a].x-self[b].x, self[a].y-self[b].y)

    def areConnectedWithin(self, a, b, maxDistance):
        if maxDistance < 0:
            return False
        elif a == b:
            return True
        else:
            for conn in self[a].connections:
                if self.areConnectedWithin(conn, b, maxDistance - self.distance(conn, b)):
                    return True
            return False

    def saveAsDotFile(self, f):
        f.write("graph G {\nnode [shape=plaintext];\n")
        for node in self.nodes:
            f.write(node.toDotFormat())
        f.write("}\n")

    def find_lines(self):
        """
        Identifica líneas en el grafo (secuencias de nodos que pueden ser representados como líneas rectas)
        Retorna una lista de líneas, donde cada línea es una lista de índices de nodos
        """
        lines = []
        used_nodes = set()
        
        # Encuentra nodos extremos (con solo una conexión) o intersecciones (con más de 2 conexiones)
        endpoints = []
        for i, node in enumerate(self.nodes):
            if len(node.connections) == 1 or len(node.connections) > 2:
                endpoints.append(i)
        
        # Si no hay puntos extremos, busca cualquier nodo para formar un ciclo
        if not endpoints and self.nodes:
            endpoints = [0]  # Comienza con el primer nodo
        
        # Procesa cada punto extremo
        for start_node in endpoints:
            if start_node in used_nodes:
                continue
                
            # Inicializa una nueva línea desde este punto extremo
            line = [start_node]
            used_nodes.add(start_node)
            
            # Sigue la línea mientras sea posible
            current = start_node
            while True:
                # Encuentra el siguiente nodo no utilizado conectado al actual
                next_nodes = [n for n in self.nodes[current].connections if n not in used_nodes]
                
                # Si no hay más nodos o tenemos una intersección, terminamos esta línea
                if not next_nodes:
                    break
                
                # Continuamos con el siguiente nodo
                next_node = next_nodes[0]
                line.append(next_node)
                used_nodes.add(next_node)
                current = next_node
                
                # Si llegamos a un punto extremo o una intersección, terminamos esta línea
                if len(self.nodes[current].connections) != 2:
                    break
            
            # Si encontramos una línea válida (más de un nodo), la guardamos
            if len(line) > 1:
                lines.append(line)
        
        # Procesa los segmentos restantes (ciclos)
        remaining = [i for i in range(len(self.nodes)) if i not in used_nodes]
        while remaining:
            start_node = remaining[0]
            line = [start_node]
            used_nodes.add(start_node)
            remaining.remove(start_node)
            
            current = start_node
            first_iter = True
            
            while True:
                # Encuentra el siguiente nodo no utilizado o completa el ciclo
                next_candidates = list(self.nodes[current].connections.keys())
                next_nodes = [n for n in next_candidates if n not in used_nodes]
                
                # Si podemos cerrar el ciclo y no es el primer paso, cerramos el ciclo
                if first_iter:
                    first_iter = False
                elif start_node in next_candidates and len(line) > 2:
                    line.append(start_node)  # Cierra el ciclo
                    break
                
                # Si no hay más nodos, terminamos esta línea
                if not next_nodes:
                    break
                
                # Continuamos con el siguiente nodo
                next_node = next_nodes[0]
                line.append(next_node)
                used_nodes.add(next_node)
                if next_node in remaining:
                    remaining.remove(next_node)
                current = next_node
            
            # Si encontramos una línea válida, la guardamos
            if len(line) > 1:
                lines.append(line)
        
        return lines

    def optimize_line(self, line, epsilon=0.5):
        """
        Optimiza una línea reduciendo el número de puntos pero manteniendo la forma general
        Utiliza el algoritmo de Douglas-Peucker para simplificar la línea
        """
        if len(line) <= 2:  # Una línea con 2 o menos puntos no puede simplificarse más
            return line
        
        # Implementación del algoritmo de Douglas-Peucker
        def distance_point_to_line(point, line_start, line_end):
            if line_start == line_end:  # Si los puntos de la línea son iguales
                return np.hypot(self[point].x - self[line_start].x, self[point].y - self[line_start].y)
            
            # Cálculo de la distancia perpendicular de un punto a una línea
            x1, y1 = self[line_start].x, self[line_start].y
            x2, y2 = self[line_end].x, self[line_end].y
            x0, y0 = self[point].x, self[point].y
            
            # Longitud de la línea al cuadrado
            line_length_squared = (x2 - x1)**2 + (y2 - y1)**2
            
            # Si la línea es un punto, calculamos la distancia al punto
            if line_length_squared == 0:
                return np.hypot(x0 - x1, y0 - y1)
            
            # Proyección del punto en la línea parametrizada
            t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length_squared))
            
            # Punto más cercano en la línea
            projection_x = x1 + t * (x2 - x1)
            projection_y = y1 + t * (y2 - y1)
            
            # Distancia perpendicular
            return np.hypot(x0 - projection_x, y0 - projection_y)
        
        def douglas_peucker(points, epsilon):
            # Encuentra el punto más lejano de la línea formada por el primer y último punto
            dmax = 0
            index = 0
            start, end = points[0], points[-1]
            
            for i in range(1, len(points) - 1):
                d = distance_point_to_line(points[i], start, end)
                if d > dmax:
                    index = i
                    dmax = d
            
            # Si el punto más lejano está a una distancia mayor que epsilon, recursivamente simplifica
            if dmax > epsilon:
                # Recursivamente simplifica las dos partes
                rec_results1 = douglas_peucker(points[:index + 1], epsilon)
                rec_results2 = douglas_peucker(points[index:], epsilon)
                
                # Combina los resultados
                return rec_results1[:-1] + rec_results2
            else:
                # Todos los puntos están suficientemente cerca de la línea, solo mantén los extremos
                return [points[0], points[-1]]
        
        # Aplica el algoritmo con un épsilon adecuado
        return douglas_peucker(line, epsilon)

    def optimize_line_order(self, lines):
        """
        Ordena las líneas para minimizar la distancia de movimiento sin dibujar
        """
        if not lines:
            return []
            
        # Copia para no modificar el original
        remaining_lines = lines.copy()
        ordered_lines = []
        
        # Comienza con la primera línea
        current_line = remaining_lines.pop(0)
        ordered_lines.append(current_line)
        current_end = current_line[-1]
        
        # Mientras haya líneas restantes
        while remaining_lines:
            best_line = None
            best_distance = float('inf')
            best_index = 0
            reverse_best = False
            
            # Encuentra la línea más cercana
            for i, line in enumerate(remaining_lines):
                # Distancia al inicio de la línea
                start_distance = self.distance(current_end, line[0])
                # Distancia al final de la línea
                end_distance = self.distance(current_end, line[-1])
                
                # Elige la menor distancia
                if start_distance < best_distance:
                    best_distance = start_distance
                    best_line = line
                    best_index = i
                    reverse_best = False
                
                if end_distance < best_distance:
                    best_distance = end_distance
                    best_line = line
                    best_index = i
                    reverse_best = True
            
            # Agrega la mejor línea, posiblemente invertida
            if reverse_best:
                ordered_lines.append(list(reversed(best_line)))
                current_end = best_line[0]
            else:
                ordered_lines.append(best_line)
                current_end = best_line[-1]
                
            # Elimina la línea utilizada
            remaining_lines.pop(best_index)
        
        return ordered_lines

    def saveAsGcodeFile(self, f, scale=1.0, simplify_factor=0.5):
        """
        Genera un archivo G-code optimizado utilizando líneas en lugar de puntos individuales
        Con coordenadas positivas X e Y
        """
        # Calcula los límites del dibujo para hacer offset si es necesario
        min_x = min(node.x for node in self.nodes) if self.nodes else 0
        min_y = min(node.y for node in self.nodes) if self.nodes else 0
        
        # Offset para asegurar coordenadas positivas
        x_offset = abs(min_x) if min_x < 0 else 0
        y_offset = abs(min_y) if min_y < 0 else 0
        
        # Identificar y optimizar las líneas
        lines = self.find_lines()
        optimized_lines = [self.optimize_line(line, simplify_factor) for line in lines]
        ordered_lines = self.optimize_line_order(optimized_lines)
        
        # Escribe encabezado del G-code
        f.write("; G-code generado con optimización de líneas\n")
        f.write("; Total de líneas: {}\n".format(len(ordered_lines)))
        f.write("G21 ; Unidades en mm\n")
        f.write("G90 ; Coordenadas absolutas\n")
        f.write("G92 X0 Y0 ; Establecer posición actual como origen\n\n")
        
        # Escribe el G-code para cada línea
        last_point = None
        for line_idx, line in enumerate(ordered_lines):
            if not line:
                continue
                
            # Comentario para identificar la línea
            f.write(f"; Línea {line_idx+1} de {len(ordered_lines)}\n")
                
            # Primer punto de la línea - movimiento rápido si es necesario
            first_node = line[0]
            if last_point is None or last_point != first_node:
                f.write(f"G0 X{scale * (self[first_node].x + x_offset):.2f} Y{scale * (self[first_node].y + y_offset):.2f}\n")
            
            # Dibuja la línea pasando por todos los puntos optimizados
            for node_idx in line[1:]:
                f.write(f"G1 X{scale * (self[node_idx].x + x_offset):.2f} Y{scale * (self[node_idx].y + y_offset):.2f}\n")
            
            # Actualiza el último punto
            last_point = line[-1]
        
        # Finaliza el G-code
        f.write("\n; Fin del G-code\n")
        f.write("G0 X0 Y0 ; Retorno al origen\n")


class EdgesToGcode:
    def __init__(self, edges):
        self.edges = edges
        self.ownerNode = np.full(np.shape(edges), -1, dtype=int)
        self.xSize, self.ySize = np.shape(edges)
        self.graph = Graph()

    def getCircularArray(self, center, r, smallerArray = None):
        circumferenceSize = len(circumferences[r])
        circularArray = np.zeros(circumferenceSize, dtype=bool)

        if smallerArray is None:
            smallerArray = np.ones(1, dtype=bool)
        smallerSize = np.shape(smallerArray)[0]
        smallerToCurrentRatio = smallerSize / circumferenceSize

        for i in range(circumferenceSize):
            x = center[0] + circumferences[r][i][0]
            y = center[1] + circumferences[r][i][1]

            if x not in range(self.xSize) or y not in range(self.ySize):
                circularArray[i] = False # consider pixels outside of the image as not-edges
            else:
                iSmaller = i * smallerToCurrentRatio
                a, b = int(np.floor(iSmaller)), int(np.ceil(iSmaller))

                if smallerArray[a] == False and (b not in range(smallerSize) or smallerArray[b] == False):
                    circularArray[i] = False # do not take into consideration not connected regions (roughly)
                else:
                    circularArray[i] = self.edges[x, y]

        return circularArray

    def toCircularRanges(self, circularArray):
        ranges = []
        circumferenceSize = np.shape(circularArray)[0]

        lastValue, lastValueIndex = circularArray[0], 0
        for i in range(1, circumferenceSize):
            if circularArray[i] != lastValue:
                ranges.append(CircularRange(lastValueIndex, i, lastValue))
                lastValue, lastValueIndex = circularArray[i], i

        ranges.append(CircularRange(lastValueIndex, circumferenceSize, lastValue))
        if len(ranges) > 1 and ranges[-1].value == ranges[0].value:
            ranges[0].begin = ranges[-1].begin - circumferenceSize
            ranges.pop() # the last range is now contained in the first one
        return ranges

    def getNextPoints(self, point):
        """
        Returns the radius of the circle used to identify the points and
        the points toward which propagate, in a tuple `(radius, [point0, point1, ...])`
        """

        bestRadius = 0
        circularArray = self.getCircularArray(point, 0)
        allRanges = [self.toCircularRanges(circularArray)]
        for radius in range(1, len(circumferences)):
            circularArray = self.getCircularArray(point, radius, circularArray)
            allRanges.append(self.toCircularRanges(circularArray))
            if len(allRanges[radius]) > len(allRanges[bestRadius]):
                bestRadius = radius
            if len(allRanges[bestRadius]) >= 4 and len(allRanges[-2]) >= len(allRanges[-1]):
                # two consecutive circular arrays with the same or decreasing number>=4 of ranges
                break
            elif len(allRanges[radius]) == 2 and radius > 1:
                edge = 0 if allRanges[radius][0].value == True else 1
                if allRanges[radius][edge].end-allRanges[radius][edge].begin < len(circumferences[radius]) / 4:
                    # only two ranges but the edge range is small (1/4 of the circumference)
                    if bestRadius == 1:
                        bestRadius = 2
                    break
            elif len(allRanges[radius]) == 1 and allRanges[radius][0].value == False:
                # this is a point-shaped edge not sorrounded by any edges
                break

        if bestRadius == 0:
            return 0, []

        circularRanges = allRanges[bestRadius]
        points = []
        for circularRange in circularRanges:
            if circularRange.value == True:
                circumferenceIndex = circularRange.halfway()
                x = point[0] + circumferences[bestRadius][circumferenceIndex][0]
                y = point[1] + circumferences[bestRadius][circumferenceIndex][1]

                if x in range(self.xSize) and y in range(self.ySize) and self.ownerNode[x, y] == -1:
                    points.append((x,y))

        return bestRadius, points

    def propagate(self, point, currentNodeIndex):
        radius, nextPoints = self.getNextPoints(point)

        # depth first search to set the owner of all reachable connected pixels
        # without an owner and find connected nodes
        allConnectedNodes = set()
        def setSeenDFS(x, y):
            if (x in range(self.xSize) and y in range(self.ySize)
                    and np.hypot(x-point[0], y-point[1]) <= radius + 0.5
                    and self.edges[x, y] == True and self.ownerNode[x, y] != currentNodeIndex):
                if self.ownerNode[x, y] != -1:
                    allConnectedNodes.add(self.ownerNode[x, y])
                self.ownerNode[x, y] = currentNodeIndex # index of just added node
                setSeenDFS(x+1, y)
                setSeenDFS(x-1, y)
                setSeenDFS(x, y+1)
                setSeenDFS(x, y-1)

        self.ownerNode[point] = -1 # reset to allow DFS to start
        setSeenDFS(*point)
        for nodeIndex in allConnectedNodes:
            if not self.graph.areConnectedWithin(currentNodeIndex, nodeIndex, 11):
                self.graph.addConnection(currentNodeIndex, nodeIndex)

        validNextPoints = []
        for nextPoint in nextPoints:
            if self.ownerNode[nextPoint] == currentNodeIndex:
                # only if this point belongs to the current node after the DFS,
                # which means it is reachable and connected
                validNextPoints.append(nextPoint)

        for nextPoint in validNextPoints:
            nodeIndex = self.graph.addNode(nextPoint)
            self.graph.addConnection(currentNodeIndex, nodeIndex)
            self.propagate(nextPoint, nodeIndex)
            self.ownerNode[point] = currentNodeIndex

    def addNodeAndPropagate(self, point):
        nodeIndex = self.graph.addNode(point)
        self.propagate(point, nodeIndex)

    def buildGraph(self):
        for point in np.ndindex(np.shape(self.edges)):
            if self.edges[point] == True and self.ownerNode[point] == -1:
                radius, nextPoints = self.getNextPoints(point)
                if radius == 0:
                    self.addNodeAndPropagate(point)
                else:
                    for nextPoint in nextPoints:
                        if self.ownerNode[nextPoint] == -1:
                            self.addNodeAndPropagate(nextPoint)

        return self.graph


def sobel(image):
    image = np.array(image, dtype=float)
    image /= 255.0
    Gx = ndimage.sobel(image, axis=0)
    Gy = ndimage.sobel(image, axis=1)
    res = np.hypot(Gx, Gy)
    res /= np.max(res)
    res = np.array(res * 255, dtype=np.uint8)
    
    # Check if the result is 2D (grayscale) or 3D (color)
    if res.ndim == 2:
        return res[2:-2, 2:-2]
    else:
        return res[2:-2, 2:-2, 0:3]


def convertToBinaryEdges(edges, threshold):
    if edges.ndim == 2:
        # For a grayscale image, simply threshold the 2D array
        result = edges >= threshold
    else:
        # For a color image, combine the channels as before
        result = np.maximum.reduce([edges[:, :, 0], edges[:, :, 1], edges[:, :, 2]]) >= threshold
        if edges.shape[2] > 3:
            result[edges[:, :, 3] < threshold] = False
    return result


def parseArgs(namespace):
    argParser = argparse.ArgumentParser(fromfile_prefix_chars="@",
        description="Detecta los bordes de una imagen y los convierte a código G para impresión 2D")

    argParser.add_argument_group("Opciones de datos")
    argParser.add_argument("-i", "--input", type=argparse.FileType('br'), required=True, metavar="ARCHIVO",
        help="Imagen a convertir a Gcode; se admiten todos los formatos compatibles con la biblioteca imageio de Python")
    argParser.add_argument("-o", "--output", type=argparse.FileType('w'), required=True, metavar="ARCHIVO",
        help="Archivo en el que guardar el resultado del código G")
    argParser.add_argument("--dot-output", type=argparse.FileType('w'), metavar="ARCHIVO",
        help="Archivo opcional en el que guardar el grafo (en formato DOT) generado durante un paso intermedio de la generación de código G")
    argParser.add_argument("-e", "--edges", type=str, metavar="MODO",
        help="Considerar el archivo de entrada ya como una matriz de bordes, no como una imagen de la que detectar los bordes. MODO debe ser 'white' o 'black', que es el color de los bordes en la imagen. La imagen solo debe estar formada por píxeles blancos o negros.")
    argParser.add_argument("-t", "--threshold", type=int, default=32, metavar="VALOR",
        help="El umbral en el rango (0,255) por encima del cual considerar un píxel como parte de un borde (después de aplicar Sobel a la imagen o al leer los bordes del archivo con la opción --edges)")

    argParser.add_argument("-s", "--scale", type=float, default=1.0, metavar="VALOR",
        help="Factor de escala para las coordenadas del código G generado")
    
    argParser.add_argument("--simplify", type=float, default=0.5, metavar="VALOR",
        help="Factor de simplificación para reducir puntos en líneas (mayor valor = menos puntos)")

    argParser.parse_args(namespace=namespace)

    if namespace.edges is not None and namespace.edges not in ["white", "black"]:
        argParser.error("mode for --edges should be `white` or `black`")
    if namespace.threshold <= 0 or namespace.threshold >= 255:
        argParser.error("value for --threshold should be in range (0,255)")

def main():
    class Args: pass
    parseArgs(Args)

    image = imageio.imread(Args.input)
    if Args.edges is None:
        edges = sobel(image)
    elif Args.edges == "black":
        edges = np.invert(image)
    else: # Args.edges == "white"
        edges = image
    edges = convertToBinaryEdges(edges, Args.threshold)

    converter = EdgesToGcode(edges)
    graph = converter.buildGraph()

    if Args.dot_output is not None:
        graph.saveAsDotFile(Args.dot_output)
    graph.saveAsGcodeFile(Args.output, Args.scale)


if __name__ == "__main__":
    main()

import numpy as np
import imageio
from PIL import Image

def convert_image_to_gcode(
    image_input,
    output_gcode="out.nc",
    edges_mode=None,
    threshold=32,
    scale=1.0,
    simplify=0.5,
    dot_output=None
):
    """
    Convierte una imagen a G-code usando la lógica de este script.
    - image_input: puede ser una ruta (str) o un objeto PIL.Image
    - output_gcode: nombre del archivo G-code de salida (por defecto 'out.nc')
    - edges_mode: 'white' o 'black' (similar a --edges en tu script)
    - threshold: umbral para convertir a binario (similar a --threshold)
    - scale: factor de escala (similar a -s)
    - simplify: factor de simplificación (similar a --simplify)
    - dot_output: si quieres guardar el grafo en formato DOT (opcional, o None)
    """

    # 1) Cargar la imagen (ruta o PIL)
    if isinstance(image_input, str):
        # Si es ruta, la leemos con imageio
        image = imageio.imread(image_input)
    elif isinstance(image_input, Image.Image):
        # Si ya es PIL.Image, convertimos a numpy array
        image = np.array(image_input)
    else:
        raise ValueError("image_input debe ser una ruta (str) o PIL.Image.")

    # 2) Si edges_mode es None, usar sobel; si es 'black' o 'white', invertimos o no
    if edges_mode is None:
        # Usa Sobel
        from scipy import ndimage
        edges = sobel(image)  # sobel() está definido en este mismo script
    elif edges_mode == "black":
        # Invertir la imagen (asume bordes en negro sobre fondo blanco)
        edges = np.invert(image)
    elif edges_mode == "white":
        # Usa la imagen tal cual, asumiendo bordes en blanco
        edges = image
    else:
        raise ValueError("edges_mode debe ser 'black', 'white' o None.")

    # 3) Umbralizar la imagen para obtener un array booleano de bordes
    edges_bool = convertToBinaryEdges(edges, threshold)

    # 4) Construir el grafo a partir de los bordes
    converter = EdgesToGcode(edges_bool)
    graph = converter.buildGraph()

    # 5) Guardar DOT opcional
    if dot_output is not None:
        with open(dot_output, "w") as f:
            graph.saveAsDotFile(f)

    # 6) Guardar G-code
    with open(output_gcode, "w") as f:
        graph.saveAsGcodeFile(f, scale=scale, simplify_factor=simplify)

    print(f"G-code guardado en: {output_gcode}")
