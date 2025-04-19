#!/usr/bin/env python3
"""
Convert an image to G‑code by thresholding to a PBM, tracing with the Potrace CLI into SVG (with optional simplification), and generating G‑code via svg2gcode with circular interpolation support.
This version incluye detección de discontinuidades internas para usar G0 en saltos de sub‑contornos y evita cortar al cerrar contornos.
"""
import argparse
import subprocess
import os
import math
from PIL import Image
import numpy as np
import svgpathtools as svgpath
from typing import List, Tuple, Optional


def bitmap_to_pbm(input_path: str, pbm_path: str, threshold: int):
    img = Image.open(input_path).convert('L')
    bitmap = img.point(lambda x: 1 if x > threshold else 0, mode='1')
    bitmap.save(pbm_path)
    print(f"Saved PBM to {pbm_path}")


def pbm_to_svg(pbm_path: str, svg_path: str, potrace_opts: str):
    cmd = f"potrace {potrace_opts} -o \"{svg_path}\" \"{pbm_path}\""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print(f"Saved SVG to {svg_path}")


class CircleFit:
    @staticmethod
    def fit_circle_to_points(points: List[complex]) -> Tuple[Optional[complex], float]:
        pts = np.array([(p.real, p.imag) for p in points])
        if len(pts) < 3:
            return None, 0
        x, y = pts[:,0], pts[:,1]
        A = np.column_stack((x, y, np.ones(len(x))))
        b = -(x**2 + y**2)
        try:
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = -c[0]/2, -c[1]/2
            r = math.sqrt((c[0]**2 + c[1]**2)/4 - c[2])
            return complex(cx, cy), r
        except:
            return None, 0

    @staticmethod
    def evaluate_circle_fit(points: List[complex], center: complex, radius: float) -> float:
        if not points or radius <= 0:
            return float('inf')
        distances = [abs(p - center) for p in points]
        mean_dev = sum(abs(d - radius) for d in distances) / len(distances)
        return mean_dev / radius


def fit_arc_to_segment(segment, samples=12, error_threshold=0.05,
                       max_radius_multiplier=3.0, min_arc_angle=5.0) -> Optional[Tuple]:
    points = [segment.point(t/samples) for t in range(samples + 1)]
    seg_len = abs(points[-1] - points[0])
    center, radius = CircleFit.fit_circle_to_points(points)
    if center is None or radius <= 0:
        return None
    error = CircleFit.evaluate_circle_fit(points, center, radius)
    if error > error_threshold or radius > seg_len * max_radius_multiplier:
        return None
    v1 = points[0] - center
    v2 = points[-1] - center
    cosang = max(-1.0, min(1.0, (v1.real*v2.real + v1.imag*v2.imag)/(abs(v1)*abs(v2))))
    arc_angle = math.degrees(math.acos(cosang))
    if arc_angle < min_arc_angle:
        return None
    # Dirección
    start, mid, end = points[0], points[len(points)//2], points[-1]
    cross_z = (mid-start).real*(end-start).imag - (mid-start).imag*(end-start).real
    direction = "G3" if cross_z > 0 else "G2"
    if radius * math.radians(arc_angle) > seg_len*2:
        return None
    return (start, end, center, radius, direction, arc_angle)


def segment_to_linear_gcode(segment, f, steps=10):
    for t in range(1, steps + 1):
        p = segment.point(t/steps)
        f.write(f"G1 X{p.real:.4f} Y{p.imag:.4f}\n")
    return segment.end


def svg_to_gcode_with_arcs(svg_path: str, gcode_path: str, feed_rate=1000,
                          z_safe=5.0, z_cut=-1.0, arc_samples=12,
                          arc_error_threshold=0.05, max_radius_multiplier=3.0,
                          min_arc_angle=5.0, verbose=False):
    paths, _ = svgpath.svg2paths(svg_path)
    total_segments = arc_segments = linear_segments = contour_transitions = 0

    with open(gcode_path, 'w') as f:
        f.write("; G-code generado con detección de discontinuidades y arcos\n")
        f.write("G21 ; unidades mm\nG90 ; posicionamiento absoluto\nG17 ; plano XY\n")
        f.write(f"F{feed_rate} ; feed rate\nG0 Z{z_safe} ; altura segura inicial\n\n")

        for p_idx, path in enumerate(paths):
            if not path:
                continue
            start_pt = path[0].start
            f.write(f"; Path {p_idx+1}\n")
            f.write(f"G0 Z{z_safe} ; Lift safe\n")
            f.write(f"G0 X{start_pt.real:.4f} Y{start_pt.imag:.4f} ; Rapid al start\n")
            f.write(f"G1 Z{z_cut:.4f} F{feed_rate//2} ; Plunge\n")
            f.write(f"F{feed_rate} ; Restore feed\n")
            current_point = start_pt

            for idx, segment in enumerate(path):
                total_segments += 1
                end = segment.end
                # 1) Discontinuidad interna: nuevo sub‑contorno
                if idx > 0 and abs(segment.start - current_point) > 1e-6:
                    contour_transitions += 1
                    f.write(f"G0 Z{z_safe} ; Lift sub‑contour\n")
                    f.write(f"G0 X{segment.start.real:.4f} Y{segment.start.imag:.4f} ; Rapid salto\n")
                    f.write(f"G1 Z{z_cut:.4f} F{feed_rate//2} ; Plunge\n")
                    f.write(f"F{feed_rate} ; Restore feed\n")
                    current_point = segment.start
                # 2) Si es línea
                if isinstance(segment, svgpath.Line):
                    # Evitar corte al cerrar contorno
                    if idx == len(path)-1 and abs(end - start_pt) < 1e-6:
                        f.write(f"G0 X{end.real:.4f} Y{end.imag:.4f} ; Cerrar sin cortar\n")
                    else:
                        f.write(f"G1 X{end.real:.4f} Y{end.imag:.4f}\n")
                    current_point = end
                    linear_segments += 1
                    continue
                # 3) Intento de arco
                arc_fit = fit_arc_to_segment(segment, arc_samples,
                                            arc_error_threshold,
                                            max_radius_multiplier,
                                            min_arc_angle)
                if arc_fit:
                    _, end_pt, center, radius, direction, angle = arc_fit
                    seg_len = abs(end - current_point)
                    if radius < seg_len*0.1 or radius > seg_len*max_radius_multiplier:
                        current_point = segment_to_linear_gcode(segment, f)
                        linear_segments += 1
                    else:
                        i_val = center.real - current_point.real
                        j_val = center.imag - current_point.imag
                        if abs(i_val) > radius*2 or abs(j_val) > radius*2:
                            current_point = segment_to_linear_gcode(segment, f)
                            linear_segments += 1
                        else:
                            f.write(f"{direction} X{end_pt.real:.4f} Y{end_pt.imag:.4f} I{i_val:.4f} J{j_val:.4f} ; Arc {angle:.1f}°\n")
                            arc_segments += 1
                else:
                    current_point = segment_to_linear_gcode(segment, f)
                    linear_segments += 1
                current_point = end

            # Fin de path
            f.write(f"G0 Z{z_safe} ; End path lift\n\n")
            contour_transitions += 1

        f.write(f"G0 Z{z_safe} ; Safe end\nM2 ; Program end\n")
    print(f"G-code guardado en {gcode_path}: {arc_segments} arcos, {linear_segments} lineales, {contour_transitions} contornos")


def main():
    parser = argparse.ArgumentParser(description="Image → PBM → SVG → G‑code con arcos y discontinuidades")
    parser.add_argument('input', help="Archivo de entrada (imagen)")
    parser.add_argument('-t','--threshold', type=int, default=128, help="Umbral 0–255")
    parser.add_argument('--pbm', default=None, help="Ruta PBM intermedio")
    parser.add_argument('--svg', default=None, help="Ruta SVG salida")
    parser.add_argument('-o','--output', default=None, help="Ruta G-code salida")
    parser.add_argument('--potrace-opts', default='-s', help="Opciones Potrace")
    parser.add_argument('--feed-rate', type=int, default=1000, help="Feed rate mm/min")
    parser.add_argument('--z-safe', type=float, default=5.0, help="Altura segura mm")
    parser.add_argument('--z-cut', type=float, default=-1.0, help="Profundidad mm")
    parser.add_argument('--arc-samples', type=int, default=12, help="Muestras para arcos")
    parser.add_argument('--arc-error', type=float, default=0.05, help="Error máx arcos")
    parser.add_argument('--max-radius', type=float, default=3.0, help="Radio máx mult.")
    parser.add_argument('--min-angle', type=float, default=5.0, help="Ángulo mín arcos")
    parser.add_argument('--simplify', action='store_true', help="Simplificar SVG")
    parser.add_argument('--verbose', action='store_true', help="Verbose")
    args = parser.parse_args()

    base, _ = os.path.splitext(args.input)
    pbm = args.pbm or f"{base}.pbm"
    svg = args.svg or f"{base}.svg"
    out = args.output or f"{base}.nc"

    bitmap_to_pbm(args.input, pbm, args.threshold)
    if args.simplify:
        opts = args.potrace_opts + ' -O 1.0 -t 50'
    else:
        opts = args.potrace_opts
    pbm_to_svg(pbm, svg, opts)
    svg_to_gcode_with_arcs(svg, out,
                          feed_rate=args.feed_rate,
                          z_safe=args.z_safe,
                          z_cut=args.z_cut,
                          arc_samples=args.arc_samples,
                          arc_error_threshold=args.arc_error,
                          max_radius_multiplier=args.max_radius,
                          min_arc_angle=args.min_angle,
                          verbose=args.verbose)

if __name__ == '__main__':
    main()
