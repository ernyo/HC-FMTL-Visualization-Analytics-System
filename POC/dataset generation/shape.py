import numpy as np
import math
from PIL import Image, ImageDraw
from IPython.display import display

COLOR_TO_RGB = {"red":(255,0,0),"green":(0,200,0),"blue":(0,128,255),"yellow":(255,200,0)}

class Shape:
    def __init__(self):
        self.color_to_rgb = COLOR_TO_RGB

    def draw_shape(self, draw, shape_name, color, bbox):
        color_rgb = self.color_to_rgb[color]
        if shape_name == "square":
            self.draw_square(draw, color_rgb, bbox)
        elif shape_name == "circle":
            self.draw_circle(draw, color_rgb, bbox)
        elif shape_name == "triangle":
            self.draw_triangle(draw, color_rgb, bbox)
        elif shape_name == "star":
            self.draw_star(draw, color_rgb, bbox)
        else:
            raise ValueError(f"Unknown shape: {shape_name}")

    def draw_square(self, draw, color, bbox):
        draw.rectangle(bbox, fill=color)

    def draw_circle(self, draw, color, bbox):
        draw.ellipse(bbox, fill=color)

    def draw_triangle(self, draw, color, bbox):
        x0,y0,x1,y1=bbox
        w=x1-x0
        draw.polygon([(x0+w//2,y0),(x1,y1),(x0,y1)], fill=color)

    def draw_star(self, draw, color, bbox):
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        width = x1 - x0
        height = y1 - y0
        outer_radius = min(width, height) // 2
        inner_radius = outer_radius // 2
        
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = cx + radius * math.cos(angle)
            y = cy - radius * math.sin(angle)
            points.append((x, y)) 
        draw.polygon(points, fill=color)

    def display_shape(self, shape_name, color, bbox):
        W = H = 128
        im = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(im)
        self.draw_shape(draw, shape_name, color, bbox)
        display(im)