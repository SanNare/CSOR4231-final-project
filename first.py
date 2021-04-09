from graph_tool.all import *

g = load_graph("search_example.xml")
name = g.vp["name"]
weight = g.ep["weight"]
pos = g.vp["pos"]
graph_draw(g, pos, vertex_text=name, vertex_font_size=12, vertex_shape="double_circle",
              vertex_fill_color="#729fcf", vertex_pen_width=3,
              edge_pen_width=weight, output="search_example.png")