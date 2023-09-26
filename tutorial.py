from manim import *
from custom import NeuralNetworkMobject

import networkx as nx
from networkx.generators import gnp_random_graph
import numpy as np

# 1
class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        circle.set_stroke(PINK, opacity=1)
        self.play(Create(circle))  # show the circle on screen

# 2
class TriangleToSquare(Scene):
    def construct(self):
        triangle = Triangle()
        triangle.set_fill(GREEN, opacity=0.5)
        triangle.set_stroke(GREEN, opacity=1)

        square = Square()
        square.set_fill(GREEN, opacity=0.5)
        square.set_stroke(GREEN, opacity=1)

        self.play(Create(triangle))
        self.play(Transform(triangle, square))
        self.play(FadeOut(square))

class AnimatedSquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        square = Square()  # create a square

        self.play(Create(square))  # show the square on screen
        self.play(square.animate.rotate(PI / 4))  # rotate the square
        self.play(ReplacementTransform(square, circle))  # transform the square into a circle
        self.play(circle.animate.set_fill(PINK, opacity=0.5))  # color the circle on screen

class DifferentRotations(Scene):
    def construct(self):
        left_square = Square(color=BLUE, fill_opacity=0.7).shift(2 * LEFT)
        right_square = Square(color=GREEN, fill_opacity=0.7).shift(2 * RIGHT)
        self.play(left_square.animate.rotate(PI), Rotate(right_square, angle=PI), run_time=2)
        self.wait()

# 3
class CircleAnimations(Scene):
    def construct(self):
        circle = Circle(color=GREEN, fill_opacity=0.5, stroke_opacity=1)
        self.play(Create(circle))
        self.play(circle.animate.shift(2 * LEFT))
        self.play(circle.animate.set_fill(PINK, opacity=0.5))
        self.play(circle.animate.set_stroke(PINK, opacity=1))
        self.play(FadeOut(circle))

# 4
class TextToVector(Scene):
    def construct(self):
        quote = Text("“If more of us valued food and cheer and song above\nhoarded gold, it would be a merrier world.”\n― J.R.R. Tolkien", gradient=('#7734eb', '#eba234'), line_spacing=1.5, color=WHITE)
        quote.scale(0.5)
        
        vector = DecimalMatrix([[0.93, -0.1, 0.65, 0.21, 1.23, 1.04]])

        self.play(Write(quote))

        # wait is like a delay you canset before another animation can play
        self.wait(1)
        self.play(Transform(quote, vector))
        self.play(FadeOut(vector))

# 5
class HtmlCode(Scene):
    def construct(self):
        code = Code('./assets/markups/sample.html',
            insert_line_no=True,
            background='window',
            background_stroke_color=WHITE,
            background_stroke_width=1,
            language='html',
            font='Consolas',
            font_size=12,
            line_spacing=1,
            style='material')
        
        self.play(Write(code, run_time=5))
        self.wait(1)

# 6
class MorphingHeaders(Scene):
    def construct(self):
        with register_font('./assets/fonts/static/NunitoSans_10pt-Regular.ttf'):
            text_1 = Text("Deep Learning/Machine Learning", font="Nunito Sans 10pt")
            text_1.scale(0.5)

            text_2 = Text("Machine Learning with Graphs", font="Nunito Sans 10pt")
            text_2.scale(0.5)

            text_3 = Text("Natural Language Processing", font="Nunito Sans 10pt")
            text_3.scale(0.5)

            text_4 = Text("Data Visualization & Analysis", font="Nunito Sans 10pt")
            text_4.scale(0.5)

            text_5 = Text("Client & Server Side Web Development", font="Nunito Sans 10pt")
            text_5.scale(0.5)

            self.play(Write(text_1))
            self.wait(1)
            self.play(ReplacementTransform(text_1, text_2))
            self.wait(1)
            self.play(ReplacementTransform(text_2, text_3))
            self.wait(1)
            self.play(ReplacementTransform(text_3, text_4))
            self.wait(1)
            self.play(ReplacementTransform(text_4, text_5))
            self.wait(1)
            self.play(FadeOut(text_5))

# 7 
class MorphingNeuralNetwork(Scene):
    def construct(self):
        nn = NeuralNetworkMobject([4, 3, 2, 2, 3, 4])
        nn.label_inputs('X')
        nn.label_hidden_layers('A[l]')
        nn.label_outputs('h_{\\theta}(X)')
        nn.scale(0.75)

        self.play(Write(nn))

# 8
class ThreeDLinearRegression(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            z_range=[-6, 6, 1],
            x_length=8,
            y_length=8,
            z_length=8
        )

        self.add(axes)
        self.wait()
        # phi moves the camera along the y-axis or up and down
        self.move_camera(phi=60 * DEGREES)
        self.wait()

        # theta moves the camera along the x-axis or left and right
        self.move_camera(theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=PI / 10, about="theta")


class SimpleGraph(Scene):
    def construct(self):
        # create graph
        random_graph = gnp_random_graph(n=20, p=0.6)

        # turn nods and edges of networkx graph to lists
        nodes = list(nx.nodes(random_graph))
        edges = list(nx.edges(random_graph))

        # pass node and edge lists to Graph mobject
        # g = Graph(nodes, edges, labels=True, label_fill_color=PURPLE, layout='shell')
        # g = Graph(nodes, edges, labels=True, layout='spring')
        g = Graph(nodes, edges, layout='spring')

        # create move_to animations for each node 
        move_nodes = [g[node].animate.move_to(
            5 * RIGHT * np.cos(index / 7 * PI) + 
            3 * UP * np.sin(index / 7 * PI)) 
            for index, node in enumerate(g.vertices)]
        # color_nodes = [g[node].animate.set_fill(PURPLE, opacity=0.5) for _, node in enumerate(g.vertices)]
        # color_nodes_stroke = [g[node].animate.set_stroke(PURPLE, opacity=1) for _, node in enumerate(g.vertices)]
        
        # color_labels = g.animate.set_color(BLACK)
        
        print(f'center of graph g: {g.get_center()}')
        print(f'first 5 vertices of graph g: {list(g.vertices)[:5]}')
        
        self.play(Create(g))
        self.wait()

        # coordinates in the z-axis
        # "dereference" node_animations to indicate that this
        # is a series of animations
        self.play(*move_nodes)
        self.play(g[14].animate.set_stroke(BLUE, opacity=1, width=1.5))
        self.play(g[14].animate.set_fill(BLUE, opacity=1))

        self.wait()

class Graph2(Scene):
    def construct(self):
        # create graph
        random_graph = gnp_random_graph(n=20, p=0.3)

        # turn nods and edges of networkx graph to lists
        nodes = list(nx.nodes(random_graph))
        edges = list(nx.edges(random_graph))

        g = Graph(nodes, edges, layout='spring', labels=True, label_fill_color=WHITE)
        g[0].move_to(LEFT + UP)
        g[1].move_to(RIGHT + UP)
        g[2].move_to(LEFT + DOWN)
        g[3].move_to(RIGHT + DOWN)

        g.set_stroke(RED)
        g.set_color(BLACK)

        node_animations = [g[3].animate.set_stroke(RED, opacity=1, width=5), g[3].animate.set_fill(RED, opacity=0.5)]
        self.play(Create(g))
        self.wait()
        self.play(*node_animations)

class Graph3(Scene):
    def construct(self):
        # create graph
        random_graph = gnp_random_graph(n=20, p=0.3)

        # turn nods and edges of networkx graph to lists
        nodes = list(nx.nodes(random_graph))
        edges = list(nx.edges(random_graph))

        g = Graph(nodes, edges, layout='spring')

class ThreeDLinearRegression2(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=6,
            y_length=6,
            z_length=6
        )

        self.play(GrowFromCenter(axes))
        
        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES)
    
        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")
        

        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        self.stop_ambient_camera_rotation(about="theta")
        self.wait(1)

class ThreeDLinearRegression3(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=6,
            y_length=6,
            z_length=6
        )

        # theta sets the camera along the x-axis or left and right
        # phi sets the camera along the y-axis or up and down
        # doing this sets camera leftward and upward simultaneously
        self.set_camera_orientation(theta=-45 * DEGREES, phi=60 * DEGREES)
        self.play(GrowFromCenter(axes))
    
        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")
        
        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        self.stop_ambient_camera_rotation(about="theta")
        self.wait(1)