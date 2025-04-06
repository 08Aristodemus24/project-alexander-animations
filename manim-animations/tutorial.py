from manim import *
from manim_ml.neural_network import *

import networkx as nx
from networkx.generators import gnp_random_graph
from sklearn.preprocessing import StandardScaler
import numpy as np

import joblib

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

class ThreeDLinearRegression4(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=6,
            y_length=6,
            z_length=6
        )

        square = Square()
        self.play(Create(square))
        self.play(square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1))

        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES)
        self.camera.frame.animate

        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")

        self.play(ReplacementTransform(square, axes))
        
        
        
        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        self.stop_ambient_camera_rotation(about="theta")
        self.wait(1)

# current
class ThreeDLinearRegression5(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=10, 
            y_length=10,
            z_length=10
        )

        square = Square()

        # place x, y, z coordinates of dot
        point_a = Dot3D(point=[1, -3, 0], radius=0.05, color=WHITE)
        point_b = Dot3D(point=[-1, 1, 1], radius=0.05, color=WHITE)
        point_c = Dot3D(point=[1, 0, 2], radius=0.05, color=WHITE)

        self.play(Create(square))

        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES, added_anims=[square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1)])

        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")

        # in between begin and end ambient camera rotation is where we 
        # place all our animations that will be simultaneous with the rotations
        self.play(*[ReplacementTransform(square, axes), Create(point_a), Create(point_b), Create(point_c)])
        # self.play()
        # self.play()
        # self.play()
        
        
        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        self.stop_ambient_camera_rotation(about="theta")
        self.wait(1)

class ThreeDLinearRegression6(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=10, 
            y_length=10,
            z_length=10
        )

        square = Square()

        # load x, y, z coordinates for dot mobject using test data
        # exclude all y values greater than 50
        data = np.loadtxt('./data/test_data.txt', dtype=np.float32, delimiter='\t')
        data = data[data[:, 2] <= 10, :]

        # sample 50 data points only for optimizing run time
        sample_indeces = np.random.choice(np.arange(data.shape[0]), size=50, replace=False)
        
        # create Dot mobject with 200 sampled data points
        # then animate them each using Create
        points = [Dot3D(point=example, radius=0.05, color=WHITE) for example in data[sample_indeces, :]]

        self.play(Create(square))
        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES, added_anims=[square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1)])

        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")

        # in between begin and end ambient camera rotation is where we 
        # place all our animations that will be simultaneous with the rotations
        animation_a = [ReplacementTransform(square, axes)]
        animation_a.extend([Create(point) for point in points])
        self.play(*animation_a)
        
        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        self.stop_ambient_camera_rotation(about="theta")
        self.wait(1)

# final design
class ThreeDLinearRegression7(ThreeDScene):
    def construct(self):
        # load x, y, z coordinates for dot mobject using test data
        # exclude all y values greater than 50
        data = np.loadtxt('./data/test_data.txt', dtype=np.float32, delimiter='\t')
        data = data[data[:, 2] <= 10, :]

        # sample 50 data points only for optimizing run time
        sample_indeces = np.random.choice(np.arange(data.shape[0]), size=50, replace=False)

        # loaded dataset has not been normalized yet so normalize here
        test_scaler = StandardScaler()
        test_scaler.fit(data[sample_indeces, :])
        X_tests_normed = test_scaler.transform(data[sample_indeces])

        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=10, 
            y_length=10,
            z_length=10
        )

        square = Square()
        
        # create Dot mobject with 200 sampled data points
        # then animate them each using Create
        points = [Dot3D(point=example, radius=0.05, color=WHITE) for example in X_tests_normed]

        # # load also the learned objective function by our linear model
        # lrm_a = joblib.load('./models/lrm_a.pkl')
        # THETA_1, THETA_2 = lrm_a.coef_
        # BETA = lrm_a.intercept_

        # the function callback must return the x, y, z coordinates
        line = ParametricFunction(lambda t: np.array([
            1.2 * np.cos(t),
            1.2 * np.sin(t),
            t * 0.05
        ]), t_range=[-5 * TAU, 5 * TAU])

        self.play(Create(square))
        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES, added_anims=[square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1)])

        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")

        # in between begin and end ambient camera rotation is where we 
        # place all our animations that will be simultaneous with the rotations
        animation_a = [ReplacementTransform(square, axes)]
        animation_a.extend([Create(point) for point in points])
        self.play(*animation_a)
        self.play(Create(line))
        
        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        self.stop_ambient_camera_rotation(about="theta")
        self.wait(1)

class Graph4(Scene):
    def construct(self):
        edges = []
        partitions = []
        c = 0
        layers = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer

        for n_nodes in layers:
            partitions.append(list(range(c + 1, c + n_nodes + 1)))
            c += n_nodes

        # create the edges of neural network
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        # create the nodes of the neural network
        vertices = np.arange(1, sum(layers) + 1)

        # note graph nodes start from 1 in this case
        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=5,
            vertex_config={'radius': 0.20},
        )

        square = Square()
        

        self.play(*[Create(square), square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1)])
        self.play(ReplacementTransform(square, graph))
        # move nodes up by 25%
        self.play(graph[1].animate.move_to(graph[1].get_center() + UP * 0.25))
        self.play(graph[1].animate.move_to(graph[1].get_center() + DOWN * 0.5))
        self.play(graph[1].animate.move_to(graph[1].get_center() + UP * 0.5))
        self.play(graph[1].animate.move_to(graph[1].get_center() + DOWN * 0.5))
        self.play(graph[1].animate.move_to(graph[1].get_center() + UP * 0.5))
        self.play(graph[1].animate.move_to(graph[1].get_center() + DOWN * 0.5))

        self.wait(3)

class Graph5(Scene):
    def construct(self):
        edges = []
        partitions = []
        c = 0
        layers = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer

        for n_nodes in layers:
            partitions.append(list(range(c + 1, c + n_nodes + 1)))
            c += n_nodes

        # create the edges of neural network
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        # create the nodes of the neural network
        vertices = np.arange(1, sum(layers) + 1)

        # note graph nodes start from 1 in this case
        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=5,
            vertex_config={'radius': 0.20},
        )

        square = Square()
        

        self.play(*[Create(square), square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1)])
        self.play(ReplacementTransform(square, graph))

        # move nodes up by 25%
        a1 = [graph[node].animate.move_to(graph[node].get_center() + UP * 0.25) for node in graph.vertices if node in [1, 2, 3, 4, 5]]
        self.play(*a1)

        a2 = [graph[node].animate.move_to(graph[node].get_center() + DOWN * 0.5) for node in graph.vertices if node in [1, 2, 3, 4, 5]]
        self.play(*a2)

        a3 = [graph[node].animate.move_to(graph[node].get_center() + UP * 0.5) for node in graph.vertices if node in [1, 2, 3, 4, 5]]
        self.play(*a3)

        self.play(*a2)
        self.play(*a3)
        self.play(*a2)
        self.play(*a3)

        self.wait(3)

# promising
class Skills(ThreeDScene):
    def construct(self):
        # Natural language processing
        quote = Text("“If more of us valued food and cheer and song above\nhoarded gold, it would be a merrier world.”\n― J.R.R. Tolkien", line_spacing=1.5, color=WHITE)
        quote.scale(0.5)
        
        vector = DecimalMatrix([[0.93, -0.1, 0.65, 0.21, 1.23, 1.04]])

        dot = Dot3D()

        # Data Visualization and anlaysis
        # load x, y, z coordinates for dot mobject using test data
        # exclude all y values greater than 50
        data = np.loadtxt('./data/test_data.txt', dtype=np.float32, delimiter='\t')
        data = data[data[:, 2] <= 10, :]

        # sample 50 data points only for optimizing run time
        sample_indeces = np.random.choice(np.arange(data.shape[0]), size=50, replace=False)

        # loaded dataset has not been normalized yet so normalize here
        test_scaler = StandardScaler()
        test_scaler.fit(data[sample_indeces, :])
        X_tests_normed = test_scaler.transform(data[sample_indeces])

        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=10, 
            y_length=10,
            z_length=10
        )
        
        # create Dot mobject with 200 sampled data points
        # then animate them each using Create
        points = [Dot3D(point=example, radius=0.05, color=WHITE) for example in X_tests_normed]

        # the function callback must return the x, y, z coordinates
        line = ParametricFunction(lambda t: np.array([
            1.2 * np.cos(t),
            1.2 * np.sin(t),
            t * 0.05
        ]), t_range=[-5 * TAU, 5 * TAU])

        # client and server side web development
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
        
        init_theta = self.camera.get_theta()
        init_phi = self.camera.get_phi()
        init_gamma = self.camera.get_gamma()

        # Natural language processing
        self.play(Write(quote))

        # wait is like a delay you canset before another animation can play
        self.wait(0.5)
        self.play(ReplacementTransform(quote, vector))
        self.wait(1)

        # Data analysis & visualization
        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        # simultaneously replace the previous mobject with axes, 
        # with the creation of points in the axes
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES, added_anims=[ReplacementTransform(vector, dot)])

        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # in between begin and end ambient camera rotation is where we 
        # place all our animations that will be simultaneous with the rotations
        # self.begin_ambient_camera_rotation(90 * DEGREES, about="theta")
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")
        # self.play(*[ReplacementTransform(dot, axes)] + [Create(point) for point in points])
        self.play(Create(line))

        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        # reset camera angle to initial values
        self.move_camera(theta=init_theta, phi=init_phi, added_anims=[Uncreate(axes), Uncreate(line)] + [Uncreate(point) for point in points])
        self.stop_ambient_camera_rotation(about="theta")

        # client & server side web development
        self.play(Write(code, run_time=5))
        self.wait(1)
        
class MLAndDL(Scene):
    def construct(self):
        edges = []
        partitions = []
        c = 0
        layers = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer

        for n_nodes in layers:
            partitions.append(list(range(c + 1, c + n_nodes + 1)))
            c += n_nodes

        # create the edges of neural network
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        # create the nodes of the neural network
        vertices = np.arange(1, sum(layers) + 1)

        # note graph nodes start from 1 in this case
        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=5,
            vertex_config={'radius': 0.20},
        )

        square = Square()
        

        self.play(*[Create(square), square.animate.set_fill(PINK, opacity=0.5).set_stroke(PINK, opacity=1)])
        self.play(ReplacementTransform(square, graph))
        self.wait(3)

class MLAndDL2(ThreeDScene):
    def construct(self):
        layers_dims = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer
        # Theta * X + Beta
        linear_func = MathTex(r"h_\Theta(X) = \sigma(\Theta X + \mathrm{B})")
        whole_op = MathTex(r"""
            h_\Theta(X) = 
            \sigma(\begin{bmatrix}
            \Theta^{(0)}_{1, 1} & \Theta^{(0)}_{1, 2} & \Theta^{(0)}_{1, 3}\\
            \Theta^{(0)}_{2, 1} & \Theta^{(0)}_{2, 2} & \Theta^{(0)}_{2, 3}\\
            \Theta^{(0)}_{3, 1} & \Theta^{(0)}_{3, 2} & \Theta^{(0)}_{3, 3}\\
            \Theta^{(0)}_{4, 1} & \Theta^{(0)}_{4, 2} & \Theta^{(0)}_{4, 3}\\
            \end{bmatrix}
            \cdot
            \begin{bmatrix}
            X^{(0)}_1 & X^{(1)}_1 & \cdots & X^{(m - 1)}_1 \\
            X^{(0)}_2 & X^{(1)}_2 & \cdots & X^{(m - 1)}_2 \\
            X^{(0)}_3 & X^{(1)}_3 & \cdots & X^{(m - 1)}_3 \\
            \end{bmatrix}
            +
            \begin{bmatrix}
            \mathrm{B}^{(0)}_{1, 0} \\
            \mathrm{B}^{(0)}_{2, 0} \\
            \mathrm{B}^{(0)}_{3, 0} \\
            \mathrm{B}^{(0)}_{4, 0} \\
            \end{bmatrix})
        """)

        # this declaration is akin to sequential in tensorflow
        nn = NeuralNetwork([FeedForwardLayer(num_nodes=layer_dim, rectangle_stroke_width=0) for layer_dim in layers_dims], layer_spacing=0.25)
        
        self.play(Write(linear_func))
        self.wait(1)
        self.play(linear_func.animate.become(whole_op))
        self.play(FadeOut(whole_op))
        self.wait(1)
        self.play(Create(nn))
        self.wait(1)
        self.play(nn.make_forward_pass_animation(run_time=5))
        self.wait(1)

class MLAndDL3(ThreeDScene):
    def construct(self):
        layers_dims = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer

        # Theta * X + Beta
        linear_func = MathTex(r"h_\Theta(X) = \sigma(\Theta X + \mathrm{B})")
        whole_op = MathTex(r"""
            h_\Theta(X) = 
            \sigma(\begin{bmatrix}
            \Theta^{(0)}_{1, 1} & \Theta^{(0)}_{1, 2} & \Theta^{(0)}_{1, 3}\\
            \Theta^{(0)}_{2, 1} & \Theta^{(0)}_{2, 2} & \Theta^{(0)}_{2, 3}\\
            \Theta^{(0)}_{3, 1} & \Theta^{(0)}_{3, 2} & \Theta^{(0)}_{3, 3}\\
            \Theta^{(0)}_{4, 1} & \Theta^{(0)}_{4, 2} & \Theta^{(0)}_{4, 3}\\
            \end{bmatrix}
            \cdot
            \begin{bmatrix}
            X^{(0)}_1 & X^{(1)}_1 & \cdots & X^{(m - 1)}_1 \\
            X^{(0)}_2 & X^{(1)}_2 & \cdots & X^{(m - 1)}_2 \\
            X^{(0)}_3 & X^{(1)}_3 & \cdots & X^{(m - 1)}_3 \\
            \end{bmatrix}
            +
            \begin{bmatrix}
            \mathrm{B}^{(0)}_{1, 0} \\
            \mathrm{B}^{(0)}_{2, 0} \\
            \mathrm{B}^{(0)}_{3, 0} \\
            \mathrm{B}^{(0)}_{4, 0} \\
            \end{bmatrix})
        """)

        # this declaration is akin to sequential in tensorflow
        nn = NeuralNetwork([FeedForwardLayer(num_nodes=layer_dim, rectangle_stroke_width=0) for layer_dim in layers_dims], layer_spacing=0.25)
        
        self.play(Write(linear_func))
        self.wait(1)
        # self.play(ReplacementTransform(linear_func, whole_op))
        # self.wait(1)
        # self.play(FadeOut(whole_op))
        # self.wait(1)
        self.play(ReplacementTransform(linear_func, nn))
        # self.wait(1)
        # self.play(nn.make_forward_pass_animation(run_time=5))
        # self.wait(1)

class MLAndDL4(ThreeDScene):
    def construct(self):
        # Theta * X + Beta
        linear_func = MathTex(r"h_\Theta(X) = \sigma(\Theta X + \mathrm{B})")
        whole_op = MathTex(r"""
            h_\Theta(X) = 
            \sigma(\begin{bmatrix}
            \Theta^{(0)}_{1, 1} & \Theta^{(0)}_{1, 2} & \Theta^{(0)}_{1, 3}\\
            \Theta^{(0)}_{2, 1} & \Theta^{(0)}_{2, 2} & \Theta^{(0)}_{2, 3}\\
            \Theta^{(0)}_{3, 1} & \Theta^{(0)}_{3, 2} & \Theta^{(0)}_{3, 3}\\
            \Theta^{(0)}_{4, 1} & \Theta^{(0)}_{4, 2} & \Theta^{(0)}_{4, 3}\\
            \end{bmatrix}
            \cdot
            \begin{bmatrix}
            X^{(0)}_1 & X^{(1)}_1 & \cdots & X^{(m - 1)}_1 \\
            X^{(0)}_2 & X^{(1)}_2 & \cdots & X^{(m - 1)}_2 \\
            X^{(0)}_3 & X^{(1)}_3 & \cdots & X^{(m - 1)}_3 \\
            \end{bmatrix}
            +
            \begin{bmatrix}
            \mathrm{B}^{(0)}_{1, 0} \\
            \mathrm{B}^{(0)}_{2, 0} \\
            \mathrm{B}^{(0)}_{3, 0} \\
            \mathrm{B}^{(0)}_{4, 0} \\
            \end{bmatrix})
        """)

        edges = []
        partitions = []
        c = 0
        layers = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer

        for n_nodes in layers:
            partitions.append(list(range(c + 1, c + n_nodes + 1)))
            c += n_nodes

        # create the edges of neural network
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        # create the nodes of the neural network
        vertices = np.arange(1, sum(layers) + 1)

        # note graph nodes start from 1 in this case
        graph = Graph(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=5,
            vertex_config={
                'radius': 0.2
            },
            edge_config={
                'buff': 1
            }
        )

        # modify each nodes fill
        for node in graph.vertices:
            graph[node].set_fill(WHITE, opacity=0.0)
            graph[node].set_stroke('#1d58e0', opacity=1, width=5)

        dot = Dot()

        
        self.play(Write(linear_func))
        self.wait(1)
        self.play(ReplacementTransform(linear_func, whole_op))
        self.wait(1)
        self.play(ReplacementTransform(whole_op, dot))
        self.wait(1)
        self.play(ReplacementTransform(dot, graph))
        self.wait(3)

class SkillsFinal(ThreeDScene):
    def construct(self):
        # MOBJECT HEADERS
        with register_font('./assets/fonts/static/NunitoSans_10pt-Regular.ttf'):
            text_1 = Text("Deep Learning & Machine Learning", font="Nunito Sans 10pt")
            text_1.scale(1)
            text_1.to_edge(UP, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER)

            text_2 = Text("Natural Language Processing", font="Nunito Sans 10pt")
            text_2.scale(1)
            text_2.to_edge(UP, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER)

            text_3 = Text("Data Analysis & Visualization", font="Nunito Sans 10pt")
            text_3.scale(1)
            text_3.set_opacity(0)
            text_3.to_edge(UP, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER)

            text_4 = Text("Client & Server Side Web Development", font="Nunito Sans 10pt")
            text_4.scale(0.75)
            text_4.to_edge(UP, buff=DEFAULT_MOBJECT_TO_EDGE_BUFFER)

        # MOBJECT MATH OBJECTS

        # Machine Learning & Deep Learning
        # Theta * X + Beta
        linear_func = MathTex(r"h_\Theta(X) = \mathrm{A}^{(L - 1)} = \sigma(\Theta^{(L - 2)} \cdots \sigma(\Theta^{(0)} X + \mathrm{B}^{(0)}) + \mathrm{B}^{(L - 2)})")
        # linear_func.next_to(text_1, DOWN, buff=1.5)

        whole_op = MathTex(r"""
            h_\Theta(X) = 
            A^{(L - 1)} = 
            \sigma(\begin{bmatrix}
                \Theta^{(L - 2)}_{1, 1} & \Theta^{(L - 2)}_{1, 2} & \cdots & \Theta^{(L - 2)}_{1, n^{(L - 2)}} \\
                \Theta^{(L - 2)}_{2, 1} & \Theta^{(L - 2)}_{2, 2} & \cdots & \Theta^{(L - 2)}_{2, n^{(L - 2)}} \\
                \vdots & \vdots & \ddots & \vdots \\
                \Theta^{(L - 2)}_{n^{(L - 1)}, 1} & \Theta^{(L - 2)}_{n^{(L - 1)}, 2} & \cdots & \Theta^{(L - 2)}_{n^{(L - 1)}, n^{(L - 2)}} \\
            \end{bmatrix} \\
            \cdots
            \sigma(\begin{bmatrix}
                    \Theta^{(0)}_{1, 1} & \Theta^{(0)}_{1, 2} & \cdots & \Theta^{(0)}_{1, n^{(0)}} \\
                    \Theta^{(0)}_{2, 1} & \Theta^{(0)}_{2, 2} & \cdots & \Theta^{(0)}_{2, n^{(0)}} \\
                    \vdots & \vdots & \ddots & \vdots \\
                    \Theta^{(0)}_{n^{(1)}, 1} & \Theta^{(0)}_{n^{(1)}, 2} & \cdots & \Theta^{(0)}_{n^{(1)}, n^{(0)}} \\
                    \end{bmatrix}
                    \cdot
                    \begin{bmatrix}
                    X^{(0)}_1 & X^{(1)}_1 & \cdots & X^{(m - 1)}_1 \\
                    X^{(0)}_2 & X^{(1)}_2 & \cdots & X^{(m - 1)}_2 \\
                    \vdots & \vdots & \ddots & \vdots \\
                    X^{(0)}_n & X^{(1)}_n & \cdots & X^{(m - 1)}_n \\
                    \end{bmatrix}
                    +
                    \begin{bmatrix}
                    \mathrm{B}^{(0)}_{1, 0} \\
                    \mathrm{B}^{(0)}_{2, 0} \\
                    \vdots \\
                    \mathrm{B}^{(0)}_{n^{(1)}, 0} \\
                \end{bmatrix}) \\
            +
            \begin{bmatrix}
            \mathrm{B}^{(L - 2)}_{1, 0} \\
            \mathrm{B}^{(L - 2)}_{2, 0} \\
            \vdots \\
            \mathrm{B}^{(L - 2)}_{n^{(L - 1)}, 0} \\
            \end{bmatrix})
        """)
        whole_op.scale(0.6)
        whole_op.next_to(text_1, DOWN, buff=0.75)

        dot_2d = Dot(color=WHITE, stroke_width=5, fill_opacity=0)
        dot_2d.scale(2.5)
        # dot_2d.next_to(text_1, DOWN, buff=1.5)

        edges = []
        partitions = []
        c = 0
        layers = [5, 4, 4, 3, 2, 3, 4, 4, 5]  # the number of neurons in each layer

        for n_nodes in layers:
            partitions.append(list(range(c + 1, c + n_nodes + 1)))
            c += n_nodes

        # create the edges of neural network
        for i, v in enumerate(layers[1:]):
                last = sum(layers[:i+1])
                for j in range(v):
                    for k in range(last - layers[i], last):
                        edges.append((k + 1, j + last + 1))

        # create the nodes of the neural network
        vertices = np.arange(1, sum(layers) + 1)

        # note graph nodes start from 1 in this case
        graph = Graph(
            vertices,
            edges,
            layout='circular',
            layout_scale=0.0075,
            vertex_config={
                'fill_color': WHITE,
                'fill_opacity': 0,
                'radius': 0.2,
                'stroke_width': 5,
                'stroke_color': WHITE,
                'stroke_opacity': 1
            },
            edge_config={
                'stroke_width': 2.5,
                'stroke_color': WHITE,
                'stroke_opacity': 0.75
            }
        )



        # Natural language processing
        quote = Text("“If more of us valued food and cheer and song above\nhoarded gold, it would be a merrier world.”\n― J.R.R. Tolkien", line_spacing=1.5, color=WHITE)
        quote.scale(0.5)
        
        
        vector = DecimalMatrix([[0.93, -0.1, 0.65, 0.21, 1.23, 1.04]])
        

        dot_3d = Dot3D()
        



        # Data visualization & analysis
        # load x, y, z coordinates for dot mobject using test data
        # exclude all y values greater than 50
        data = np.loadtxt('./data/test_data.txt', dtype=np.float32, delimiter='\t')
        data = data[data[:, 2] <= 10, :]

        # sample 50 data points only for optimizing run time
        sample_indeces = np.random.choice(np.arange(data.shape[0]), size=50, replace=False)

        # loaded dataset has not been normalized yet so normalize here
        test_scaler = StandardScaler()
        test_scaler.fit(data[sample_indeces, :])
        X_tests_normed = test_scaler.transform(data[sample_indeces])

        axes = ThreeDAxes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            z_range=[-10, 10, 1],
            x_length=10, 
            y_length=10,
            z_length=5,
            tips=False,
            axis_config={
                "include_numbers": True
            }
        )
        
        
        # create Dot mobject with 200 sampled data points
        # then animate them each using Create
        points = [Dot3D(point=example, radius=0.05, color=WHITE) for example in X_tests_normed]
        # for index, point in enumerate(points):
        #     points[index].next_to(text_3, DOWN, buff=1.5)

        # load also the learned objective function by our linear model
        lrm_a = joblib.load('./models/lrm_a.pkl')
        THETA_1, THETA_2 = lrm_a.coef_
        BETA = lrm_a.intercept_
        # print(THETA_1, THETA_2, BETA)

        # the function callback must return the x, y, z coordinates
        line = ParametricFunction(lambda t: np.array([
            THETA_1 * t,
            THETA_2 * t,
            BETA * t
        ]), t_range=[-1, 1])
        

        init_theta = self.camera.get_theta()
        init_phi = self.camera.get_phi()
        init_gamma = self.camera.get_gamma()



        # client and server side web development
        code = Code('./assets/markups/sample.html',
            insert_line_no=True,
            background='window',
            background_stroke_color=WHITE,
            background_stroke_width=1,
            language='html',
            font='Consolas',
            font_size=12,
            line_spacing=1,
            style='gruvbox-dark')
        code.next_to(text_4, DOWN, buff=0.75)


        
        # ANIMATIONS
        self.play(Write(text_1))
        self.play(Write(linear_func))
        self.wait(0.5)
        self.play(ReplacementTransform(linear_func, whole_op))
        self.wait(0.5)
        self.play(ReplacementTransform(whole_op, dot_2d))
        self.play(ReplacementTransform(dot_2d, graph), run_time=0.0001)
        self.play(graph.animate.change_layout('partite', partitions=partitions, layout_scale=5).next_to(text_1, DOWN, buff=0.75))
        
        # animate the nodes of the first layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [1, 2, 3, 4, 5]], run_time=0.275)


        # animate the edges of the first layer and remove animation of nodes in first layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [1, 2, 3, 4, 5]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(1, 6), (1, 7), (1, 8), (1, 9), 
        (2, 6), (2, 7), (2, 8), (2, 9),
        (3, 6), (3, 7), (3, 8), (3, 9),
        (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 6), (5, 7), (5, 8), (5, 9),]], run_time=0.275)
        
        # animate the nodes of the second layer and remove animation of edges in the first layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [6, 7, 8, 9]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(1, 6), (1, 7), (1, 8), (1, 9), 
        (2, 6), (2, 7), (2, 8), (2, 9),
        (3, 6), (3, 7), (3, 8), (3, 9),
        (4, 6), (4, 7), (4, 8), (4, 9),
        (5, 6), (5, 7), (5, 8), (5, 9),]], run_time=0.275)


        # animate the edges of the second layer and remove animation of nodes in second layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [6, 7, 8, 9]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(6, 10), (6, 11), (6, 12), (6, 13), 
        (7, 10), (7, 11), (7, 12), (7, 13),
        (8, 10), (8, 11), (8, 12), (8, 13),
        (9, 10), (9, 11), (9, 12), (9, 13),]], run_time=0.275)

        # animate the nodes of the third layer and remove animation of edges in the second layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [10, 11, 12, 13]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(6, 10), (6, 11), (6, 12), (6, 13), 
        (7, 10), (7, 11), (7, 12), (7, 13),
        (8, 10), (8, 11), (8, 12), (8, 13),
        (9, 10), (9, 11), (9, 12), (9, 13),]], run_time=0.275)


        # animate the edges of the third layer and remove animation of nodes in third layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [10, 11, 12, 13]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(10, 14), (10, 15), (10, 16), 
        (11, 14), (11, 15), (11, 16),
        (12, 14), (12, 15), (12, 16),
        (13, 14), (13, 15), (13, 16),]], run_time=0.275)

        # animate the nodes of the 4th layer and remove animation of edges in the third layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [14, 15, 16]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(10, 14), (10, 15), (10, 16), 
        (11, 14), (11, 15), (11, 16),
        (12, 14), (12, 15), (12, 16),
        (13, 14), (13, 15), (13, 16),]], run_time=0.275)


        # animate the edges of the 4th layer and remove animation of nodes in 4th layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [14, 15, 16]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(14, 17), (14, 18), 
        (15, 17), (15, 18),
        (16, 17), (16, 18),]], run_time=0.275)

        # animate the nodes of the 5th layer and remove animation of edges in the 4th layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [17, 18]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(14, 17), (14, 18), 
        (15, 17), (15, 18),
        (16, 17), (16, 18),]], run_time=0.275)


        # animate the edges of the 5th layer and remove animation of nodes in 5th layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [17, 18]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(17, 19), (17, 20), (17, 21),
        (18, 19), (18, 20), (18, 21),]], run_time=0.275)

        # animate the nodes of the 6th layer and remove animation of edges in the 5th layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [19, 20, 21]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(17, 19), (17, 20), (17, 21),
        (18, 19), (18, 20), (18, 21),]], run_time=0.275)


        # animate the edges of the 6th layer and remove animation of nodes in 6th layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [19, 20, 21]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(19, 22), (19, 23), (19, 24), (19, 25),
        (20, 22), (20, 23), (20, 24), (20, 25),
        (21, 22), (21, 23), (21, 24), (21, 25),]], run_time=0.275)

        # animate the nodes of the 7th layer and remove animation of edges in the 6th layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [22, 23, 24, 25]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(19, 22), (19, 23), (19, 24), (19, 25),
        (20, 22), (20, 23), (20, 24), (20, 25),
        (21, 22), (21, 23), (21, 24), (21, 25),]], run_time=0.275)


        # animate the edges of the 7th layer and remove animation of nodes in 7th layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [22, 23, 24, 25]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(22, 26), (22, 27), (22, 28), (22, 29),
        (23, 26), (23, 27), (23, 28), (23, 29),
        (24, 26), (24, 27), (24, 28), (24, 29),
        (25, 26), (25, 27), (25, 28), (25, 29),]], run_time=0.275)

        # animate the nodes of the 8th layer and remove animation of edges in the 7th layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [26, 27, 28, 29]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(22, 26), (22, 27), (22, 28), (22, 29),
        (23, 26), (23, 27), (23, 28), (23, 29),
        (24, 26), (24, 27), (24, 28), (24, 29),
        (25, 26), (25, 27), (25, 28), (25, 29),]], run_time=0.275)


        # animate the edges of the 8thth layer and remove animation of nodes in 8th layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [26, 27, 28, 29]] + 
        [line.animate.set_stroke('#ffffff', opacity=0.375, width=2.5) for edge, line in graph.edges.items()
        if edge in [(26, 30), (26, 31), (26, 32), (26, 33), (26, 34),
        (27, 30), (27, 31), (27, 32), (27, 33), (27, 34),
        (28, 30), (28, 31), (28, 32), (28, 33), (28, 34),
        (29, 30), (29, 31), (29, 32), (29, 33), (29, 34),]], run_time=0.275)

        # animate the nodes of the 9th layer and remove animation of edges in the 8th layer
        self.play(*[graph[node].animate.set_stroke('#ffffff', opacity=0.375, width=5) for node in graph.vertices if node in [30, 31, 32, 33, 34]] + 
        [line.animate.set_stroke(WHITE, opacity=1, width=2.5) for edge, line in graph.edges.items()
        if edge in [(26, 30), (26, 31), (26, 32), (26, 33), (26, 34),
        (27, 30), (27, 31), (27, 32), (27, 33), (27, 34),
        (28, 30), (28, 31), (28, 32), (28, 33), (28, 34),
        (29, 30), (29, 31), (29, 32), (29, 33), (29, 34),]], run_time=0.275)


        # remove animation of nodes in 9th layer
        self.play(*[graph[node].animate.set_stroke(WHITE, opacity=1, width=5) for node in graph.vertices if node in [30, 31, 32, 33, 34]], run_time=0.275)



        # Natural language processing
        self.play(*[ReplacementTransform(graph, quote), FadeTransformPieces(text_1, text_2), FadeOut(text_1)])

        # wait is like a delay you canset before another animation can play
        self.wait(0.5)
        self.play(ReplacementTransform(quote, vector))
        self.wait(1)



        # Data analysis & visualization
        # theta moves the camera along the x-axis or left and right
        # phi moves the camera along the y-axis or up and down
        # doing this moves camera leftward and upward simultaneously
        # simultaneously replace the previous mobject with axes, 
        # with the creation of points in the axes
        self.add_fixed_in_frame_mobjects(*[text_2, text_3])
        self.move_camera(theta=-45 * DEGREES, phi=60 * DEGREES, added_anims=[ReplacementTransform(vector, dot_3d), FadeOut(text_2)])

        # note that dividing by a whole number increases the 
        # rate our camera rotates
        # in between begin and end ambient camera rotation is where we 
        # place all our animations that will be simultaneous with the rotations
        self.begin_ambient_camera_rotation(rate=PI / 4.5, about="theta")
        # self.play(*[ReplacementTransform(dot_3d, axes), text_3.animate.set_opacity(1)])
        self.play(*[ReplacementTransform(dot_3d, axes), Create(line), text_3.animate.set_opacity(1)] + [Create(point) for point in points])

        # imperative that we wait and give a chance for camera to rotate
        # since as soon as wait stops then rotation stops
        self.wait(5)

        # reset camera angle to initial values
        self.move_camera(theta=init_theta, phi=init_phi, added_anims=[Uncreate(axes), Uncreate(line), FadeOut(text_3)] + [Uncreate(point) for point in points])
        self.stop_ambient_camera_rotation(about="theta")

        

        # client & server side web development
        self.play(FadeIn(text_5))
        self.play(Write(code), run_time=5)
        self.wait(0.5)
        self.play(*[FadeOut(code), FadeOut(text_4)])