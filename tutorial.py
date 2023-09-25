from manim import *
from custom import NeuralNetworkMobject

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

