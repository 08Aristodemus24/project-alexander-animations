from manim import *

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
        quote = Text("“If more of us valued food and cheer and song above\nhoarded gold, it would be a merrier world.”\n― J.R.R. Tolkien", gradient=('#7734eb', '#eba234'))
        quote.scale(0.5)
        
        vector = DecimalMatrix([[0.93, -0.1, 0.65, 0.21, 1.23, 1.04]])

        self.play(Create(quote))

        # wait is like a delay you canset before another animation can play
        self.wait(1)
        self.play(Transform(quote, vector))
        self.play(FadeOut(vector))
        

