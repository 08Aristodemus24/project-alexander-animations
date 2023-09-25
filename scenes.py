from manim import *

class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        circle.set_stroke(PINK, opacity=1)
        self.play(Create(circle))  # show the circle on screen


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