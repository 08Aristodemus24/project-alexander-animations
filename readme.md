# A repository that will store all the animations I will test and use in my portfolio website `project-alexander`

1. `manim -pql <scenes.py> <class name>` will create the animation given a "template" defined as our class
2. `manim -pqh <scenes.py> <class name>` will create the animation in high quality given the `qh` part in the -pqh flag which means quality will be high of the outputted video
3. `square.next_to(circle, RIGHT, buff=0.5)  # set the position` is like placing an element beside another element in html and then setting the squares margin to 0.5
4. When you prepend `.animate` to any method call that modifies a Mobject, the method becomes an animation which can be played using `self.play`. Akin to how we used a class `Create()`, `FadeOut()`, or `Transform()` to define animations for our manim objects 
```
self.play(ReplacementTransform(square, circle))  # transform the square into a circle
self.play(circle.animate.set_fill(PINK, opacity=0.5))  # color the circle on screen
```
5. Available code styles should you use code manim object are: 'abap', 'algol_nu', 'algol', 'arduino', 'autumn', 'borland', 'bw', 'colorful', 'default', 'dracula', 'emacs', 'friendly_grayscale', 'friendly', 'fruity', 'github-dark', 'gruvbox-dark', 'gruvbox-light', 'igor', 'inkpot', 'lightbulb', 'lilypond', 'lovelace', 'manni', 'material', 'monokai', 'murphy', 'native', 'nord-darker', 'nord', 'one-dark', 'paraiso-dark', 'paraiso-light', 'pastie', 'perldoc', 'rainbow_dash', 'rrt', 'sas', 'solarized-dark', 'solarized-light', 'staroffice', 'stata-dark', 'stata-light', 'stata', 'tango', 'trac', 'vim', 'vs', 'xcode', and 'zenburn'

6. to export manim animation to a 4k resolution video instead of just `qh` or `ql` we use now `qk` as the part of the flag `-pqk` in the command `manim -pqk tutorial.py MorphingHeaders`

7. to export manim animation to a 4k resolution gif with transparent background instead of a video and black background respectively use `manim -pqk tutorial.py MorphingHeaders -t --format=gif`. Which has flags `-t` and `--format=gif` representing that we want only the manim objects to be visible with color and not with a background, and that we want the output file to in a `.gif` format respectively

8. setting frame wid th and height without sacrificing quality canbe found here https://flyingframes.readthedocs.io/en/latest/ch5.html

9. creating graphs can be found here: https://docs.manim.community/en/stable/reference/manim.mobject.graph.Graph.html

10. `manim -pqk tutorial.py MorphingHeaders -c GRAY` instead of specifying a flag of -t for a transparent gif or video we can use instead the -c flag which indicates the color we want for our background

11. LEFT, RIGHT, DOWN, UP are macros consisting of coordinates for a 3 dimeensional cartesian plane, the x, y, and z axis, this is why when we use move_to we actually supply a list consiting of the x, y, and z coordinates, so that our mobject moves along the x-axis, along the y-axis, and the z-axis which may look left right, down up, and front and back.

```
self.play(g[1].animate.move_to([1, 1, 0]),
    g[2].animate.move_to([-1, 1, 0]),
    g[3].animate.move_to([1, -1, 0]),
    g[4].animate.move_to([-1, -1, 0]))
```

```
>>> from manim import *
>>>
>>> RIGHT
array([1., 0., 0.])
>>> LEFT
array([-1.,  0.,  0.])
>>>
>>> # shifts one value to the right and one value to the left in the x-axis
>>>
>>> UP
array([0., 1., 0.])
>>> DOWN
array([ 0., -1.,  0.])
>>>
>>> # shifts one value up and one value down in the y-axis
```

But note that these right, left, up, down values are still

12. availabel layouts for graph mobject are: "circular": nx.layout.circular_layout, "kamada_kawai": nx.layout.kamada_kawai_layout,
        "planar": nx.layout.planar_layout,
        "random": nx.layout.random_layout,
        "shell": nx.layout.shell_layout,
        "spectral": nx.layout.spectral_layout,
        "partite": nx.layout.multipartite_layout,
        "tree": _tree_layout,
        "spiral": nx.layout.spiral_layout,
        "spring": nx.layout.spring_layout,

13. any property for intance .set_color() of a mobject like Graph() or its node g[0] can be preprended by the .animate property e.g. g.animate.set_color() or g[0].animate.set_color()

14. the args x_range, y_range, and z_range of ThreeDAxes object or plane indicates the ff. [x_min, x_max, x_step] values of the x-axis, [y_min, y_max, y_step] values of the y-axis, and [z_min, z_max, z_step] values of the z-axis respectively

15. for using the degrees macro, since we cannot user degree values recall that in math we have to convert it first to an integer in order to make respective calculations 
```
>>> from manim import *
>>> DEGREES
0.017453292519943295
>>> DEGREES
0.017453292519943295
>>> 90 * DEGREES
1.5707963267948966
>>> 180 * DEGREES
3.141592653589793
>>> 270 * DEGREES
4.71238898038469
>>> 360 * DEGREES
6.283185307179586
>>>
```

16. to plot lines in 3d space we use a parametric function and to plot surfaces or 2d shapes in 3d spaces we use a parametric surface function

17. for 3d objects look here:
* https://docs.manim.community/en/stable/reference/manim.mobject.three_d.three_dimensions.html
* https://docs.manim.community/en/stable/reference/manim.mobject.three_d.polyhedra.html
* https://docs.manim.community/en/stable/reference/manim.mobject.three_d.three_dimensions.Surface.html

18. so there is a difference in playing animations simultaneously using Animation objects and property animations of a Mobject e.g. `self.play(*[ReplacementTransform(square, axes), Create(point_a), Create(point_b), Create(point_c)])` is very different from `self.play(*[square.animate.set_fill(PINK, opacity=0.5), square.animate.set_stroke(PINK, opacity=1)])`, doing the latter will only just do the second animation, however we can chain animating mobject property animations since .`<mobject>.animate.<property>()` returns the same mobject animation builder object that we can use to chain another property we want to animate using dot notation.

19. `TAU` is a constant like `DEGREES`, etc.
```
>>> TAU
6.283185307179586
```

20. increasing the layout_scale argument in a Graph mobject will increase the spacing between nodes and not make the graph too compressed and dense

21. when using sikmultaenous animations you can do this 
```
>>> y = lambda *x: print(x)
>>> y(*([1, 2] + [2]))
(1, 2, 2)
```

which translates in manim to...

```
self.play(*[ReplacementTransform(dot, axes)] + [Create(point) for point in points])
```

22. initially `phi` and `theta` or left & right and up and down respectively had `0.0` and `-1.5707963267948966` but after setting to `60 * degrees` and `-45 * degrees` respectively the values are now `1.0471975511965976` and `-0.7853981633974483`

and because camera rotations begins then waits for 5 seconds theta excluding phi since rotation is only set to rotate on the theta "axis" so to speak theta after 5 second camera rotation is now `3.310307814615915`. So we need to reset theta angle such that it is `-1.5707963267948966` again and phi is `0.0`, to do this we just have to pass the initial `self.camera.get_theta()` and the initial `self.camera.get_phi()` we had before we even called `self.move_camera()` to `self.move_camera()` again to reverse all rotations

23. note it is imperative that latex symbols have spaces in between in each caharacter unless a symbol requires other parts like subscripts and superscripts in that case the characters preceding or succeeding these superscripts or subscripts must not have spaces between other parts of the whole latex symbol e.g. below is the correct way
```
linear_func = MathTex(r"\Theta X + B")
        self.add(linear_func)
```
or
```
t = MathTex(r"\int_a^b f'(x) dx = f(b)- f(a)")
        self.add(t)
```
and the wrong way is 
```
linear_func = MathTex(r"\ThetaX+B")
        self.add(linear_func)
```

24. Matrices
```
>>> MathTex(r"\begin{bmatrix} 0 & \tau_{1, 2} \\ \end{bmatrix}")
MathTex('\\begin{bmatrix} 0 & \\tau_{1, 2} \\\\ \\end{bmatrix}')

>>> MathTex(r"\begin{bmatrix} \beta^{(0)}_{1, 0} \\ \end{bmatrix}")
MathTex('\\begin{bmatrix} \\beta^{(0)}_{1, 0} \\\\ \\end{bmatrix}')
>>> exit()
```

25. Note: some greek letters may not be available. For instance the command \Tau is defined when unicode-math is used, because it must refer to the Greek letter.

In legacy TeX there is no such command, because a Tau has the same shape as a T. The same for

\Alpha \Beta \Epsilon \Zeta \Eta \Iota \Kappa \Mu \Nu \Omicron \Rho \Chi
because the corresponding glyphs have the same shape as a Latin Letter.

26. unlike using transformation animations like ReplacementTransform(), FadeTransform(), and Transform() which allows us to convert a mobject A into virutally mobject B that we both declared where we can now use mobject B to continue chaining the animations fi we want. However should we use a mobjects animate property and subsequently its animation methods like set_fill(), set_stroke(). the method .become() of a mobjects animation property does not turn the mobject a into mobject b, but rather mobject a is still mobject a but now with the characteristics or the "shape" so to speak of mobject b so if we wanted to perform animations on this newly "transformed" object we want to call animation liek FadeTransform(), Transform(), ReplacementTransform() etc, and pass in still mobject a, since it is still indeed mobject a just in the form of mobject b.
```
self.play(Write(linear_func))
self.wait(1)
self.play(linear_func.animate.become(whole_op))
self.wait(1)
self.play(FadeOut(linear_func))
```
Will work but...
```
self.play(Write(linear_func))
self.wait(1)
self.play(linear_func.animate.become(whole_op))
self.wait(1)
self.play(FadeOut(whole_op))
```
Will not

27. video about making neural networks may be helpful: https://www.reddit.com/r/programming/comments/l8jwl5/i_created_a_video_about_neural_networks_that_is/

# Usage:
**Prerequesities to do:**
1. make sure you have `ffmpeg` and `python` installed, and optionally `miketex`. ManimCE details the installation in this link: https://docs.manim.community/en/stable/installation.html

**To do:**
1. clone repository with `git clone https://github.com/08Aristodemus24/project-alexander-animations.git`
2. navigate to directory with manage.py file and requirements.txt file
3. run command; `conda create -n <name of env e.g. project-alexander-animations> python=3.11.5`. Note that 3.11.5 must be the python version otherwise packages to be installed might not be compatible with a different python version e.g. manim, numpy, etc. 
4. once environment is created activate it by running command `conda activate`
5. then run `conda activate project-alexander-animations`
6. check if pip is installed by running `conda list -e`
7. if it is there then move to step 8, if not then install `pip` by typing `conda install pip`
8. if `pip` exists or install is done run `pip install -r requirements.txt` in the directory you are currently in
9. once done installing you can view animations by `manim -pql scenes.py <class name to see video output of>`. Note you can replace `<class name to see video output of>` to the any of the classes defined in scenes.py
10. note that in installing miketex latex commands will not be available yet so add the latex executable files path to our path environment variable. This may be `C:\Program Files\MiKTeX\miktex\bin\x64` but it will vary across systems

# Model Building for animation
**To do:**
1. on a side note the reason why your liner model gives an R2 score of near 0 is because of this https://stackoverflow.com/questions/44218972/very-low-score-with-scikit-learn-linear-regression-for-obvious-pattern