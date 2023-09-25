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

