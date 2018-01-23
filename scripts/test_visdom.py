import visdom
import numpy as np

vis = visdom.Visdom()
vis.text('Hello, world!')
vis.image(np.random.rand(3,10,10))
