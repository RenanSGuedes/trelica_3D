import matplotlib.pyplot as plt
import numpy as np

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

# Fixing random state for reproducibility
np.random.seed(19680801)

grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, .5, 0, .5, .4, 1, 0, 0, 0, 0],
    [.3, .4, .5, 1, 1, 1, 1, .6, 0, 0],
    [.4, .4, .5, .7, 1, 1, 1, 1, 1, .8],
    [0, .2, .3, .4, .7, 1, 1, 1, 1, .9],
    [.1, .2, .4, .5, .6, 1, 1, 1, .8, .7],
    [.2, .2, .2, .4, .5, 1, 1, 1, .7, .6],
    [.3, .4, .5, 1, 1, 1, 1, 1, 1, .4],
    [1, 1, 1, 1, 1, .7, .7, .6, .5, .4],
    [1, 1, 1, .5, .4, .3, .6, .9, 1, 1]
]

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(grid, interpolation=interp_method, cmap='viridis')
    ax.set_title(str(interp_method))

plt.tight_layout()
plt.show()