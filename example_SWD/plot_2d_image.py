import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_2d_image(x, z, data, cmap="viridis", zero_midpoint=False, resX=1000, resZ=1000):
    """
    Plots values of a SPECEFM2D model/gradient by interpolating onto a regular grid

    :type x: np.array
    :param x: x values of GLL mesh
    :type z: np.array
    :param z: z values of GLL mesh
    :type data: np.array
    :param data: D
    :type cmap: str
    :param cmap: matplotlib colormap to be applied to the contour plot. Defaults
        to 'viridis'
    :type zero_midpoint: bool
    :param zero_midpoint: set 0 as the midpoint for the colorbar. Useful for
        diverging colorscales (e.g., for gradients), where the neutral color
        (e.g., white) is set at value=0
    :type resX: int
    :param resX: number of points for the interpolation in x- direction (default=1000)
    :type resZ: int
    :param resZ: number of points for the interpolation in z- direction (default=1000)
    """
    # Figure out aspect ratio of the figure
    r = (np.max(x) - np.min(x))/(np.max(z) - np.min(z))
    rx = r/np.sqrt(1 + r**2)
    ry = 1/np.sqrt(1 + r**2)

    # Assign zero as the midpoint for things like gradients
    if zero_midpoint:
        abs_max_val = np.max(np.abs(data))
        vmin = -1 * abs_max_val
        vmax = abs_max_val
    else:
        vmin, vmax = None, None

    f = plt.figure(figsize=(10 * rx, 10 * ry))
    

    # trick interpolation using the maximum values of z in case of concave topography.
    # nan values helps interpolation act expectedly.
    # Can be tested using the default specfem2D model: simple_topography_and_also_a_simple_fluid_layer
    x = np.append(x, [np.min(x), np.max(x)])
    z = np.append(z, [np.max(z), np.max(z)])
    data = np.append(data, [np.nan, np.nan])

    xi = np.linspace(np.min(x), np.max(x), resX)
    zi = np.linspace(np.min(z), np.max(z), resZ)
    X, Z = np.meshgrid(xi, zi)
    V = griddata((x, z), data, (X, Z), method='linear')
    im = plt.imshow(V, vmax=vmax, vmin=vmin,
                    extent=[x.min(), x.max(), z.min(), z.max()],
                    cmap=cmap,
                    origin='lower')

    cbar = plt.colorbar(im, shrink=0.8, pad=0.025)
    plt.axis("image")

    return f, im, cbar