import numpy as np
from PIL import Image
from plat.interpolate import get_interpfn

def grid2img(arr, rows, cols, with_space):
    """Convert an image grid to a single image"""
    N = len(arr)
    channels, height, width = arr[0].shape

    total_height = rows * height
    total_width  = cols * width

    if with_space:
        total_height = total_height + (rows - 1)
        total_width  = total_width + (cols - 1)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(rows*cols):
        if i < N:
            r = i // cols
            c = i % cols

            cur_im = arr[i]

            if cur_im is not None:
                if with_space:
                    offset_y, offset_x = r*height+r, c*width+c
                else:
                    offset_y, offset_x = r*height, c*width
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = cur_im

    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    out = (255 * out).astype(np.uint8)
    return Image.fromarray(out)

def create_splash_grid(rows, cols, dim, space, anchors, spherical, gaussian):
    """Create a grid of latents with splash layout"""
    lerpv = get_interpfn(spherical, gaussian)

    u_list = np.zeros((rows, cols, dim))
    # compute anchors
    cur_anchor = 0
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space == 0:
                if anchors is not None and cur_anchor < len(anchors):
                    u_list[y,x,:] = anchors[cur_anchor]
                    cur_anchor = cur_anchor + 1
                else:
                    u_list[y,x,:] = np.random.normal(0,1, (1, dim))
    # interpolate horizontally
    for y in range(rows):
        for x in range(cols):
            if y%space == 0 and x%space != 0:
                lastX = space * (x // space)
                nextX = lastX + space
                fracX = (x - lastX) / float(space)
#                 print("{} - {} - {}".format(lastX, nextX, fracX))
                u_list[y,x,:] = lerpv(fracX, u_list[y, lastX, :], u_list[y, nextX, :])
    # interpolate vertically
    for y in range(rows):
        for x in range(cols):
            if y%space != 0:
                lastY = space * (y // space)
                nextY = lastY + space
                fracY = (y - lastY) / float(space)
                u_list[y,x,:] = lerpv(fracY, u_list[lastY, x, :], u_list[nextY, x, :])

    u_grid = u_list.reshape(rows * cols, dim)

    return u_grid

def create_gradient_grid(rows, cols, dim, analogy, anchors, spherical, gaussian):
    """Create a grid of latents with gradient layout (includes analogy)"""
    lerpv = get_interpfn(spherical, gaussian)
    hyper = False

    numsamples = rows * cols
    u_list = np.zeros((numsamples, dim))
    if anchors is not None:
        # xmin_ymin, xmax_ymin, xmin_ymax = anchors[0:3]
        xmin_ymin, xmin_ymax, xmax_ymin = anchors[0:3]
    else:
        xmin_ymin = np.random.normal(0, 1, dim)
        xmax_ymin = np.random.normal(0, 1, dim)
        xmin_ymax = np.random.normal(0, 1, dim)
    if(analogy):
        xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
        if hyper:
            tl = xmin_ymin
            tr = xmax_ymin
            bl = xmin_ymax
            xmax_ymax = bl + (tr - tl)
            xmin_ymax = bl - (tr - tl)
            xmax_ymin = tr + (tl - bl)
            xmin_ymin = xmin_ymax + (xmax_ymin - xmax_ymax)
    elif anchors is not None:
        xmax_ymax = anchors[3]
    else:
        xmax_ymax = np.random.normal(0, 1, dim)

    for y in range(rows):
        if  y == 0:
            # allows rows == 0
            y_frac = 0
        else:
            y_frac = y / (rows - 1)
        xmin_ycur = lerpv(y_frac, xmin_ymin, xmin_ymax)
        xmax_ycur = lerpv(y_frac, xmax_ymin, xmax_ymax)
        for x in range(cols):
            if x == 0:
                # allows cols == 0
                x_frac = 0
            else:
                x_frac = x / (cols - 1)
            xcur_ycur = lerpv(x_frac, xmin_ycur, xmax_ycur)
            n = y * cols + x
            u_list[n:n+1,:] = xcur_ycur

    return u_list
