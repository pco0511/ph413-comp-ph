import tqdm

import numpy as np
import matplotlib.pyplot as plt


def create_random_mask(
    dim0: int,
    dim1: int,
    min_dist_from_edge: int,
    rand_circle_rad_min: int,
    rand_circle_rad_max: int,
    min_area: int,
    max_area: int,
    continue_loop_prob: float,
) -> np.ndarray:
    '''
    Generate a random 2‑D binary mask composed of overlapping circles.

    Returns
    -------
    mask : np.ndarray
        Binary image of shape (dim0, dim1) containing the generated mask.
    '''
    mask = np.zeros((dim0, dim1), dtype=np.uint8)

    grid0, grid1 = np.meshgrid(np.arange(dim0), np.arange(dim1), indexing='ij')

    # The object should be within this big circle to avoid edge artifacts
    big_circle = (
        (grid0 - dim0 / 2) ** 2 + (grid1 - dim1 / 2) ** 2
    ) < (dim0 / 2 - min_dist_from_edge) ** 2

    continue_flag = True
    while continue_flag:
        # Random center coordinates and radius for the new circle
        rand_ind0 = np.random.randint(0, dim0)
        rand_ind1 = np.random.randint(0, dim1)
        rand_rad = np.random.randint(rand_circle_rad_min, rand_circle_rad_max)

        # Create candidate circle
        curr_circle = (
            (grid0 - rand_ind0) ** 2 + (grid1 - rand_ind1) ** 2
        ) < rand_rad ** 2

        # Reject if any part of the candidate circle falls outside the big circle
        if np.any(curr_circle & (~big_circle)):
            continue

        # Ensure connectivity: if the mask already has content, the new circle
        # must overlap at least one existing pixel.
        if mask.sum() > 0 and mask[rand_ind0, rand_ind1] == 0:
            continue

        # Add the circle to the mask
        mask[curr_circle] = 1

        # Decide whether to continue adding circles
        current_area = mask.sum()
        if current_area > min_area:
            if np.random.uniform() > continue_loop_prob or current_area > max_area:
                continue_flag = False

    return mask


def create_random_mask_optimized(
    dim0: int,
    dim1: int,
    min_dist_from_edge: int,
    rand_circle_rad_min: int,
    rand_circle_rad_max: int,
    min_area: int,
    max_area: int,
    continue_loop_prob: float,
) -> np.ndarray:
    """
    Generate a random 2-D binary mask composed of overlapping circles,
    optimized to avoid full-array operations on each iteration.
    """
    mask = np.zeros((dim0, dim1), dtype=np.uint8)

    cy, cx = dim0 / 2, dim1 / 2
    big_R = min(cy, cx) - min_dist_from_edge

    current_area = 0
    continue_flag = True

    first = True

    while continue_flag:
        r = np.random.randint(rand_circle_rad_min, rand_circle_rad_max)

        if first:
            valid = False
            while not valid:
                y = np.random.randint(int(cy - big_R + r), int(cy + big_R - r))
                x = np.random.randint(int(cx - big_R + r), int(cx + big_R - r))
                if (y - cy)**2 + (x - cx)**2 <= (big_R - r)**2:
                    valid = True
            first = False
        else:
            ys, xs = np.where(mask)
            idx = np.random.randint(len(ys))
            y0, x0 = ys[idx], xs[idx]
            dy = np.random.randint(-r//2, r//2 + 1)
            dx = np.random.randint(-r//2, r//2 + 1)
            y = np.clip(y0 + dy, 0, dim0-1)
            x = np.clip(x0 + dx, 0, dim1-1)
            if (y - cy)**2 + (x - cx)**2 > (big_R - r)**2:
                continue

        y1, y2 = max(0, y-r), min(dim0, y+r+1)
        x1, x2 = max(0, x-r), min(dim1, x+r+1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (yy - y)**2 + (xx - x)**2 <= r**2

        mask[y1:y2, x1:x2][circle] = 1

        current_area = mask.sum()
        
        if current_area > min_area:
            if np.random.rand() > continue_loop_prob or current_area > max_area:
                continue_flag = False

    return mask