from dataclasses import dataclass

import tqdm

import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

from . import radon_transformation

@dataclass
class ShapeConfig:
    dim0: int
    dim1: int
    margin: int
    min_rad: int
    max_rad: int
    min_area: int
    max_area: int
    continue_prob: float

@dataclass
class AtomConfig:
    pixel_size: float
    target_density_dist: float
    target_fill_ratio: float
    patch_radius: int
    atom_min_dist: float
    atom_size: float

@dataclass
class TomographyConfig:
    angle_arr: np.ndarray
    interpolation: str = "cubic"

def create_random_mask(shape_cfg: ShapeConfig) -> np.ndarray:
    """
    Generate a random 2-D binary mask composed of overlapping circles.
    """
    dim0, dim1 = shape_cfg.dim0, shape_cfg.dim1
    mask = np.zeros((dim0, dim1), dtype=np.uint8)

    cy, cx = dim0 / 2, dim1 / 2
    big_R = min(cy, cx) - shape_cfg.margin

    current_area = 0
    continue_flag = True
    first = True

    while continue_flag:
        r = np.random.randint(shape_cfg.min_rad, shape_cfg.max_rad)

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
        if current_area > shape_cfg.min_area:
            if np.random.rand() > shape_cfg.continue_prob or current_area > shape_cfg.max_area:
                continue_flag = False

    return mask


def create_image_from_atomic_structure(
    shape_cfg: ShapeConfig,
    atom_cfg: AtomConfig,
    atom_pos_arr: np.ndarray,
) -> np.ndarray:
    """
    Render a grayscale image by placing 2-D Gaussian blobs at atomic positions.
    """
    dim0 = shape_cfg.dim0
    dim1 = shape_cfg.dim1
    
    # here we compute against given positions

    image = np.zeros((dim0, dim1), dtype=np.float32)

    # Pre-compute Gaussian kernel
    offset = np.arange(-atom_cfg.patch_radius, atom_cfg.patch_radius + 1)
    g0, g1 = np.meshgrid(offset, offset, indexing="ij")
    sigma_pix = atom_cfg.atom_size / atom_cfg.pixel_size
    gaussian_kernel = np.exp(-(g0**2 + g1**2) / (2 * sigma_pix**2))

    for r, c in np.round(atom_pos_arr).astype(int):
        rows = np.arange(r - atom_cfg.patch_radius, r + atom_cfg.patch_radius + 1)
        cols = np.arange(c - atom_cfg.patch_radius, c + atom_cfg.patch_radius + 1)

        valid_r = (rows >= 0) & (rows < image.shape[0])
        valid_c = (cols >= 0) & (cols < image.shape[1])
        if not (valid_r.any() and valid_c.any()):
            continue

        ir, ic = rows[valid_r], cols[valid_c]
        kr = np.where(valid_r)[0]
        kc = np.where(valid_c)[0]
        image[np.ix_(ir, ic)] += gaussian_kernel[np.ix_(kr, kc)]

    return image


def create_data_from_gt(
    gt_image_array: np.ndarray,
    tomo_cfg: TomographyConfig,
    display_progress: bool=True,
    pbar_kwargs = {}
) -> np.ndarray:
    """
    Perform forward Radon transform + filtered back-projection.
    """
    data = np.zeros_like(gt_image_array, dtype=np.float32)
    n_slices = gt_image_array.shape[0]

    if not display_progress and pbar_kwargs:
        raise ValueError("pbar_kwargs is not none but display_progress is false")

    stream = tqdm.trange(n_slices, **pbar_kwargs) if display_progress else range(n_slices)
    
    for i in stream:
        sino = radon_transformation.radon_transform(
            gt_image_array[i, :, :],
            tomo_cfg.angle_arr,
        )
        data[i, :, :] = radon_transformation.iradon_transform(
            sino,
            tomo_cfg.angle_arr,
            interpolation=tomo_cfg.interpolation,
        )
    return data


def create_data(
    shape_cfg: ShapeConfig,
    atom_cfg: AtomConfig,
    num_data: int,
    display_progress: bool=True,
    pbar_kwargs = {}
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Generate ground-truth masks, images, and atom position lists.
    """
    dim0, dim1 = shape_cfg.dim0, shape_cfg.dim1
    masks = np.zeros((num_data, dim0, dim1), dtype=np.uint8)
    images = np.zeros((num_data, dim0, dim1), dtype=np.float32)
    positions: list[np.ndarray] = []

    grid0, grid1 = np.meshgrid(np.arange(dim0), np.arange(dim1), indexing="ij")
    min_dist_pix = atom_cfg.atom_min_dist / atom_cfg.pixel_size
    
    if not display_progress and pbar_kwargs:
        raise ValueError("pbar_kwargs is not none but display_progress is false")
    
    stream = tqdm.trange(num_data, **pbar_kwargs) if display_progress else range(num_data)
    
    for idx in stream:
        mask = create_random_mask(shape_cfg)

        phys_area = mask.sum() * atom_cfg.pixel_size**2
        target_n = int(
            np.round(phys_area / atom_cfg.target_density_dist**2
                     * atom_cfg.target_fill_ratio)
        )

        pos_arr = np.empty((0, 2), dtype=np.float32)
        while pos_arr.shape[0] < target_n:
            r0 = np.random.uniform() * dim0
            r1 = np.random.uniform() * dim1
            circle = ((grid0 - r0)**2 + (grid1 - r1)**2) <= atom_cfg.patch_radius**2
            if (circle & (mask == 0)).any():
                continue
            if pos_arr.shape[0] > 0:
                dist2 = np.sum((pos_arr - np.array([r0, r1]))**2, axis=1)
                if (dist2 < min_dist_pix**2).any():
                    continue
            pos_arr = np.vstack([pos_arr, [r0, r1]])

        masks[idx, :, :] = mask
        images[idx, :, :] = create_image_from_atomic_structure(shape_cfg, atom_cfg, pos_arr)
        positions.append(pos_arr)

    return images, masks, positions


# def create_data_optimized(
#     shape_cfg: ShapeConfig,
#     atom_cfg: AtomConfig,
#     num_data: int,
#     display_progress: bool = True,
# ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
#     """
#     Generate ground-truth masks, images, and atom position lists (Optimized).
#     """
#     dim0, dim1 = shape_cfg.dim0, shape_cfg.dim1
#     masks_out = np.zeros((dim0, dim1, num_data), dtype=np.uint8)
#     images_out = np.zeros((dim0, dim1, num_data), dtype=np.float32)
#     positions_out: list[np.ndarray] = []

#     grid0, grid1 = np.meshgrid(np.arange(dim0), np.arange(dim1), indexing="ij")
#     min_dist_pix_sq = (atom_cfg.atom_min_dist / atom_cfg.pixel_size)**2
    
#     pr = atom_cfg.patch_radius
#     ceil_pr = int(np.ceil(pr))
#     y_s, x_s = np.ogrid[-ceil_pr : ceil_pr + 1, -ceil_pr : ceil_pr + 1]
#     struct_el = x_s**2 + y_s**2 <= pr**2

#     iterator = tqdm.trange(num_data) if display_progress else range(num_data)
#     for idx in iterator:
#         mask = create_random_mask(shape_cfg) 

#         phys_area = mask.sum() * atom_cfg.pixel_size**2
#         target_n = int(
#             np.round(phys_area / atom_cfg.target_density_dist**2
#                      * atom_cfg.target_fill_ratio)
#         )

#         if target_n == 0:
#             masks_out[:, :, idx] = mask
#             positions_out.append(np.empty((0, 2), dtype=np.float32))
#             images_out[:, :, idx] = create_image_from_atomic_structure(
#                 shape_cfg, atom_cfg, np.empty((0,2),dtype=np.float32)
#             )
#             continue
            
#         safe_center_mask = ndi.binary_erosion(mask.astype(bool), structure=struct_el, border_value=0)
#         valid_center_coords = np.argwhere(safe_center_mask)

#         pos_list: list[list[float]] = [] 
        
#         if valid_center_coords.shape[0] == 0:
#             masks_out[:, :, idx] = mask
#             positions_out.append(np.empty((0, 2), dtype=np.float32))
#             images_out[:, :, idx] = create_image_from_atomic_structure(
#                 shape_cfg, atom_cfg, np.empty((0,2),dtype=np.float32)
#             )
#             continue
        
#         max_attempts = target_n * 100 
#         num_attempts = 0

#         while len(pos_list) < target_n and num_attempts < max_attempts:
#             num_attempts += 1
            
#             center_pixel_coords = valid_center_coords[np.random.randint(valid_center_coords.shape[0])]
            
#             r0_candidate = center_pixel_coords[0] + np.random.uniform() - 0.5
#             r1_candidate = center_pixel_coords[1] + np.random.uniform() - 0.5
            
#             circle = ((grid0 - r0_candidate)**2 + (grid1 - r1_candidate)**2) <= atom_cfg.patch_radius**2
#             if np.any(circle & (mask == 0)):
#                 continue
            
#             # 4. Check minimum distance to other already placed atoms
#             current_candidate_pos = np.array([r0_candidate, r1_candidate])
#             is_too_close = False
#             if len(pos_list) > 0:
#                 existing_pos_arr = np.array(pos_list, dtype=np.float32)
#                 dist_sq_to_existing = np.sum((existing_pos_arr - current_candidate_pos)**2, axis=1)
#                 if np.any(dist_sq_to_existing < min_dist_pix_sq):
#                     is_too_close = True
            
#             if is_too_close:
#                 continue
            
#             pos_list.append([r0_candidate, r1_candidate])

#         final_pos_arr = np.array(pos_list, dtype=np.float32) if pos_list else np.empty((0,2), dtype=np.float32)
        
#         masks_out[:, :, idx] = mask
#         images_out[:, :, idx] = create_image_from_atomic_structure(shape_cfg, atom_cfg, final_pos_arr)
#         positions_out.append(final_pos_arr)

#     return images_out, masks_out, positions_out