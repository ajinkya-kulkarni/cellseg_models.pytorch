from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["post_proc_instanseg"]


def _generate_coordinate_map(
    spatial_dim: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if spatial_dim == 2:
        xx = torch.linspace(0, width * 64 / 256, width, device=device).view(1, 1, -1).expand(1, height, width)
        yy = torch.linspace(0, height * 64 / 256, height, device=device).view(1, -1, 1).expand(1, height, width)
        xxyy = torch.cat((xx, yy), 0)
    else:
        xxyy = torch.zeros((spatial_dim, height, width), device=device)
    return xxyy


def _torch_peak_local_max(
    image: torch.Tensor,
    neighbourhood_size: int = 4,
    minimum_value: float = 0.5,
    return_map: bool = False,
    dtype: torch.dtype = torch.int,
) -> torch.Tensor:
    h, w = image.shape
    image = image.view(1, 1, h, w)
    device = image.device

    kernel_size = 2 * neighbourhood_size + 1
    pooled, max_inds = F.max_pool2d(
        image,
        kernel_size=kernel_size,
        stride=1,
        padding=neighbourhood_size,
        return_indices=True,
    )

    inds = torch.arange(0, image.numel(), device=device, dtype=dtype).reshape(image.shape)
    peak_local_max = (max_inds == inds) * (pooled > minimum_value)

    if return_map:
        return peak_local_max

    return torch.nonzero(peak_local_max.squeeze()).to(dtype)


def _centre_crop(
    centroids: torch.Tensor,
    window_size: int,
    h: int,
    w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    C = centroids.shape[0]
    centroids = centroids.clone()
    centroids[:, 0] = centroids[:, 0].clamp(min=window_size // 2, max=h - window_size // 2)
    centroids[:, 1] = centroids[:, 1].clamp(min=window_size // 2, max=w - window_size // 2)
    window_slices = (
        centroids[:, None] + torch.tensor([[-1, -1], [1, 1]], device=centroids.device) * (window_size // 2)
    )

    grid_x, grid_y = torch.meshgrid(
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        torch.arange(window_size, device=centroids.device, dtype=centroids.dtype),
        indexing="ij",
    )

    mesh = torch.stack((grid_x, grid_y))
    mesh_grid = mesh.expand(C, 2, window_size, window_size)
    mesh_grid_flat = torch.flatten(mesh_grid, 2).permute(1, 0, 2)
    mesh_grid_flat = mesh_grid_flat + window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_grid_flat = torch.flatten(mesh_grid_flat, 1)

    return mesh_grid_flat, window_slices


def _feature_engineering(
    x: torch.Tensor,
    c: torch.Tensor,
    sigma: torch.Tensor,
    window_size: int,
    mesh_grid_flat: torch.Tensor,
) -> torch.Tensor:
    E = x.shape[0]
    C = c.shape[0]
    S = sigma.shape[0]

    x_sigma = torch.cat([x, sigma])
    x_sampled = x_sigma[:, mesh_grid_flat[0], mesh_grid_flat[1]]
    x_sampled = x_sampled.reshape(E + S, C, 2 * window_size, 2 * window_size).permute(1, 0, 2, 3)
    c_shaped = c.view(-1, E, 1, 1)
    x_sampled[:, :E] -= c_shaped
    x_sampled = x_sampled.permute(0, 2, 3, 1).reshape(C * 2 * window_size * 2 * window_size, E + S)
    return x_sampled


def _convert(
    prob_input: torch.Tensor,
    coords_input: torch.Tensor,
    size: Tuple[int, int],
    mask_threshold: float = 0.5,
) -> torch.Tensor:
    all_labels = torch.arange(1, 1 + prob_input.shape[0], dtype=torch.float32, device=prob_input.device)
    labels = torch.ones_like(prob_input) * torch.reshape(all_labels, (-1, 1, 1, 1))

    labels = labels.flatten()
    prob = prob_input.flatten()
    x = coords_input[0, ...].flatten()
    y = coords_input[1, ...].flatten()

    if size is None:
        size = (int(y.max() + 1), int(x.max() + 1))

    inds_prob = prob >= mask_threshold
    n_thresholded = torch.count_nonzero(inds_prob)
    if n_thresholded == 0:
        return torch.zeros(size, dtype=torch.float32, device=labels.device)

    arr = torch.zeros((int(n_thresholded), 5), dtype=coords_input.dtype, device=labels.device)
    arr[:, 1] = y[inds_prob]
    arr[:, 2] = x[inds_prob]
    arr[:, 0] = arr[:, 2] * size[1] + arr[:, 1]
    arr[:, 3] = labels[inds_prob]

    inds_sorted = prob[inds_prob].argsort(descending=True, stable=True)
    arr = arr[inds_sorted, :]

    inds_sorted = arr[:, 0].argsort(descending=False, stable=True)
    arr = arr[inds_sorted, :]

    inds_unique = torch.ones_like(arr[:, 0], dtype=torch.bool)
    inds_unique[1:] = arr[1:, 0] != arr[:-1, 0]

    output = torch.zeros(size, dtype=torch.float32, device=labels.device)
    output[arr[inds_unique, 2], arr[inds_unique, 1]] = arr[inds_unique, 3].float()

    return output


def _remap_values(remapping: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sorted_remapping = remapping[:, remapping[0].argsort()]
    index = torch.bucketize(x.ravel(), sorted_remapping[0])
    return sorted_remapping[1][index].reshape(x.shape)


def _fast_sparse_iou(sparse_onehot: torch.Tensor) -> torch.Tensor:
    intersection = torch.sparse.mm(sparse_onehot, sparse_onehot.T).to_dense()
    sparse_sum = torch.sparse.sum(sparse_onehot, dim=(1,))[None].to_dense()
    union = sparse_sum.T + sparse_sum - intersection
    return intersection / union


def _find_connected_components(
    adjacency_matrix: torch.Tensor,
    max_iterations: int = 100,
) -> torch.Tensor:
    n = adjacency_matrix.shape[0]
    if n == 0:
        return torch.zeros(2, 1, device=adjacency_matrix.device, dtype=torch.long)

    labels = torch.arange(1, n + 1, device=adjacency_matrix.device, dtype=torch.long)
    M = adjacency_matrix + torch.eye(n, device=adjacency_matrix.device)
    indices = torch.nonzero(M)
    if indices.size(0) == 0:
        raise ValueError("Graph has no edges or self-loops")
    row = indices[:, 0]
    col = indices[:, 1]

    for _ in range(max_iterations):
        prev_labels = labels.clone()
        min_labels = torch.full((n,), float("inf"), device=adjacency_matrix.device, dtype=torch.float)
        min_labels.scatter_reduce_(0, row, labels[col].float(), reduce="amin")
        labels = torch.minimum(labels, min_labels)
        if torch.equal(labels, prev_labels):
            break

    node_indices = torch.arange(1, n + 1, device=adjacency_matrix.device, dtype=torch.long)
    tentative_remapping = torch.stack((node_indices, labels))
    remapping = torch.cat(
        (torch.zeros(2, 1, device=adjacency_matrix.device, dtype=torch.long), tentative_remapping), dim=1
    )
    return remapping


def _merge_sparse_predictions(
    x: torch.Tensor,
    coords: torch.Tensor,
    mask_map: torch.Tensor,
    size: list,
    mask_threshold: float = 0.5,
    window_size: int = 128,
    min_size: int = 10,
    overlap_threshold: float = 0.5,
    mean_threshold: float = 0.5,
) -> torch.Tensor:
    labels = _convert(x, coords, size=(size[1], size[2]), mask_threshold=mask_threshold)[None]

    idx = torch.arange(1, size[0] + 1, device=x.device, dtype=coords.dtype)
    stack_ID = torch.ones((size[0], window_size, window_size), device=x.device, dtype=coords.dtype)
    stack_ID = stack_ID * (idx[:, None, None] - 1)

    coords_flat = torch.stack((stack_ID.flatten(), coords[0] * size[2] + coords[1])).to(coords.dtype)

    fg = x.flatten() > mask_threshold
    x_flat = x.flatten()[fg]
    coords_flat = coords_flat[:, fg]
    mask_map_flat = mask_map.flatten()

    using_mps = False
    device = x_flat.device
    if x_flat.is_mps:
        using_mps = True
        device = torch.device("cpu")
        x_flat = x_flat.to(device)
        mask_map_flat = mask_map_flat.to(device)
        coords_flat = coords_flat.to(device)

    sparse_onehot = torch.sparse_coo_tensor(
        coords_flat,
        (x_flat > mask_threshold).float(),
        size=(size[0], size[1] * size[2]),
        dtype=x_flat.dtype,
        device=device,
        requires_grad=False,
    )

    object_areas = torch.sparse.sum(sparse_onehot, dim=1).values()
    sum_mask_value = torch.sparse.sum((sparse_onehot * mask_map_flat[None]), dim=1).values()
    mean_mask_value = sum_mask_value / object_areas
    objects_to_remove = ~torch.logical_and(mean_mask_value > mean_threshold, object_areas > min_size)

    iou = _fast_sparse_iou(sparse_onehot)
    remapping = _find_connected_components((iou > overlap_threshold).float())

    if using_mps:
        remapping = remapping.to(x.device)
        labels = labels.to(x.device)

    labels = _remap_values(remapping, labels)

    labels_to_remove = (
        torch.arange(0, len(objects_to_remove), device=objects_to_remove.device, dtype=coords.dtype) + 1
    )[objects_to_remove]

    labels[torch.isin(labels, labels_to_remove)] = 0

    return labels


def post_proc_instanseg(
    inst_map: np.ndarray,
    aux_map: np.ndarray,
    pixel_classifier: torch.nn.Module = None,
    mask_threshold: float = 0.53,
    peak_distance: int = 4,
    seed_threshold: float = 0.5,
    overlap_threshold: float = 0.5,
    mean_threshold: float = -10000.0,
    window_size: int = 128,
    min_size: int = 10,
    max_seeds: int = 2000,
    **kwargs,
) -> np.ndarray:
    if aux_map.ndim == 2:
        aux_map = aux_map[None]
    aux_map_t = torch.from_numpy(aux_map.copy()).float()
    device = aux_map_t.device

    dim_coords = 2
    n_sigma = 1

    height, width = aux_map_t.shape[1:]

    coord_ch = aux_map_t[:dim_coords]
    sigma = aux_map_t[dim_coords : dim_coords + n_sigma]
    seed_map = aux_map_t[dim_coords + n_sigma :]

    if seed_map.ndim == 3:
        seed_map = seed_map[0]

    xxyy = _generate_coordinate_map(dim_coords, height, width, device)
    fields = (torch.sigmoid(coord_ch) - 0.5) * 8
    mask_map = seed_map

    if (mask_map > mask_threshold).max() == 0:
        return np.zeros((height, width), dtype=np.int32)

    local_centroids_idx = _torch_peak_local_max(
        mask_map, neighbourhood_size=int(peak_distance), minimum_value=seed_threshold
    )

    fields = fields + xxyy
    fields_at_centroids = fields[:, local_centroids_idx[:, 0], local_centroids_idx[:, 1]]

    if local_centroids_idx.shape[0] > max_seeds or local_centroids_idx.shape[0] == 0:
        return np.zeros((height, width), dtype=np.int32)

    C = fields_at_centroids.shape[1]
    window_size = min(window_size, height, width)
    window_size = window_size - window_size % 2

    h, w = height, width
    window_size = min(window_size, h, w)

    crops, coords = _compute_crops_wrapper(
        fields,
        fields_at_centroids.T,
        sigma,
        local_centroids_idx.int(),
        pixel_classifier,
        mask_threshold,
        window_size,
    )

    coords = coords[1:]

    C = crops.shape[0]
    if C == 0:
        return np.zeros((height, width), dtype=np.int32)

    crops = torch.sigmoid(crops)
    label = _merge_sparse_predictions(
        crops,
        coords,
        mask_map,
        size=(C, h, w),
        mask_threshold=mask_threshold,
        window_size=window_size,
        min_size=min_size,
        overlap_threshold=overlap_threshold,
        mean_threshold=mean_threshold,
    ).int()

    return label.squeeze().cpu().numpy().astype(np.int32)


def _compute_crops_wrapper(
    fields: torch.Tensor,
    centres: torch.Tensor,
    sigma: torch.Tensor,
    centroids_idx: torch.Tensor,
    pixel_classifier: torch.nn.Module,
    mask_threshold: float,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    h, w = fields.shape[-2:]
    C = centres.shape[0]

    mesh_grid_flat, window_slices = _centre_crop(centroids_idx, window_size, h, w)

    x = _feature_engineering(fields, centres, sigma, window_size // 2, mesh_grid_flat)

    if pixel_classifier is not None:
        x = pixel_classifier(x)
    else:
        x = x.mean(dim=-1, keepdim=True)

    x = x.view(C, 1, window_size, window_size)
    idx = torch.arange(1, C + 1, device=x.device, dtype=mesh_grid_flat.dtype)

    rep = torch.ones((C, window_size, window_size), device=x.device, dtype=mesh_grid_flat.dtype)
    rep = rep * (idx[:, None, None] - 1)

    iidd = torch.cat((rep.flatten()[None], mesh_grid_flat)).to(mesh_grid_flat.dtype)

    return x, iidd
