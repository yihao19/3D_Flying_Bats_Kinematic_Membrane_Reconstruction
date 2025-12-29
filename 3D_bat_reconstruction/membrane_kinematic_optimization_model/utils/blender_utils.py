# -*- coding: utf-8 -*-
"""
Created on Sat May 10 13:26:04 2025

@author: yihao
"""
import soft_renderer as sr
import torch
import numpy as np
def update_simulations(remote_sims, params):
    """Updates all remote simulations with new supershape samples.

    We split N parameter samples into N/R chunks where R is the number of
    simulation instances. Besides the parameters, we send subset indices
    to the simulation instances which will be returned to us alongside
    with the rendered images. The subset indices allow us to associate
    parameters with images in the optimization.
    
    ids = torch.arange(params.shape[0]).long()
    R = len(remote_sims)
    for remote, subset, subset_ids in zip(
        remote_sims, torch.chunk(params, R), torch.chunk(ids, R)
    ):
        print(subset)
        print(f"subset_shape: {subset.shape}")
        print(f"subset_id shape: {subset_ids.shape}")
        remote.send(shape_params=subset.cpu().numpy(), shape_ids=subset_ids.numpy())
    """
    
    for remote in remote_sims: 
        remote.send(shape_params = params, shape_ids = 0)
        
def get_target_objs(dl, remotes, n):
    """Returns a set of images from the target distribution."""
    """"""
    target_objs = []
    gen = iter(dl)
    for _ in range(n):
        item = next(gen)
        mesh = item['obj']
        '''
        for vertex in mesh['vertices'][0]: 
            vertex[1] = -vertex[1]
            vertex[2] = -vertex[2]
            vertex *= 0.0035
        '''
        output = sr.Mesh(mesh['vertices'][0].repeat(9, 1, 1).float().cuda(),mesh['faces'][0].repeat(9, 1, 1).float().cuda())
        target_objs.append(output)
    return target_objs


def y_forward_z_up(vertices, scale_factor: float=1.0): 
    x = vertices[:, :,  0] * scale_factor
    y = -vertices[:, :, 2] * scale_factor
    z = vertices[:, :,  1] * scale_factor
    vertices = torch.stack([x, y, z], axis=2)
    
    return vertices


def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """
    Save a mesh to OBJ format.
    
    :param vertices: (N,3) float array of vertex coordinates.
    :param faces:    (M,K) int array of face indices (0-based).
    :param filepath: Path to output .obj file.
    :param comment:  Optional string to insert as a comment at top.
    """
    # Validate shapes
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices must be shape (N,3)"
    assert faces.ndim == 2, "faces must be shape (M,K)"
    
    with open(filepath, 'w') as f:
        # optional comment
        # write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # write faces (convert to 1-based)
        for face in faces:
            # Convert each index to 1-based
            face_indices = face + 1
            # Join them into space-separated
            idx_str = " ".join(str(idx) for idx in face_indices)
            f.write(f"f {idx_str}\n")

    return