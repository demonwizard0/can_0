from typing import Tuple
from modules.converters.schemas import MeshData

import torch
import torch.nn.functional as F


def subdivide(mesh_data: MeshData, iterations: int = 1) -> MeshData:
    """Subdivides edges by adding a vertex at the midpoint of each edge creating 4 new faces in space of one."""
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    uvs = mesh_data.uvs.clone() if mesh_data.uvs is not None else None
    vertex_normals = mesh_data.vertex_normals.clone() if mesh_data.vertex_normals is not None else None

    for _ in range(iterations):
        face_edges = torch.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), dim=1)
        sorted_edges = torch.sort(face_edges, dim=-1).values
        unique_edges, inverse = torch.unique(sorted_edges.reshape(-1, 2), dim=0, return_inverse=True)

        edge_midpoints = (vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]]) * 0.5
        midpoint_indices = torch.arange(
            vertices.shape[0],
            vertices.shape[0] + unique_edges.shape[0],
            device=faces.device,
            dtype=faces.dtype,
        )
        vertices = torch.cat((vertices, edge_midpoints), dim=0)

        if uvs is not None:
            edge_uvs = (uvs[unique_edges[:, 0]] + uvs[unique_edges[:, 1]]) * 0.5
            uvs = torch.cat((uvs, edge_uvs), dim=0)

        if vertex_normals is not None:
            edge_normals = F.normalize(
                (vertex_normals[unique_edges[:, 0]] + vertex_normals[unique_edges[:, 1]]) * 0.5,
                dim=-1,
                eps=1e-8,
            )
            vertex_normals = torch.cat((vertex_normals, edge_normals), dim=0)

        edge_ids = midpoint_indices[inverse].reshape(-1, 3)
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        m01, m12, m20 = edge_ids[:, 0], edge_ids[:, 1], edge_ids[:, 2]

        faces = torch.stack(
            (
                torch.stack((v0, m01, m20), dim=-1),
                torch.stack((v1, m12, m01), dim=-1),
                torch.stack((v2, m20, m12), dim=-1),
                torch.stack((m01, m12, m20), dim=-1),
            ),
            dim=1,
        ).reshape(-1, 3)

    # Create new MeshData with subdivided mesh
    return MeshData(
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        vertex_normals=vertex_normals,
        bvh=None  # BVH should be rebuilt if needed
    )


def map_vertices_positions(mesh_data: MeshData, hi_res_mesh_data: MeshData, weight: float = 1.0, *, inplace: bool = False) -> MeshData:
    """Moves vertex postions to positions mapped from high resolution mesh using BVH. Iterpolates between postions"""
    bvh = hi_res_mesh_data.bvh
    assert bvh is not None, "BVH must be built for high-res mesh"
    
    _, face_id, uvw = bvh.unsigned_distance(mesh_data.vertices, return_uvw=True)
    tris = hi_res_mesh_data.faces[face_id.long()]
    tri_verts = hi_res_mesh_data.vertices[tris]
    mapped_positions = (tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    new_vertices = mesh_data.vertices.mul_(1 - weight) if inplace else mesh_data.vertices.mul(1 - weight)
    new_vertices.add_(mapped_positions, alpha=weight)

    mapped_mesh = mesh_data if inplace else MeshData(vertices=new_vertices, faces=mesh_data.faces, uvs=mesh_data.uvs, vertex_normals=mesh_data.vertex_normals, bvh=mesh_data.bvh)

    if mapped_mesh.bvh is not None:
        mapped_mesh.bvh = None

    return mapped_mesh


def sort_mesh(mesh_data: MeshData, axes: Tuple[int] = (0,1,2), desc: Tuple[bool] | bool = (False, False, False)) -> MeshData:
    """
    Sorts mesh vertices lexicografically by axes.
    Each axes is sorted independently in either ascending (default) or descending order
    Then it sorts faces by min vertex index in new ordering.
    """
    vertices = mesh_data.vertices
    faces = mesh_data.faces

    if isinstance(desc, bool):
        desc = (desc,) * len(axes)

    perm = torch.arange(vertices.shape[0], device=vertices.device)
    for axis, reverse in zip(reversed(axes), reversed(desc)):
        key = vertices[perm, axis]
        order = torch.argsort(key, descending=reverse, stable=True)
        perm = perm[order]

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device, dtype=perm.dtype)

    sorted_vertices = vertices[perm]
    sorted_uvs = mesh_data.uvs[perm] if mesh_data.uvs is not None else None
    sorted_normals = mesh_data.vertex_normals[perm] if mesh_data.vertex_normals is not None else None

    remapped_faces = inv_perm[faces]
    face_min = remapped_faces.amin(dim=1)
    face_order = torch.argsort(face_min, stable=True)
    sorted_faces = remapped_faces[face_order]

    return MeshData(
        vertices=sorted_vertices,
        faces=sorted_faces,
        uvs=sorted_uvs,
        vertex_normals=sorted_normals,
        bvh=None,
    )
