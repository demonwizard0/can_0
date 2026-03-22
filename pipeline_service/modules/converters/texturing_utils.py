from typing import Tuple
import torch
import kaolin
import torch.nn.functional as F
from .schemas import AttributesMasked, MeshData, AttributeGrid, MeshRasterizationData
from flex_gemm.ops.grid_sample import grid_sample_3d


def rasterize_mesh_data(mesh_data: MeshData, texture_size: int | Tuple[int,int], use_vertex_normals: bool = False) -> MeshRasterizationData:
    """Rasterize the mesh shape onto the mesh UV space."""
    
    uvs = mesh_data.uvs
    assert uvs is not None, "UVs are rquired for rasterization"

    faces = mesh_data.faces
    vertices = mesh_data.vertices

    height, width = torch.Size(torch.as_tensor(texture_size).broadcast_to(2))

    
    # Prepare UVs
    uvs_ndc = uvs * 2 - 1
    uvs_ndc[:, 1] = -uvs_ndc[:, 1]
    uvs_ndc = uvs_ndc.unsqueeze(0) if uvs_ndc.dim() == 2 else uvs_ndc
    
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)
    faces = faces.long() if faces.dim() == 2 else faces.squeeze(0).long()

    surflets = vertices
    if use_vertex_normals:
        vertex_normals = mesh_data.vertex_normals.view_as(vertices)
        surflets = torch.cat((vertices, vertex_normals), dim=-1)

    # Index by faces
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(uvs_ndc, faces)
    face_vertex_suflets = kaolin.ops.mesh.index_vertices_by_faces(surflets, faces)

    batch_size, num_faces = face_vertices_image.shape[:2]

    face_vertices_z = torch.zeros(
        (batch_size, num_faces, 3),
        device=vertices.device,
        dtype=vertices.dtype
    )
    
    with torch.no_grad():
        surf_interpolated, face_idx = kaolin.render.mesh.rasterize(
            height=height,
            width=width,
            face_vertices_z=face_vertices_z,
            face_vertices_image=face_vertices_image,
            face_features=face_vertex_suflets,
            backend='cuda',
            multiplier=1000,
            eps=1e-8
        )
    
    surf = surf_interpolated[0]
    mask = face_idx[0] >= 0

    valid_surf = surf[mask]
    valid_positions, valid_normals = valid_surf[...,:3], valid_surf[...,3:]
    valid_normals = valid_normals if use_vertex_normals else None

    return MeshRasterizationData(face_ids=face_idx[0], positions=valid_positions, normals=valid_normals)


def map_mesh_rasterization(rast_data: MeshRasterizationData, mesh_data: MeshData, flip_vertex_normals: bool = False) -> MeshRasterizationData:

    bvh = mesh_data.bvh
    assert bvh is not None, "Mesh BVH needs to be build for mapping"
    valid_pos = rast_data.positions

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    tris = mesh_data.faces[face_id.long()]
    tri_verts = mesh_data.vertices[tris]  # (N_new, 3, 3)
    valid_positions = (tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    valid_normals = None

    if mesh_data.vertex_normals is not None:
        tri_norms = mesh_data.vertex_normals[tris]
        valid_normals = (tri_norms * uvw.unsqueeze(-1)).sum(dim=1)

        if flip_vertex_normals:
            flip_sign = (rast_data.normals * valid_normals).sum(dim=-1, keepdim=True).sign()
            valid_normals.mul_(flip_sign)
    
    return MeshRasterizationData(face_ids=rast_data.face_ids, positions=valid_positions, normals=valid_normals)


def sample_grid_attributes(rast_data: MeshRasterizationData, grid: AttributeGrid) -> AttributesMasked:
    
    voxel_size = grid.voxel_size
    aabb = grid.aabb
    coords = grid.coords
    attr_volume = grid.values

    valid_pos = rast_data.positions
    mask = rast_data.mask.to(grid.values.device)

    attrs = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=grid.dense_shape(),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        ).squeeze(0)
    
    return AttributesMasked(values=attrs, mask=mask)


def dilate_attributes(attributes: AttributesMasked, kernel_size: int) -> torch.Tensor:
    """Fill seams by dilating valid pixels into nearby empty UV space."""
    
    if kernel_size <= 1:
        return attributes

    attrs = attributes.to_dense().permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    mask = attributes.mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    invalid = ~mask
    pooled = F.unfold(attrs, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2).view(1, attrs.shape[1], kernel_size * kernel_size, -1)
    mask_unfold = F.unfold(mask.float(), kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2).view(1, 1, kernel_size * kernel_size, -1)
    # Get mean value of valid pixels in the kernel
    summed = pooled.mul_(mask_unfold).sum(dim=-2)
    count = mask_unfold.sum(dim=-2).clamp_min_(1.0)
    pooled = summed.div_(count).view_as(attrs).clamp_min_(0.0)

    filled = torch.where(invalid, pooled, attrs)
    return filled.squeeze(0).permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)


    


