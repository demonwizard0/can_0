import torch
import cumesh
from typing import Optional
from pydantic import BaseModel
from schemas.types import AnyTensor, IntegerTensor, FloatTensor, BoolTensor
from schemas.internal import Internal
from dataclasses import dataclass


class AttributeGrid(BaseModel):
    values: AnyTensor          # (N, K) attribute values for N voxels
    coords: IntegerTensor          # (N, 3) voxel coordinates on the grid
    aabb: FloatTensor          # (2, 3) axis-aligned bounding box (optional)
    voxel_size: FloatTensor    # (3,) size of voxel in each dimension

    @property
    def grid_size(self) -> FloatTensor:
        return ((self.aabb[1] - self.aabb[0]) / self.voxel_size).round().int()
    

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (self.values.shape[1], *self.grid_size.tolist()))
    

class AttributesMasked(BaseModel):
    values: AnyTensor    # (N, K) attribute values for N valid pixels
    mask: BoolTensor  # annotes which pixels are valid via boolean value

    def dense_shape(self, with_batch_size: bool = True) -> torch.Size:
        batch_size = (1,) if with_batch_size else ()
        return torch.Size(batch_size + (*self.mask.shape, self.values.shape[-1]))

    def to_dense(self, invalid: float = 0.0) -> torch.Tensor:
        size = self.dense_shape(with_batch_size=False)
        dense = self.values.new_full(size, fill_value=invalid)
        dense[self.mask] = self.values
        return dense

class MeshData(BaseModel):
    """Mesh geometry with vertices, faces, vertex normals and UVs."""
    vertices: FloatTensor                            # (V, 3) vertex positions
    faces: IntegerTensor                             # (F, 3) face indices
    vertex_normals: Optional[FloatTensor] = None     # (V, 3) vertex normals (optional)
    uvs: Optional[FloatTensor] = None                # (V, 2) UV coordinates (optional)
    bvh: Optional[Internal[cumesh.cuBVH]] = None     # BVH tree for ray tracing and projection

    def build_bvh(self):
        self.bvh = cumesh.cuBVH(self.vertices, self.faces)


class MeshDataWithAttributeGrid(MeshData):
    attrs: Optional[AttributeGrid] = None


class MeshRasterizationData(BaseModel):
    face_ids: IntegerTensor                # (H, W) indices of faces for each pixel (invalid=-1)
    positions: FloatTensor                 # (N_valid, 3) position of mesh for each valid pixel
    normals: Optional[FloatTensor] = None  # (N_valid, 3) surface normal for each valid pixel

    @property
    def mask(self) -> BoolTensor:
        return self.face_ids.ge(0)

