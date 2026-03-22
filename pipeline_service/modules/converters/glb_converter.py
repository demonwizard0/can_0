import time
from typing import Dict, Tuple
from logger_config import logger
import numpy as np
import torch
import kaolin
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import torch.nn.functional as F
from .enums import AlphaMode
from .utils import bayer_dither_pattern
from .schemas import MeshData, MeshDataWithAttributeGrid, AttributeGrid
from libs.trellis2.representations.mesh.base import MeshWithVoxel
from modules.converters.mesh_utils import sort_mesh, subdivide, map_vertices_positions
from modules.converters.texturing_utils import dilate_attributes, map_mesh_rasterization, rasterize_mesh_data, sample_grid_attributes
from .params import GLBConverterParams
from .settings import GLBConverterConfig
import cumesh


DITHER_PATTERN_SIZE = 16
DITHER_PATTERN = (bayer_dither_pattern(4096, 4096, DITHER_PATTERN_SIZE) * 255).astype(np.uint8)


class GLBConverter:
    """Converter for extracting and texturing meshes to GLB format."""
    DEFAULT_AABB = torch.as_tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    DILATION_KERNEL_SIZE = 5
    
    def __init__(self, settings: GLBConverterConfig):
        """Initialize converter with settings."""
        self.default_params = GLBConverterParams.from_settings(settings)
        self.logger = logger
        self.device = torch.device(f'cuda:{settings.gpu}' if torch.cuda.is_available() else 'cpu')
    
    def convert(self, mesh: MeshWithVoxel, aabb: torch.Tensor = DEFAULT_AABB, params: GLBConverterParams = None) -> trimesh.Trimesh:
        """Convert the given mesh to a textured GLB format."""
        logger.debug(f"Original mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        
        params = self.default_params.overrided(params)
        logger.debug(f"Using GLB conversion parameters: {params}")

        # 1. Prepare original mesh data with BVH
        original_mesh_data = self._prepare_original_mesh(mesh, aabb)
        
        # 2. Remesh if required (cleanup otherwise)
        if params.remesh:
            mesh_data = self._remesh_mesh(original_mesh_data, params)
        else:
            mesh_data = self._cleanup_mesh(original_mesh_data, params)
            
        # 3. UV unwrap the mesh
        mesh_data = self._uv_unwrap_mesh(mesh_data, params)

        # 4. subdivide unwrapped mesh
        mesh_data = self._subdivide_mesh(mesh_data, original_mesh_data, params)
        
        # 5. Rasterize attributes onto the mesh UVs
        attributes = self._rasterize_attributes(mesh_data, original_mesh_data, params)
        
        # 6. Post-process the rasterized attributes into textures
        base_color, metallic_roughness = self._texture_postprocess(attributes, mesh.layout, params)

        # 7. Create the textured mesh
        textured_mesh = self._create_textured_mesh(mesh_data, base_color, metallic_roughness, params)

        return textured_mesh


    def _prepare_original_mesh(self, mesh: MeshWithVoxel, aabb: torch.Tensor, compute_vertex_normals: bool = False) -> MeshDataWithAttributeGrid:
        """
        Convert MeshWithVoxel to OriginalMeshData.
        WARNING: This method also fills holes outputing additional faces compared to input one.
        """
        logger.debug(f"Preparing original mesh data")
        start_time = time.time() 

        # Prepare attribute grid
        attrs = AttributeGrid(
            values=mesh.attrs.to(self.device),
            coords=mesh.coords.to(self.device),
            aabb = torch.as_tensor(aabb, dtype=torch.float32, device=self.device),
            voxel_size = torch.as_tensor(mesh.voxel_size, dtype=torch.float32, device=self.device).broadcast_to(3)
        )

        vertices = mesh.vertices.to(self.device)
        faces = mesh.faces.to(self.device)
        
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(vertices, faces)
        cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
        logger.debug(f"After filling holes: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")

        vertices, faces = cumesh_mesh.read()
        vertex_normals = None

        if compute_vertex_normals:
            cumesh_mesh.compute_vertex_normals()
            vertex_normals = cumesh_mesh.read_vertex_normals()

        original_mesh_data = MeshDataWithAttributeGrid(vertices=vertices, faces=faces,vertex_normals=vertex_normals, attrs=attrs)
        
        # Build BVH for the current mesh to guide remeshing
        logger.debug(f"Building BVH for current mesh...")
        original_mesh_data.build_bvh()
        logger.debug(f"Done building BVH | Time: {time.time() - start_time:.2f}s")
        
        return original_mesh_data

    def _cleanup_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Cleanup and optimize the mesh using decimation and remeshing."""
        # Create cumesh from current mesh data
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)
        
        # Step 1: Aggressive simplification (3x target)
        cumesh_mesh.simplify(params.decimation_target * 3, verbose=False)
        logger.debug(f"After initial simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        cumesh_mesh.remove_duplicate_faces()
        cumesh_mesh.repair_non_manifold_edges()
        cumesh_mesh.remove_small_connected_components(1e-5)
        cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
        logger.debug(f"After initial cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        cumesh_mesh.simplify(params.decimation_target, verbose=False)
        logger.debug(f"After final simplification: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        cumesh_mesh.remove_duplicate_faces()
        cumesh_mesh.repair_non_manifold_edges()
        cumesh_mesh.remove_small_connected_components(1e-5)
        cumesh_mesh.fill_holes(max_hole_perimeter=3e-2)
        logger.debug(f"After final cleanup: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        cumesh_mesh.unify_face_orientations()
        
        # Extract cleaned mesh data
        vertices, faces = cumesh_mesh.read()
        return MeshData(
            vertices=vertices,
            faces=faces
        )

    def _remesh_mesh(self, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Remesh the given mesh to improve quality."""
        # Create cumesh from current mesh data
        logger.debug("Starting remeshing")
        start_time = time.time()
        
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(original_mesh_data.vertices, original_mesh_data.faces)

        voxel_size = original_mesh_data.attrs.voxel_size
        aabb = original_mesh_data.attrs.aabb
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

        resolution = grid_size.max().item()
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        
        # Perform Dual Contouring remeshing (rebuilds topology)
        vertices, faces = cumesh.remeshing.remesh_narrow_band_dc(
            *cumesh_mesh.read(),
            center=center,
            scale=(resolution + 3 * params.remesh_band) / resolution * scale,
            resolution=resolution,
            band=params.remesh_band,
            project_back=params.remesh_project,  # Snaps vertices back to original surface
            verbose=False,
            bvh=original_mesh_data.bvh,
        )
        cumesh_mesh.init(vertices, faces)
        logger.debug(f"After remeshing: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        # Simplify and clean the remeshed result
        cumesh_mesh.simplify(params.decimation_target, verbose=False)
        logger.debug(f"After simplifying: {cumesh_mesh.num_vertices} vertices, {cumesh_mesh.num_faces} faces")
        
        # Extract remeshed data
        vertices, faces = cumesh_mesh.read()
        
        logger.debug(f"Done remeshing | Time: {time.time() - start_time:.2f}s")
        return MeshData(
            vertices=vertices,
            faces=faces
        )

    def _uv_unwrap_mesh(self, mesh_data: MeshData, params: GLBConverterParams) -> MeshData:
        """Perform UV unwrapping on the mesh."""
        # Create cumesh from current mesh data
        logger.debug("Starting UV unwrapping")
        start_time = time.time()
        
        cumesh_mesh = cumesh.CuMesh()
        cumesh_mesh.init(mesh_data.vertices, mesh_data.faces)
        
        xatlas_compute_charts_kwargs = {
            "max_chart_area": 1.0,
            "max_boundary_length": 2.0,
            "max_cost": 10.0,
            "normal_seam_weight": 5.0,
            "normal_deviation_weight": 1.0,
            "fix_winding": True
        }
        
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh_mesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": np.radians(params.mesh_cluster_threshold_cone_half_angle),
                "refine_iterations": params.mesh_cluster_refine_iterations,
                "global_iterations": params.mesh_cluster_global_iterations,
                "smooth_strength": params.mesh_cluster_smooth_strength,
            },
            xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
            return_vmaps=True,
            verbose=True,
        )
        out_vertices = out_vertices.to(self.device)
        out_faces = out_faces.to(self.device)
        out_uvs = out_uvs.to(self.device)
        out_vmaps = out_vmaps.to(self.device)
        
        cumesh_mesh.compute_vertex_normals()
        out_normals = cumesh_mesh.read_vertex_normals()[out_vmaps]
        
        logger.debug(f"Done UV unwrapping | Time: {time.time() - start_time:.2f}s")
        
        return MeshData(
            vertices=out_vertices,
            faces=out_faces,
            vertex_normals=out_normals,
            uvs=out_uvs
        )

    def _subdivide_mesh(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> MeshData:
        """Subdivide mesh with uv data and optionally reproject vertices to original mesh surface."""
        subdivided_mesh = subdivide(mesh_data, iterations=params.subdivisions)
        
        if params.vertex_reproject > 0.0:
            subdivided_mesh = map_vertices_positions(subdivided_mesh, original_mesh_data, weight=params.vertex_reproject, inplace=True)

        subdivided_mesh = sort_mesh(subdivided_mesh, axes=(2,1,0))
        
        return subdivided_mesh

    def _rasterize_attributes(self, mesh_data: MeshData, original_mesh_data: MeshDataWithAttributeGrid, params: GLBConverterParams) -> torch.Tensor:
        """Rasterize the given attributes onto the mesh UVs."""
        logger.debug("Sampling attributes(Texture rasterization)")
        start_time = time.time()

        # Rasterize mesh surface
        rast_data = rasterize_mesh_data(mesh_data, params.texture_size)
        logger.debug(f"Texture baking: sampling {rast_data.positions.shape[0]} valid pixels out of {params.texture_size * params.texture_size}")
        logger.debug(f"Attribute volume has {original_mesh_data.attrs.values.shape[0]} voxels")

        # Map these positions back to the *original* high-res mesh to get accurate attributes
        # This corrects geometric errors introduced by simplification/remeshing
        rast_data = map_mesh_rasterization(rast_data, original_mesh_data)

        # Trilinear sampling from the attribute volume (Color, Material props)
        attributes = sample_grid_attributes(rast_data, original_mesh_data.attrs)

        # Fill seams by dilating valid pixels into nearby empty UV space
        attrs = dilate_attributes(attributes, self.DILATION_KERNEL_SIZE)

        logger.debug(f"Done attribute sampling | Time: {time.time() - start_time:.2f}s")
        
        return attrs
    
    def _texture_postprocess(self, attributes: torch.Tensor, attr_layout: Dict, params: GLBConverterParams) -> Tuple[Image.Image, Image.Image]:
        """Post-process the rasterized attributes into final textures."""
        logger.debug("Finalizing mesh textures")
        start_time = time.time()
        
        # Extract channels based on layout (BaseColor, Metallic, Roughness, Alpha)
        base_color = np.clip(attributes[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attributes[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attributes[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        # Adjust alpha with gamma
        alpha = attributes[..., attr_layout['alpha']].clamp(0,1).pow_(params.alpha_gamma)
        alpha = np.clip(alpha.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # Handle alpha mode
        alpha_mode = params.alpha_mode
        if alpha_mode == AlphaMode.BLEND:
            alpha_mode = AlphaMode.OPAQUE if np.all(alpha == 255) else alpha_mode


        # Apply alpha dithering if flag is set
        if alpha_mode == AlphaMode.DITHER:
            h, w = alpha.shape[:2]
            dither_pattern = DITHER_PATTERN[:h, :w, None]
            alpha = ((alpha > dither_pattern) * 255).astype(np.uint8)
            logger.debug(f"Dithered alpha channel has {np.sum(alpha == 0)} transparent pixels out of {alpha.size} total pixels")
            # alpha_mode = AlphaMode.MASK : After dithering, treat as MASK

        base_color_texture = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallic_roughness_texture = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        
        logger.debug(f"Done finalizing mesh textures | Time: {time.time() - start_time:.2f}s")
        return base_color_texture, metallic_roughness_texture

    def _create_textured_mesh(self, mesh_data: MeshData, base_color: Image.Image, metallic_roughness: Image.Image, params: GLBConverterParams) -> trimesh.Trimesh:
        """Create a textured trimesh mesh from the mesh data and textures."""
        
        logger.debug("Creating textured mesh")
        start_time = time.time()

        alpha_mode = params.alpha_mode
        alpha_mode = AlphaMode.MASK if alpha_mode is AlphaMode.DITHER else alpha_mode

        # Create PBR material
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=base_color,
            baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
            metallicRoughnessTexture=metallic_roughness,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=alpha_mode.value,
            alphaCutoff=alpha_mode.cutoff,
            doubleSided=bool(not params.remesh)
        )
        
        # --- Coordinate System Conversion & Final Object ---
        vertices_np = mesh_data.vertices.mul(params.rescale).cpu().numpy()
        faces_np = mesh_data.faces.cpu().numpy()
        uvs_np = mesh_data.uvs.cpu().numpy()
        normals_np = mesh_data.vertex_normals.cpu().numpy()
        
        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
        uvs_np[:, 1] = 1 - uvs_np[:, 1]  # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
        )
        
        logger.debug(f"Done creating textured mesh | Time: {time.time() - start_time:.2f}s")
                
        return textured_mesh
