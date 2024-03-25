#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
#from train import percent_10_index 

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
      
    #print("RGB:")
    #print(colors_precomp.shape)

    #Multiply by -1 for second half of Gaussians
    #size_colors_precomp = colors_precomp.size(0)
    #colors_precomp[(size_colors_precomp//2):size_colors_precomp] = -(colors_precomp[(size_colors_precomp//2):size_colors_precomp])

    #Multiply by -1 first 10% of Gaussians
    #colors_precomp[:pc.percent_10_index] *= -1
    colors_precomp[:30000] *= -1

    #print("pc.percent_10_index: ")
    #print(pc.percent_10_index);
    #print()

    """
    print("Before resterizer: ")
    index = 0
    for i in range(0, 33):
      for j in range(0, 3):
          print(index, ":", end=' ')
          print(colors_precomp[i][j])
          index+=1
    """

    #Test1
    #colors_precomp[:30000] *= -1

    #Test2
    #colors_precomp[:30000] *= -1
    #colors_precomp[30000:] *= -1

    #Test3
    """
    nGausses = means3D.shape[0]
    print(nGausses)
    means3D [((nGausses+1)//2):nGausses] = means3D [0:(nGausses//2)]
    means2D [((nGausses+1)//2):nGausses] = means2D [0:(nGausses//2)]
    if shs != None:
      shs[((nGausses+1)//2):nGausses] = shs[0:(nGausses//2)]
    colors_precomp [((nGausses+1)//2):nGausses] = colors_precomp[0:(nGausses//2)]
    opacity [((nGausses+1)//2):nGausses] = opacity[0:(nGausses//2)]
    scales [((nGausses+1)//2):nGausses] = scales [0:(nGausses//2)]
    rotations [((nGausses+1)//2):nGausses] = rotations [0:(nGausses//2)]
    if cov3D_precomp != None:
      cov3D_precomp[((nGausses+1)//2):nGausses] = cov3D_precomp [0:(nGausses//2)]

    for i in range(nGausses//2):
      colors_precomp[i] *= -1
    """

    """
    #Negative Gausses - red coloring
    for i in range(30000):
      colors_precomp[i][0] = 1.0
      colors_precomp[i][1] = 0
      colors_precomp[i][2] = 0
    """

    """
    #Save opacity to file
    import numpy as np
    np.savetxt('opacity.out', opacity.cpu().numpy(), delimiter=',')   # X is an array
    """


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    #print("Rendered image: ")
    #torch.set_printoptions(profile="full")
    #print(rendered_image) # prints the whole tensor
    #torch.set_printoptions(profile="default") # reset
    #print()


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}




