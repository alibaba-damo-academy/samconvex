import os
from datetime import datetime
from stat import S_IREAD
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from SAMReg.tools.interfaces import init_model
from SAMReg.tools.utils.general import (
    make_dir,
    set_seed_for_demo,
)
from SAMReg.cores.functionals import spatial_transformer, compose
from SAMReg.cores.MMConvex import MMConvex
import time
from preprocess import proc_image, get_shape, read_image, save_image, get_shape_pad, show_memoryUsage
import argparse
import torchio as tio
import math
from shutil import copyfile




if __name__ == "__main__":
    """
    Run Deformable registration.
    Arguments:
        --reference/ -r: the path to the target DICOM data folder or NII data
        --floating/ -f: the path to the source DICOM data folder or NII data
        --output_path/ -o: the path of output folder
    """
    parser = argparse.ArgumentParser(
        description="An interface for SAM-convex registration"
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True, type=str, default="", help="the path to the target DICOM data folder or NII data",
    )
    parser.add_argument(
        "-f",
        "--floating",
        required=True, type=str, default="", help="the path to the source DICOM data folder or NII data",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True, type=str, default="", help="the path of output folder",
    )
    args = parser.parse_args()
    print(args)
    set_seed_for_demo()

    ## Create experiment folder
    exp_folder = args.output_path
    make_dir(exp_folder)

    ## Setup GPU
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    device = torch.cuda.current_device()
    print(f"The current device: {torch.cuda.current_device()}")
   
    ## Setup model
    embed, cfg = init_model(config="configs/sam/sam_r18_i3d_fpn_1x_multisets_sgd_T_0.5_half_test.py",
                            checkpoint="iter_38000.pth")
    for param in embed.parameters():
        param.requires_grad = False
    embed.eval()
    model = MMConvex(embed)
    model = model.cuda()

    ## Load data
    tgt_img, tgt_img_info = read_image(args.reference)
    tgt_shape = get_shape(tgt_img, tgt_img_info)
    src_img, src_img_info = read_image(args.floating)
    src_shape = get_shape(src_img, src_img_info)
    
    pre_align_affine = None
    pre_align_flow = None
    print( f"Start register {args.floating} to {args.reference}") 

    ## Process data
    padding_shape, padding = get_shape_pad(tgt_shape, src_shape)
    tgt_sub, tgt_tio_affine = proc_image(tgt_img, tgt_img_info, tgt_shape, mask=None, pad_shape=padding_shape)
    src_sub, src_tio_affine = proc_image(src_img, src_img_info, src_shape, mask=None, pad_shape=padding_shape)
    target = tgt_sub["image"].data.unsqueeze(0).to(device).float() - 50.0
    source = src_sub["image"].data.unsqueeze(0).to(device).float() - 50.0

    # print(source.shape, target.shape)
    ######### 
    # source_arr = np.pad(source[0, 0].cpu().numpy(), [(0, 0), (0, 0), (1, 0)], mode='reflect')
    # source = torch.from_numpy(source_arr).unsqueeze(0).unsqueeze(0).to(device)
    # target_arr = np.pad(target[0, 0].cpu().numpy(), [(0, 0), (0, 0), (1, 0)], mode='reflect')
    # target = torch.from_numpy(target_arr).unsqueeze(0).unsqueeze(0).to(device)
    ######### 
    # print(source.shape, target.shape)
    

    ## SAMConvex registration
    start = time.time()
    source_, _, pre_align = model(source, target, pre_align_affine, pre_align_flow)
    warped, phi = model.instanceOptimization(source_, target)
    end = time.time()
    show_memoryUsage()
    print(f"Running time: {end - start}")

    # save_image(warped.permute(0, 1, 4, 3, 2)[0, 0].cpu().numpy(), os.path.join(exp_folder, 'warped.nii.gz'), tgt_img_info)

    grid = (
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0).cuda(),
                    source.shape,
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .flip(1)
        )
    phi = grid + compose((pre_align - grid), phi - grid)
   

    ## Apply to original image
    nii_img = nib.load(args.floating)
    original = torch.from_numpy(nii_img.get_data()).unsqueeze(0).unsqueeze(0).to(device)
    p2d = (1, math.ceil(padding[2] * 2 /  src_img_info['spacing'][2]),
           0, math.ceil(padding[1] * 2 /  src_img_info['spacing'][1]),
           0, math.ceil(padding[0] * 2 /  src_img_info['spacing'][0]))
    print(p2d, original.shape)
    original = F.pad(original, p2d, "constant", -3024)
    original_phi = F.interpolate(phi, size=original.shape[2:], mode="trilinear", align_corners=True)

    direc = np.array(tgt_img_info['origin_direction']).reshape((3,3))
    for axis in range(3):
        if direc[axis, axis] == -1:
            original = torch.flip(original, [2+axis])

    warped = spatial_transformer(original.float(), original_phi.float(), mode='bilinear', padding_mode="background")

    for axis in range(3):
        if direc[axis, axis] == -1:
            warped = torch.flip(warped, [2+axis])

    src_img_info['origin'] = tgt_img_info['origin']
    src_img_info['direction'] = tgt_img_info['origin_direction']
    save_image(warped[:,:,:,:,1:].permute(0, 1, 4, 3, 2)[0, 0].cpu().numpy(), os.path.join(exp_folder, 'warped.nii.gz'), src_img_info)
    copyfile(args.reference, os.path.join(exp_folder, 'target.nii.gz'))
    
    print(f"Finish running experiment at {exp_folder}")



