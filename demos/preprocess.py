import numpy as np
import torch
import torch.nn.functional as F
from SAMReg.tools.utils.med import read_image, seg_bg_mask, save_image, seg_bed
from sam.datasets.piplines import Resample
import torchio as tio
import nibabel as nib
import os
import itk
import scipy
import psutil

    
class RescaleIntensity():
    def __init__(self, out_min_max=(0, 255), in_min_max=(-1024, 3071)):
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max

    def __call__(self, data):
        rescale_transform = tio.RescaleIntensity(out_min_max=self.out_min_max, in_min_max=self.in_min_max)
        data["subject"]["image"] = rescale_transform(data["subject"]["image"])
        return data

   
def show_memoryUsage():
    print(u'Mem usage ：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


def is_odd(x):
    return x % 2 == 1


def get_shape(img, img_info):
    size =(round(img.shape[1] * img_info['spacing'][0])// 2, 
                round(img.shape[2] * img_info['spacing'][1])// 2,
                round(img.shape[0] * img_info['spacing'][2])// 2) 
    return size


def get_shape_pad(tgt_shape, src_shape):
    assert len(tgt_shape) == 3, "The shape of the image should be three."
    
    x = max(tgt_shape[0], src_shape[0])
    y = max(tgt_shape[1], src_shape[1])
    z = max(tgt_shape[2], src_shape[2])
    size = [x, y, z]

    padding = []
    for i in range(3):
        if src_shape[i] < size[i]:
            sub = size[i] - src_shape[i]
            # if is_odd(sub):
            #     padding.append(sub + 1)
            #     size[i] += 1
            # else:
            #     padding.append(sub)
            padding.append(sub)
        else:
            padding.append(0)

    # return np.array(size), (np.array(padding)/2).astype(int)
    return np.array(size), np.array(padding)


def proc_image(im, im_info, shape, mask=None, pad_shape=None):
    assert np.all(np.reshape(im_info['direction'], (3, 3)) == np.eye(3)), f'unsupported direction!'

    img_data = torch.from_numpy(im).permute(2, 1, 0)[None]
    tio_affine = np.hstack((np.diag(im_info['spacing']), np.array(im_info['origin'])[:, None]))
    tio_affine = np.vstack((tio_affine, [0, 0, 0, 1]))

    # Prepare subject data
    subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_data, affine=tio_affine)
        )

    if mask is not None:
        subject["mask"] = tio.LabelMap(
                        tensor=torch.from_numpy(mask.astype(np.int8)).permute(2,1,0)[None],
                        affine=tio_affine)

    data = {}
    data['image_fn'] = im_info['im_path']
    data['subject'] = subject
    
    # Resample
    resample = Resample()
    data = resample(data)

    # # pad or crop
    # if pad_shape is not None:
    #     if not np.array_equal(pad_shape, im.shape):
    #         transform = tio.CropOrPad(pad_shape, mask_name="mask", padding_mode=-1000)
    #         data["subject"]["image"] = transform(data["subject"]["image"])
    #         if mask is not None:
    #             transform_mask = tio.CropOrPad(pad_shape, mask_name="mask", padding_mode=0)
    #             data["subject"]["mask"] = transform_mask(data["subject"]["mask"])
    
    # pad or crop
    if pad_shape is not None:
        if not np.array_equal(pad_shape, shape):
            padding = (0, pad_shape[0]-shape[0], 
                       0, pad_shape[1]-shape[1], 
                       0, pad_shape[2]-shape[2])
            transform = tio.Pad(padding, padding_mode=-1000)
            data["subject"]["image"] = transform(data["subject"]["image"])
            if mask is not None:
                transform_mask = tio.Pad(padding, padding_mode=0)
                data["subject"]["mask"] = transform_mask(data["subject"]["mask"])

    rescale = RescaleIntensity()
    data = rescale(data)

    return data['subject'], tio_affine


def proc_image_for_deeds(im, im_info, shape, mask=None, pad_shape=None):
    assert np.all(np.reshape(im_info['direction'], (3, 3)) == np.eye(3)), f'unsupported direction!'

    img_data = torch.from_numpy(im).permute(2, 1, 0)[None]
    tio_affine = np.hstack((np.diag(im_info['spacing']), np.array(im_info['origin'])[:, None]))
    tio_affine = np.vstack((tio_affine, [0, 0, 0, 1]))

    # Prepare subject data
    subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_data, affine=tio_affine)
        )

    if mask is not None:
        subject["mask"] = tio.LabelMap(
                        tensor=torch.from_numpy(mask.astype(np.int8)).permute(2,1,0)[None],
                        affine=tio_affine)

    data = {}
    data['image_fn'] = im_info['im_path']
    data['subject'] = subject
    
    # Resample
    resample = Resample(norm_spacing=(2., 2., 2.))
    data = resample(data)
    
    # pad or crop
    if pad_shape is not None:
        padding = (0, pad_shape[0]-shape[0], 
                    0, pad_shape[1]-shape[1], 
                    0, pad_shape[2]-shape[2])
        print('Debug', padding)
        transform = tio.Pad(padding, padding_mode=-1000)
        data["subject"]["image"] = transform(data["subject"]["image"])

        # ####
        # del transform
        # transform = tio.CropOrPad(pad_shape, mask_name="mask", padding_mode=-1000)
        # data["subject"]["image"] = transform(data["subject"]["image"])
        # ###

        if mask is not None:
            transform_mask = tio.Pad(padding, padding_mode=0)
            data["subject"]["mask"] = transform_mask(data["subject"]["mask"])
            # ####
            # del transform_mask
            # transform_mask = tio.CropOrPad(pad_shape, mask_name="mask", padding_mode=0)
            # data["subject"]["mask"] = transform_mask(data["subject"]["mask"])
            # ###

    return data['subject'], tio_affine


   

# def mask_gen(img, threshold=-900, num_closing=5, num_opening=5, num_smo=3):

#     mask = (img>threshold).float()
    
#     for _ in range(num_closing):
#         mask = dilate(mask, ksize=5)

#     for _ in range(num_closing):
#         mask = erode(mask, ksize=5)

#     for _ in range(num_opening):
#         mask = erode(mask, ksize=5)
    
#     for _ in range(num_opening):
#         mask = dilate(mask, ksize=5)

#     for _ in range(num_smo):
#         mask = F.avg_pool3d(mask, kernel_size=5, stride=1, padding=5//2)

#     return mask

def mask_gen(img, threshold=-900, num_closing=9, num_opening=5):
    print(img.min(), img.max())

    mask = (img>threshold).float()
    
    for _ in range(num_opening):
        mask = erode(mask)

    for _ in range(num_opening):
        mask = dilate(mask)

    for _ in range(num_closing):
        mask = dilate(mask)

    for _ in range(num_closing):
        mask = erode(mask)

    mask = F.max_pool3d(mask, kernel_size=5, stride=1, padding=5//2)
    
    largest_mask = scipy.ndimage.binary_fill_holes(mask[0, 0].cpu().numpy())

    return largest_mask


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    out = F.max_pool3d(bin_img, kernel_size=ksize, stride=1, padding=pad)
    return out


def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

