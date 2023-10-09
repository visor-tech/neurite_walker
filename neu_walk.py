#!/usr/bin/env python3
# add neu3dviewer to the path, we need some helper function in it

import numpy as np
from numpy.linalg import norm
import tifffile
import SimpleITK as sitk
import zarr
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/xyy/code/py/vtk_test/')
from neu3dviewer.data_loader import (
    LoadSWCTree, SplitSWCTree, SWCDFSSort, SWCNodeRelabel, GetUndirectedGraph
)

_a  = lambda x: np.array(x, dtype=np.float64)
_ai = lambda x: np.array(x, dtype=int)

def NormalSlice3DImage(img3d, p_center, vec_normal, vec_up):
    """
    Extract normal-plane image in a 3D image.
    @param img3d: numpy array, the 3D image in uniform spacing
    @param p_center: numpy array, the center point of the normal-plane
    @param vec_normal: numpy array, the normal vector of the normal-plane
    @param vec_up: vector denote the upper direction (y) of the image
    return img_normal: numpy array, the normal-plane image
    """

    # convert the image to sitk image
    # note that the index ordering is changed from [z,y,x] to [x,y,z]
    img3ds = sitk.GetImageFromArray(img3d)
    img3ds.SetOrigin((0,0,0))
    img3ds.SetSpacing((1,1,1))
    img3ds.SetDirection((1,0,0, 0,1,0, 0,0,1))

    sz = (img3d.shape[0], img3d.shape[1], 1)
    trfm = sitk.Transform()  # default to identity

    out_origin = _a(p_center)  # make sure float64
    out_spacing = (1, 1, 1)
    vec_normal = vec_normal / norm(vec_normal)   # let the devided-by-zero raise error
    vec_up = vec_up / norm(vec_up)
    vec_x = np.cross(vec_up, vec_normal)
    vec_y = np.cross(vec_normal, vec_x)
    # row-major order, but vectors are column vectors
    out_direction = _a([vec_x, vec_y, vec_normal]).T.flatten(order='C')
    # shift out_origin from image corner to center
    out_origin = out_origin - vec_x * sz[0] / 2 - vec_y * sz[1] / 2

    # use resampling filter to get the normal-plane image
    img_normal = sitk.Resample(img3ds, sz, trfm, sitk.sitkLinear,
                               out_origin, out_spacing, out_direction,
                               10000, sitk.sitkUInt16)
    
    # convert the image to numpy array
    img_normal = sitk.GetArrayFromImage(img_normal)
    img_normal = img_normal[0, :, :]
    return img_normal

def WalkTreeTrial(swc_path, image_block_path):
    # get an ordered and continuous node index tree and its graph
    ntree = LoadSWCTree(swc_path)
    processes = SplitSWCTree(ntree)
    ntree, processes = SWCDFSSort(ntree, processes)
    tr_idx = SWCNodeRelabel(ntree)
    ntree = (tr_idx, ntree[1])
    ngraph = GetUndirectedGraph(ntree)

    node_idx = 1936

    # show node position and neighbor indices
    print(ntree[0][node_idx, :], ntree[1][node_idx, :])
    print(ngraph[node_idx].indices)

    neig_node_idx = ngraph[node_idx].indices
    # use SVD to get axis
    # neighbor point positions
    neig_pos = ntree[1][_ai(list(neig_node_idx)+[node_idx]), :3]
    print(neig_pos)
    u, s, vt = np.linalg.svd(neig_pos - neig_pos.mean(axis=0), full_matrices=True)

    p_focused = ntree[1][node_idx, :3]
    p_img_center = p_focused

    desired_block_size = (128, 128, 128)

    # load image around p_img_center
    imgz = zarr.open(image_block_path, mode='r')
    print(p_img_center)
    p_img_corner = p_img_center - _a(desired_block_size) / 2
    idx_rg = [slice(int(p_img_corner[i]),
                    int(p_img_corner[i] + desired_block_size[i]))
              for i in range(3)]
    print(idx_rg)
    img3d = imgz[*idx_rg]

    print(p_img_center)
    print(img3d.shape)
    print(neig_pos[-1])

    img_tangent = NormalSlice3DImage(img3d, neig_pos[-1], vt[2], -u[1])

def ShowThreeViews(img3d, p_center):
    plt.figure(9)
    # fixed z, facing -z, draw x-y plane (horizontal, vertical)
    plt.imshow(img3d[int(p_center[2]), :, :], cmap='gray', origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"z = {int(p_center[2])}")

    plt.figure(8)
    # x fixed, facing -x, draw y-z plane
    plt.imshow(img3d[:, :, int(p_center[0])], cmap='gray', origin='lower')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title(f"x = {int(p_center[0])}")

    plt.figure(7)
    # y fixed, facing y, draw x-z plane
    plt.imshow(img3d[:, int(p_center[1]), :], cmap='gray', origin='lower')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(f"y = {int(p_center[1])}")

def Test3dImageSlicing():
    # load the 3D image
    tif_path = "/home/xyy/code/py/neurite_walker/52224-30976-56064.tif"
    img3d = tifffile.imread(tif_path)

    # set the center point and normal vector
    p_center   = _a([30, 90, 40])
    vec_normal = _a([1, 0, 0])  # vectors always follow (x,y,z)
    vec_up     = _a([0, 0, 1])
    #vec_normal = _a([0, 0, 1])  # vectors always follow (x,y,z)
    #vec_up     = _a([0, 1, 0])
    # get the normal-plane image
    img_normal = NormalSlice3DImage(img3d, p_center, vec_normal, vec_up)

    # show the reference image
    #plt.ion()
    ShowThreeViews(img3d, p_center)

    plt.figure(10)
    # rescale the image by min and max
    #img_normal = (img_normal - img_normal.min()) / (img_normal.max() - img_normal.min())
    plt.imshow(img_normal, cmap='gray', origin='lower')
    plt.show()

if __name__ == '__main__':
    #Test3dImageSlicing()

    # node_idx = 1936, node_id = 932514
    # xyz: [52785.  28145.6 55668.9]
    # Branch depth: 1
    # Node depth: 1254
    # Path length to root: 23228.5
    swc_path = 'neuron#255.lyp.swc'
    #block_lym_path = 'RM009_traced_blocks/full_set/block.lym'
    img_block_path = '/mnt/xiaoyy/dataset/zarrblock'
    WalkTreeTrial(swc_path, img_block_path)
