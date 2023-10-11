#!/usr/bin/env python3
# add neu3dviewer to the path, we need some helper function in it

# tips in ipython
#%reload_ext autoreload
#%autoreload 2

import numpy as np
from numpy.linalg import norm
import tifffile
import SimpleITK as sitk
import zarr
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel, title, figure

import sys
sys.path.append('/home/xyy/code/py/vtk_test/')
from neu3dviewer.data_loader import (
    LoadSWCTree, SplitSWCTree, SWCDFSSort, SWCNodeRelabel, GetUndirectedGraph,
    OnDemandVolumeLoader
)

_a  = lambda x: np.array(x, dtype=np.float64)
_ai = lambda x: np.array(x, dtype=int)
f_l_gamma = lambda a, g: np.uint16(((np.float64(a) - a.min()) / (a.max()-a.min())) **(1/g) * (a.max()-a.min()) + a.min())
imgshow = lambda im: plt.imshow(f_l_gamma(im, 3.0), cmap='gray', origin='lower')

def NormalSlice3DImage(img3d, p_center, vec_normal, vec_up):
    """
    Extract normal-plane image in a 3D image.
    @param img3d: numpy array, the 3D image in uniform spacing
    @param p_center: numpy array, the center point of the normal-plane
    @param vec_normal: numpy array, the normal vector of the normal-plane
    @param vec_up: vector denote the upper direction (y) of the image
    return img_normal: numpy array, the normal-plane image
    """

    # input index order is [x,y,z]
    # convert the image to sitk image
    # note that there is a transpose in GetImageFromArray
    img3ds = sitk.GetImageFromArray(img3d.T)
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
                               0, sitk.sitkUInt16)
    
    # convert the image to numpy array
    img_normal = sitk.GetArrayFromImage(img_normal)
    img_normal = img_normal[0, :, :].T
    return img_normal

def ShowThreeViews(img3d, p_center):
    # input (x,y,z) order
    img3d = img3d.T
    # now in (z,y,x) order

    figure(9)
    # fixed z, facing -z, draw x-y plane (horizontal, vertical)
    imgshow(img3d[int(p_center[2]), :, :])
    xlabel('x')
    ylabel('y')
    title(f"z = {int(p_center[2])}")

    figure(8)
    # x fixed, facing -x, draw y-z plane
    imgshow(img3d[:, :, int(p_center[0])])
    xlabel('y')
    ylabel('z')
    title(f"x = {int(p_center[0])}")

    figure(7)
    # y fixed, facing y, draw x-z plane
    imgshow(img3d[:, int(p_center[1]), :])
    xlabel('x')
    ylabel('z')
    title(f"y = {int(p_center[1])}")

def ShowThreeViewsMIP(img3d):
    # input (x,y,z) order
    img3d = img3d.T
    # now in (z,y,x) order

    figure(9)
    # facing -z, draw x-y plane (horizontal, vertical)
    imgshow(img3d.max(axis=0))
    xlabel('x')
    ylabel('y')
    title("MIP")

    figure(8)
    # facing -x, draw y-z plane
    imgshow(img3d.max(axis=2))
    xlabel('y')
    ylabel('z')
    title("MIP")

    figure(7)
    # facing y, draw x-z plane
    imgshow(img3d.max(axis=1))
    xlabel('x')
    ylabel('z')
    title("MIP")

def ExamBigImageContinuity():
    ### used in "order":"C" era

    p0 = _a([52785, 28145.6, 55668.9])
    p_corner = np.floor(p0 / 128) * 128
    img_block_path = '/mnt/xiaoyy/dataset/zarrblock'
    imgz = zarr.open(img_block_path, mode='r')
    idf_pb = lambda p, b: (slice(int(p[i]), int(p[i] + b[i]))
                            for i in range(3))
    block_size = _a([128, 128, 128])

    ## test direction of imshow
    # consider (z,y,x)
    figure(101)
    img_t = np.zeros((64, 64), dtype=np.uint16)
    img_t[0:10, 0:3] = 100   # (y,x)
    imgshow(img_t)
    xlabel('x')
    ylabel('y')

    # consider (x,y,z)
    figure(102)
    img_t = np.zeros((64, 64), dtype=np.uint16)
    img_t[0:10, 0:3] = 100   # (x,y)
    imgshow(img_t.T)
    xlabel('x')
    ylabel('y')
    
    ## test zarr

    # consider (x,y,z)
    img3d000 = imgz[*idf_pb(p_corner + _a([0,   0,   0]), block_size)]
    img3d010 = imgz[*idf_pb(p_corner + _a([0, 128,   0]), block_size)]
    img3d001 = imgz[*idf_pb(p_corner + _a([0,   0, 128]), block_size)]
    img3d011 = imgz[*idf_pb(p_corner + _a([0, 128, 128]), block_size)]

    figure(112)
    imgshow(img3d000.max(axis=0).T)
    xlabel('y')
    ylabel('z')

    # consider (z,y,x)
    img3d000 = imgz[*idf_pb(p_corner + _a([0,   0,   0]), block_size)]
    img3d010 = imgz[*idf_pb(p_corner + _a([0, 128,   0]), block_size)]
    img3d001 = imgz[*idf_pb(p_corner + _a([0,   0, 128]), block_size)]
    img3d011 = imgz[*idf_pb(p_corner + _a([0, 128, 128]), block_size)]

    # MIP along x
    figure(130)
    imgshow(f_l_gamma(img3d000.max(axis=2), 3.0))
    xlabel('y')
    ylabel('z')
    title('000')

    figure(131)
    imgshow(f_l_gamma(img3d001.max(axis=2), 3.0))
    xlabel('y')
    ylabel('z')
    title('001')

    figure(132)
    imgshow(f_l_gamma(img3d010.max(axis=2), 3.0))
    xlabel('y')
    ylabel('z')
    title('010')

    figure(133)
    imgshow(f_l_gamma(img3d011.max(axis=2), 3.0))
    xlabel('y')
    ylabel('z')
    title('011')

    # trying the plan that transpose the local array
    # convert to (x,y,z) order
    img3dfull = np.zeros((128,256,256), dtype=np.uint16)
    img3dfull[*idf_pb(_a([0,   0,   0]), block_size)] = img3d000.transpose()
    img3dfull[*idf_pb(_a([0,   0, 128]), block_size)] = img3d001.transpose()
    img3dfull[*idf_pb(_a([0, 128,   0]), block_size)] = img3d010.transpose()
    img3dfull[*idf_pb(_a([0, 128, 128]), block_size)] = img3d011.transpose()
    figure(134)
    imgshow(f_l_gamma(img3dfull.max(axis=0).T, 3.0))
    xlabel('y')
    ylabel('z')
    title('full')
    # good

    figure(135)
    # in .zarray
    #"order": "C",
    imgshow(f_l_gamma(
        imgz[*idf_pb(p_corner, (128, 256, 256))] \
        .max(axis=0).T, 3.0))
    xlabel('y')
    ylabel('z')
    title('zarr full')
    # not a very meaning full picture

    # Results above seems consistent:
    #   The small block in zarr is in (z,y,x) order
    #   But the indexing for zarr is in (x,y,z) order
    # Solution 1: always fetch zarr array in block, and transpose the result
    # Solution 2: rewrite the zarr array block to (x,y,z) order
    # Solution 3: tuning the .zarry configuration, set the order:C to F.
    # Adopt solution 3.

def WalkTreeTrial(swc_path, image_block_path):
    
    ## prepare fiber position

    # get an ordered and continuous node index tree and its graph
    ntree = LoadSWCTree(swc_path)
    processes = SplitSWCTree(ntree)
    ntree, processes = SWCDFSSort(ntree, processes)
    tr_idx = SWCNodeRelabel(ntree)
    ntree = (tr_idx, ntree[1])
    ngraph = GetUndirectedGraph(ntree)

    node_idx = 1936

    # show node position and neighbor indices
    print(ntree[0][node_idx, :], ntree[1][node_idx, :3])
    print(ngraph[node_idx].indices)

    neig_node_idx = ngraph[node_idx].indices
    # use SVD to get axis
    # neighbor point positions
    neig_pos = ntree[1][_ai(list(neig_node_idx)+[node_idx]), :3]
    print(neig_pos)
    u, s, vt = np.linalg.svd(neig_pos - neig_pos.mean(axis=0), full_matrices=True)

    p_focused = ntree[1][node_idx, :3]
    p_img_center = p_focused

    ## prepare image
    desired_block_size = (128, 128, 128)

    # load image around p_img_center
    imgz = zarr.open(image_block_path, mode='r')

    #p_img_corner = p_img_center - _a(desired_block_size) / 2

    p_img_corner = p_img_center.copy()
    p_img_corner[0] = np.floor(p_img_center[0] / 128) * 128
    p_img_corner[1] = p_img_center[1] - 128 / 2
    p_img_corner[2] = p_img_center[2] - 128 / 2

    # align to 128 boundary
    #p_img_corner = np.floor(p_img_center / 128) * 128

    idx_rg = [slice(int(p_img_corner[i]),
                    int(p_img_corner[i] + desired_block_size[i]))
              for i in range(3)]

    img3d = imgz[*idx_rg]

    print("p_img_center", p_img_center)
    print("p_img_corner", p_img_corner)
    print("diff", p_img_center - p_img_corner)

    #ShowThreeViews(img3d, p_img_center_s)
    ShowThreeViewsMIP(img3d)

    # load by lychnis block
    block_loader = OnDemandVolumeLoader()
    block_lym_path = 'RM009_traced_blocks/full_set/block.lym'
    block_loader.ImportLychnixVolume(block_lym_path)
    vol_sel = block_loader.LoadVolumeAt(p_img_center)
    print(vol_sel)
    figure(30)
    img_lym = tifffile.imread(vol_sel[0]['image_path']).T
    imgshow(img_lym.max(axis=0).T)
    xlabel('y')
    ylabel('z')
    title('lym block')

    # substract every coor by p_img_corner to align to the image
    #p_img_center_s = p_img_center - p_img_corner

    #img_tangent = NormalSlice3DImage(img3d, p_img_center_s, vt[2], -u[1])

    #figure(20)
    #imgshow(img_tangent)

def Test3dImageSlicing():
    # load the 3D image
    tif_path = "/home/xyy/code/py/neurite_walker/52224-30976-56064.tif"
    img3d = tifffile.imread(tif_path).T

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

    figure(10)
    # rescale the image by min and max
    #img_normal = (img_normal - img_normal.min()) / (img_normal.max() - img_normal.min())
    imgshow(img_normal.T)
    title(f'slice: cen{p_center}, nor{vec_normal}, up{vec_up}')
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
    
    plt.ion()

    WalkTreeTrial(swc_path, img_block_path)

    plt.show()
