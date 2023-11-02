#!/usr/bin/env python3

## Before run:
# * Prepare a directory named "pic_tmp" to hold output cMIP images
# * Put external dependencies in directory "external", such as external/neu3dviewer
# * See requirements.txt for required python packages.
# * need python3.11 or higher

## Usage:
# To generate cMIP for a neuron, run the following command:
# python neu_walk.py <neuron_swc_path>

# To generate cMIP for a batch of neurons, run the following command (Linux only):
# find <path_to_swc_directory> -type f -print0 | xargs -0 -P 8 -n 1 ./neu_walk.py --zarr_dir <zarr_dir> --cmip_dir <cmip_dir>

# To view the resulting cMIP, run:
# python neu_walk.py --cmip_dir <path_to_cMIP_images_directory> --view <swc_path>

# for more options, see python neu_walk.py -h

# tips in ipython
#%reload_ext autoreload
#%autoreload 2

# TODO:
# * Add context menu to plot, allow choose error type, and more natural interaction
#   - https://matplotlib.org/stable/gallery/widgets/menu.html
#   - or try fig.add_axes([0.7, 0.05, 0.1, 0.075]) to add a button
#   - https://matplotlib.org/stable/gallery/widgets/buttons.html
#   See https://matplotlib.org/stable/gallery/widgets/menu.html
# * Try multi-threading call to neu3dviewer, each with its own gui_ctrl.
# * Try update gui_ctrl in a seperate thread, faster response. q to minimize.
# * test interpolator for sitk.Resample: sitkLanczosWindowedSinc, sitkGaussian
# * Try Range slider: https://matplotlib.org/stable/gallery/widgets/range_slider.html

import os
import sys
import glob   # for list files
import argparse
from datetime import datetime
import json

import numpy as np
from numpy import diff, sin, cos, pi, linspace
from numpy.linalg import norm
import scipy.ndimage
import scipy.interpolate as interpolate

import tifffile
import zarr
import SimpleITK as sitk

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel, title, figure

plt.rcParams['keymap.save'] = ['ctrl+s']
plt.rcParams['keymap.quit'] = ['ctrl+w', 'cmd+w']

# add path of current py file
pkg_path_neu3dviewer_rel = 'external/neu3dviewer'
cur_path = os.path.dirname(os.path.abspath(__file__))
pkg_path_neu3dviewer = os.path.join(cur_path, pkg_path_neu3dviewer_rel)
sys.path.append(pkg_path_neu3dviewer)
# add neu3dviewer to the path, we need some helper functions in it
import neu3dviewer.utils
from neu3dviewer.img_block_viewer import GUIControl
from neu3dviewer.data_loader import (
    dtype_id, dtype_coor,
    LoadSWCTree, SplitSWCTree, SWCDFSSort, SWCNodeRelabel, GetUndirectedGraph,
    SimplifyTreeWithDepth,
    OnDemandVolumeLoader
)
from neu3dviewer.utils import ArrayfyList

# utility functions
_a  = lambda x: np.array(x, dtype=np.float64)
_ai = lambda x: np.array(x, dtype=int)
_va = lambda *a: np.vstack(a)
_ha = lambda *a: np.hstack(a)   # concatenate along horizontal axis
imgshow = lambda im, **kwval: plt.imshow(f_l_gamma(im, 3.0), cmap='gray', origin='lower', **kwval)

def f_l_gamma(a, g):
    if len(a) == 0:
        return a
    min = a.min()
    max = a.max()
    if max == min:
        return a
    return np.uint16(((np.float64(a) - min) / (max-min)) **(1/g) * (max-min) + min)

def _idx_blk(p, b):
    q = p + b   # if p is np.array, b can be a number or an array
    return (slice(int(p[k]), int(q[k]))
            for k in range(len(p)))
    #return (slice(int(p[0]), int(q[0])),
    #        slice(int(p[1]), int(q[1])),
    #        slice(int(p[2]), int(q[2])))

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

def SliceZarrImage(zarr_image, blk_sz, p_center, vec_normal, vec_up):
    blk_sz = 128
    p0 = p_center - blk_sz/2
    img3d = zarr_image[*_idx_blk(p0, blk_sz)]
    #print('idx =', tuple(_idx_blk(p_center - blk_sz/2, blk_sz)))
    #print('maxmin =', np.max(img3d), np.min(img3d))
    #print('p_center =', p_center)
    #print('vec_normal =', vec_normal)
    #print('vec_up =', vec_up)
    #tifffile.imwrite('a.tif', img3d.T)
    p_center = p_center - p0
    simg = NormalSlice3DImage(img3d, p_center, vec_normal, vec_up)
    return simg

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

def WalkTreeTangent(swc_path, image_block_path, node_idx):
    """
    Plot the osculating plane of the swc at node_idx.
    """
    
    ## prepare fiber position

    # get an ordered and continuous node index tree and its graph
    ntree = LoadSWCTree(swc_path)
    processes = SplitSWCTree(ntree)
    ntree, processes = SWCDFSSort(ntree, processes)
    tr_idx = SWCNodeRelabel(ntree)
    ntree = (tr_idx, ntree[1])
    ngraph = GetUndirectedGraph(ntree)

    p_focused = ntree[1][node_idx, :3]
    p_img_center = p_focused

    ## prepare image
    desired_block_size = (128, 128, 128)

    # load image around p_img_center
    imgz = zarr.open(image_block_path, mode='r')

    p_img_corner = p_img_center - _a(desired_block_size) / 2

    idx_rg = [slice(int(p_img_corner[i]),
                    int(p_img_corner[i] + desired_block_size[i]))
              for i in range(3)]

    img3d = imgz[*idx_rg]

    print("p_img_center", p_img_center)
    print("p_img_corner", p_img_corner)
    print("diff", p_img_center - p_img_corner)

    #ShowThreeViews(img3d, p_img_center_s)
    ShowThreeViewsMIP(img3d)

    # load lychnis block for comparison
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

    ## try to show a (primary) tangent plane

    # show node position and neighbor indices
    print(ntree[0][node_idx, :], ntree[1][node_idx, :3])
    print(ngraph[node_idx].indices)

    neig_node_idx = ngraph[node_idx].indices
    # use SVD to get axis
    # neighbor point positions
    neig_pos = ntree[1][_ai(list(neig_node_idx)+[node_idx]), :3]
    print(neig_pos)
    u, s, vt = np.linalg.svd(neig_pos - neig_pos.mean(axis=0), full_matrices=True)

    # substract every coor by p_img_corner to align to the image
    p_img_center_s = p_img_center - p_img_corner

    img_tangent = NormalSlice3DImage(img3d, p_img_center_s, vt[2], vt[1])

    figure(20)
    imgshow(img_tangent.T)

class SmoothCurve:
    def __init__(self, rp, spl_smooth = 0):
        """
        Get a (approximate) natural parametrization of the curve passing through points `rp`.
        rp is in the form [[x1,y1,z1], [x2,y2,z2], ...]
        spl_smooth see reference below, 0 means interpolation, None to use default smoothing. 
        Ref. Smoothing splines - https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
        """
        piece_len = norm(diff(rp, axis=0), axis=1)
        if np.any(piece_len == 0):
            print('Warning: repeated point(s), ignoring.')
            # leave only the non-repeated points
            rp = rp[_ha(True, piece_len>0)]
            piece_len = norm(diff(rp, axis=0), axis=1)
        if len(rp) <=3:
            if len(rp) < 2:
                raise ValueError('The number of points should be at least 2.')
            if len(rp) == 2:
                # add two extra point to ease the interpolation
                rp = _va([rp[0], 2/3*rp[0] + 1/3*rp[1], 1/3*rp[0] + 2/3*rp[1], rp[1]])
            if len(rp) == 3:
                rp0 = rp
                rp = interpolate.interp1d([0, 1, 2], rp0.T)(np.linspace(0, 2, 4)).T
            piece_len = norm(diff(rp, axis=0), axis=1)
        # Use cord length parametrization to approximate the natural parametrization
        tck, u = interpolate.splprep(rp.T, u = _ha(0, piece_len.cumsum()), s = spl_smooth)
        self.tck = tck
        self.u = u

    def length(self):
        """
        Curve geometric length
        """
        return self.u[-1]

    def __call__(self, t, der = 0):
        """
        Return point coordinate at parameter t.
        Optionnally return the curve derivative of order `der`.
        """
        t = _a(t)
        a = _a(interpolate.splev(t, self.tck, der = der))
        # transpose so that the last dim is the point coor
        a = a.transpose(list(range(1,len(a.shape))) + [0])
        return a

    def PointTangent(self, t):
        """
        Return point coordinate and tangent vector at parameter t.
        t can be a scalar or a numpy array. In the case of array, the last dim is the point coor and tangent.
        """
        p = self(t)
        # get tanget vector
        dp = self(t, der=1)  # approximately we have ||dp|| = 1
        l2 = norm(dp, axis=-1)
        if np.any(l2):
            dp = dp / l2.reshape(list(l2.shape)+[1])   # reshape to help auto broadcasting
        return p, dp

    def PointTangentNormal(self, t):
        """
        In addition to PointTangent, return the (unormalized) normal vector, its length is the curvature.
        """
        p = self(t)
        # get tanget vector
        dp = self(t, der=1)  # approximately we have ||dp|| = 1
        dtds = 1 / norm(dp)   # = dt / ds
        dtds = dtds.reshape(list(dtds.shape)+[1])
        dp = dp * dtds
        # get normal vector
        ddp = self(t, der=2)
        ddp = ddp * dtds**2
        ddp = ddp - np.dot(dp, ddp) * dp
        # the ddp is not a normalized vector, used to determine the stength of curving.
        return p, dp, ddp

    def FrenetFrame(self, t):
        """
        Return the point coordinate and the Frenet frame at parameter t.
        """
        p, dp, ddp = self.PointTangentNormal(t)
        ddp = ddp / norm(ddp)
        return p, (dp, ddp, np.cross(dp, ddp))

def WalkProcessCircularMIP(process_pos, image_block_path, interp_resolution = 2):
    blk_sz = 128

    radius_max_soft = blk_sz / 2
    radius_step = interp_resolution
    n_radius = int(radius_max_soft / radius_step) + 1

    zimg = zarr.open(image_block_path, mode='r')

    try:
        # interpolation the process by a smooth curve
        curve = SmoothCurve(process_pos, spl_smooth=None)
    except ValueError:
        print('Warning: (looks like) the process too short! ignoring.')
        print('Processes coordinate:\n', process_pos)
        extent = [0, len(process_pos), 0, n_radius*radius_step]
        return np.zeros((len(process_pos), n_radius), dtype=zimg.dtype), extent

    # parameters for getting points to be inspected
    t_step = interp_resolution
    n_t = int(curve.length() / t_step) + 1
    t_interp = np.linspace(0, curve.length(), n_t)

    if 0:
        fig = figure(200)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(process_pos[:,0], process_pos[:,1], process_pos[:,2])
        intpoints = curve(t_interp)
        ax.plot3D(intpoints[:,0], intpoints[:,1], intpoints[:,2])
        ax.axis('equal')

        p, frame = curve.FrenetFrame(curve.length())
        print(p)
        print(frame)
        alen = 20
        ax.quiver(p[0], p[1], p[2], frame[0][0], frame[0][1], frame[0][2], length=alen, color='r')
        ax.quiver(p[0], p[1], p[2], frame[1][0], frame[1][1], frame[1][2], length=alen, color='g')
        ax.quiver(p[0], p[1], p[2], frame[2][0], frame[2][1], frame[2][2], length=alen, color='b')

    axon_circular_mip = np.zeros((n_t, n_radius), dtype=zimg.dtype)
    for idx_c_k in range(n_t):
        p, dp, ddp = curve.PointTangentNormal(t_interp[idx_c_k])
        #print('p =', p)

        if 0:
            p, frame = curve.FrenetFrame(t_interp[idx_c_k])
            # get the normal-plane image
            timg = SliceZarrImage(zimg, blk_sz, p, frame[2], frame[1])
            figure(201)
            plt.cla()
            imgshow(timg.T)

        normal_img = SliceZarrImage(zimg, blk_sz, p, dp, ddp)
        
        if 0:
            figure(202)
            plt.cla()
            imgshow(normal_img.T)

        # construct circle sample grid
        r_pixel_max = np.zeros(n_radius, dtype=zimg.dtype)
        for j in range(n_radius):
            r = radius_step * j
            n_deg = int(2 * np.pi * r / radius_step) + 1
            s_deg = 2*pi/n_deg * np.arange(n_deg)
            p_sample = r * _a([cos(s_deg), sin(s_deg)]).T + blk_sz / 2
            #plt.scatter(p_sample[:,0], p_sample[:,1])

            # interpolation in Image:
            # See also:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            # https://discourse.itk.org/t/how-can-i-get-an-interpolated-value-between-voxels/1535/6
            vals = scipy.ndimage.map_coordinates(normal_img, p_sample.T, order=3)
            r_pixel_max[j] = np.max(vals)
        
        axon_circular_mip[idx_c_k, :] = r_pixel_max
    
    extent = [0, curve.length(), 0, n_radius*radius_step]
    return axon_circular_mip, extent
    
def LoadSWCTreeProcess(swc_path, sort_processes = True):
    ntree = LoadSWCTree(swc_path)
    processes = SplitSWCTree(ntree)
    if sort_processes:
        ntree, processes = SWCDFSSort(ntree, processes)
    #tr_idx = SWCNodeRelabel(ntree)
    #ntree = (tr_idx, ntree[1])
    #ngraph = GetUndirectedGraph(ntree)
    return ntree, processes

class NTreeOps:
    def __init__(self, swc_path, sort_proc = True):
        self.ntree, self.processes = LoadSWCTreeProcess(swc_path, sort_proc)
        # get a tree with consitive node index
        self.tr_idx, self.map_id_idx = SWCNodeRelabel(self.ntree, output_map=True)
        self.map_idx_id = self.ntree[0][:,0]
        self.ntree_cons = (self.tr_idx, self.ntree[1])
        # get the graph of the tree
        self.ngraph = GetUndirectedGraph(self.ntree)
        self.build_tree_depth()
    
    def build_tree_depth(self):
        # find roots
        root_idx = np.flatnonzero(self.tr_idx[:, 1] == -1)  # assume relabel is perfect
        # find branch/leaf points
        n_id = self.tr_idx.shape[0]      # number of nodes
        n_child,_ = np.histogram(self.tr_idx[:, 1],
                        bins = np.arange(-1, n_id + 1, dtype=dtype_id))
        assert(n_child[0] == len(root_idx))
        # leave out the node '-1'
        n_child = np.array(n_child[1:], dtype=dtype_id)
        # n_child == 0: leaf
        # n_child == 1: middle of a path or root(?)
        # n_child >= 2: branch point or root(?)
        node_depth = np.zeros((n_id, 2), dtype=dtype_id)
        node_depth[root_idx] = 0
        node_root_path_length = np.zeros(n_id, dtype=dtype_coor)
        root_n_child = n_child[root_idx]    # we still need this
        n_child[root_idx] = -1              # let's label the root
        for k in range(n_id):
            if n_child[k] == -1:
                continue
            # add depth on top of parent
            p_idx = self.tr_idx[k, 1]
            node_depth[k, 0] = node_depth[p_idx, 0] + 1
            node_depth[k, 1] = node_depth[p_idx, 1] + (n_child[p_idx] != 1)
            node_root_path_length[k] += node_root_path_length[p_idx] + \
                norm(self.ntree[1][k, :3] - self.ntree[1][p_idx, :3])

        n_child[root_idx] = root_n_child
        self.root_idx = root_idx
        self.node_depthes = node_depth
        self.v_node_depth = node_depth[:,0]
        self.v_branch_depth = node_depth[:,1]
        self.node_root_path_length = node_root_path_length
        self.n_child = n_child

    def branch_depth(self, node_id):
        """
        Return the branch depth of the node (ID).
        Demo tree (0 is the root):
        0 -- 1 -- 2 -- 3
              \
               4
        Node ID : Branch depth
          0     :   0
          1     :   1
          2     :   2
          3     :   2
          4     :   2
        """
        # TODO: check id validity
        if isinstance(node_id, int) or isinstance(node_id[0], int):
            return self.v_branch_depth[self.map_id_idx[node_id]]
        else:
            # assume list or array, i.e. node_id is processes
            # Note: in this case, processes is idx instead of id
            processes = node_id
            node_id = self.map_idx_id[_ai([p[-1] for p in processes])]
            return self.v_branch_depth[self.map_id_idx[node_id]]

    def end_point(self, processes):
        """
        Return the end point of the processes.
        """
        return self.map_idx_id[_ai([p[-1] for p in processes])]

    def path_length_to_root(self, node_id):
        # TODO: check id validity
        return self.node_root_path_length[self.map_id_idx[node_id]]

def test_ntreeops():
    swc_path = os.path.join(pkg_path_neu3dviewer, 'tests/ref_data/swc_ext/t3.3.swc')
    ntrop = NTreeOps(swc_path, True)
    # test reading a tree
    assert(len(ntrop.ntree[0]) == 16)
    assert(len(ntrop.ntree[1]) == 16)
    assert(len(ntrop.processes) == 10)
    # test map_id_idx
    n_id  = 6
    n_idx = ntrop.map_id_idx[n_id]
    assert(n_idx == 5)
    np_idx = ntrop.ntree_cons[0][n_idx, 1]
    assert(ntrop.ntree[0][np_idx, 0] == 7)
    assert(norm(ntrop.ntree[1][np_idx,:3] - _a([2,0,0])) == 0)
    # test ngraph
    true_pairs = [(7, 6), (6, 7), (8, 0), (12, 13)]
    false_pairs = [(1,1), (13, 14), (1, 3)]
    for pr in true_pairs:
        idx_pr = ntrop.map_id_idx[_ai(pr)]
        assert(ntrop.ngraph[*idx_pr] == True)
    for pr in false_pairs:
        idx_pr = ntrop.map_id_idx[_ai(pr)]
        assert(ntrop.ngraph[*idx_pr] == False)
    # test depth
    ans = np.array([
       [11,  0,  0],
       [ 1,  0,  0],
       [ 4,  1,  1],
       [ 3,  2,  2],
       [ 7,  2,  2],
       [ 6,  3,  3],
       [ 9,  3,  3],
       [ 2,  4,  3],
       [12,  0,  0],
       [13,  1,  1],
       [17,  0,  0],
       [18,  1,  1],
       [ 0,  0,  0],
       [ 8,  1,  1],
       [10,  2,  2],
       [19,  2,  2]
    ])
    for i, d1, d2 in ans:
        #print(i, d1, d2)
        assert(norm(ntrop.node_depthes[ntrop.map_id_idx[i],:] - _a([d1, d2])) == 0)
    assert(ntrop.node_root_path_length[ntrop.map_id_idx[2]] - (np.sqrt(1 + 0.01)*2 + 2) < 1e-6)
    # test branch_depth
    assert(ntrop.branch_depth(2) == 3)
    assert(ntrop.branch_depth([2]) == 3)
    pc1 = ntrop.map_id_idx[_ai([7,9,2])]
    pc2 = ntrop.map_id_idx[_ai([12,13])]
    assert(np.all(ntrop.branch_depth([pc1, pc2]) == [3, 1]))
    # test end_point
    assert(np.all(ntrop.end_point([pc1, pc2]) == [2, 13]))

def exec_filter_string(filter_str, ntrop):
    if (filter_str is None) or (filter_str == ''):
        return np.ones(len(ntrop.processes), dtype=bool)
    local_vars = {
        'processes': ntrop.processes,
        'branch_depth': ntrop.branch_depth,
        'path_length_to_root': ntrop.path_length_to_root,
        'end_point': ntrop.end_point,
        # additional
        'ntrop': ntrop,
        'swc_path': swc_path,
        'np' : np,
    }
    vec_filted = eval(filter_str, {}, local_vars)
    return vec_filted

def test_tree_filter(swc_path):
    ntrop = NTreeOps(swc_path)
    #print(len(ntrop.processes))
    dep = ntrop.branch_depth(ntrop.processes)
    #print(dep)
    dep_ref = SimplifyTreeWithDepth(ntrop.processes)[:,2]
    assert(np.all(dep == dep_ref[1:]))
    vec_filted_ref = (dep <= 3) & \
                     (ntrop.path_length_to_root(ntrop.end_point(ntrop.processes)) > 10000)
    #print(vec_filted_ref)
    filter_str = '(branch_depth(processes)<=3) & (path_length_to_root(end_point(processes))>10000)'
    vec_filted = exec_filter_string(filter_str, ntrop)
    #print(vec_filted)
    #print(np.array2string(_ha(vec_filted[:,None], vec_filted_ref[:,None]), threshold=3000))
    assert(np.all(vec_filted == vec_filted_ref))

class FileLogger:
    """
    Format:
    [
        {
            'clicked_pos_um': cmip_pos,
            'id_proc': id_proc,
            'cmip_local_pos': local_pos,
            'interpolated_pos': r,
            'nearest_node_id': node_id,
            'nearest_node_pos': proc_coor[idx_min],
            't_str': t_str,
        },
        ...
    ]
    """
    def __init__(self, file_path):
        self.file_path = file_path
        if os.path.isfile(file_path):
            # read old logs as json
            with open(file_path, 'r', encoding='utf-8') as fin:
                self.logs = json.load(fin)
        else:
            self.logs = []
    
    def __len__(self):
        return len(self.logs)
    
    def clicked_pos_um(self):
        return [log['clicked_pos_um'] for log in self.logs]
    
    def Append(self, item):
        self.logs.append(item)
        with open(self.file_path, 'w', encoding='utf-8') as fout:
            json.dump(self.logs, fout, indent=4)
    
    def Pop(self):
        self.logs.pop()
        with open(self.file_path, 'w', encoding='utf-8') as fout:
            json.dump(self.logs, fout, indent=4)

def WalkTreeCircularMIP(swc_path, image_block_path, cmip_dir, resolution):
    # get an ordered and continuous node index tree and its graph
    print("WalkTreeCircularMIP")
    ntree, processes = LoadSWCTreeProcess(swc_path)

    swc_name = os.path.basename(swc_path).split('.')[0]
    print('SWC name:', swc_name)
    print('Number of processes:', len(processes))
    print('Number of nodes:', len(ntree[0]))

    for idx_processes in range(len(processes)):
        print(f'process {idx_processes}, node length {len(processes[idx_processes])}')

        proc_coor = ntree[1][processes[idx_processes],:3]
        axon_circular_mip, extent = WalkProcessCircularMIP(proc_coor, image_block_path, resolution)

        if 0:
            figure(205)
            plt.cla()
            imgshow(axon_circular_mip.T, extent=extent)
            xlabel('neurite position (um)')
            ylabel('distance to neurite (um)')
            title('circular MIP')
        
        img_out_name = f'{cmip_dir}/{swc_name}_cmip_proc{idx_processes}.tif'
        tifffile.imwrite(img_out_name, axon_circular_mip.T)

class TreeCircularMIPViewer:
    def __init__(self, swc_path, image_block_path, pic_path, res, view_len_str, filter_str = None):
        self.swc_path = swc_path
        self.image_block_path = image_block_path
        self.pic_path = pic_path
        self.view_length = int(view_len_str.split('/')[0])   # 1000
        self.view_n_rows = int(view_len_str.split('/')[1])   # 4
        # extract neuron id, e.g. 255 in 'neuron#255.lyp.swc'
        self.neu_id = int(os.path.basename(swc_path).split('neuron#')[1].split('.')[0])
        print(f'neuron id: {self.neu_id}')

        print('Loading swc...', end='')
        #self.ntree, self.processes = LoadSWCTreeProcess(self.swc_path)
        self.ntrop = NTreeOps(self.swc_path)
        self.ntree = self.ntrop.ntree
        self.processes = self.ntrop.processes
        print('done.\nNumber of processes:', len(self.processes))

        # list and sort pic path
        self.tif_pathes = glob.glob(os.path.join(
            self.pic_path, f'neuron#{self.neu_id}_[rc]mip_*.tif'))
        # sort, so that follow the order in the swc file.
        get_proc_id = lambda s: int(s.split('proc')[1].split('.')[0])
        self.tif_pathes = sorted(self.tif_pathes, key=get_proc_id)
        # process indexes
        self.proc_ids = [get_proc_id(os.path.basename(s)) for s in self.tif_pathes]
        
        if len(self.tif_pathes) == 0:
            raise ValueError(f'No cMIP image found. Search path is "{self.pic_path}".')
        #print('\n'.join(self.tif_pathes))
        #print(self.proc_ids)

        if len(self.processes) != len(self.tif_pathes):
            print('==========================================================')
            print('Warning: number of processes != number of cMIP')
            print('Indicating cMIPs are not complete or not the same version.')
            print('The recorded click could be wrong. Proceed anyway.')
            print('----------------------------------------------------------')
        print(f'Filtering according to "{filter_str}" ...', end='')
        vec_filted = exec_filter_string(filter_str, self.ntrop)
        n_proc = len(self.tif_pathes)
        self.tif_pathes = [self.tif_pathes[i] for i in range(n_proc) if vec_filted[self.proc_ids[i]]]
        self.proc_ids   = [self.proc_ids[i]   for i in range(n_proc) if vec_filted[self.proc_ids[i]]]
        print(f'done. {len(self.tif_pathes)} processes left.')

        self.cmip_res = res
        self.gap_size = 3

        # load images and construct index (lookup table)
        print('Loading Circular MIP images...', end='')
        self.proc_img_s = [tifffile.imread(s).T for s in self.tif_pathes]
        self.row_idxs = np.cumsum(_ha(0, _ai(
            [i.shape[0] + self.gap_size for i in self.proc_img_s])))
        if not self.proc_img_s:
            self.img_height = 1
        else:
            self.img_height = self.proc_img_s[0].shape[1]
        print('done.', '\nNumber of loaded images:', len(self.proc_img_s))

        # default "brightness"
        self.screen_img_gamma = 3.0

        #t_now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #f_pos_path = f'neuron#{self.neu_id}_cmip_marks_{t_now_str}.json'
        f_pos_path = f'neuron#{self.neu_id}_cmip_marks.json'
        self.logger = FileLogger(f_pos_path)
    
    def cmip_pos_to_coordinate(self, cmip_pos):
        # get position in terms of process id (id_proc) and path distance to starting point (local_pos)
        idx_proc = np.searchsorted(self.row_idxs, cmip_pos, 'right') - 1
        if idx_proc == len(self.proc_ids):
            idx_proc = len(self.proc_ids) - 1
        id_proc = self.proc_ids[idx_proc]
        local_pos = cmip_pos - self.row_idxs[idx_proc]
        if local_pos > self.proc_img_s[idx_proc].shape[0]:
            print('Warning: Clicked in the gap.')
            local_pos = self.proc_img_s[idx_proc].shape[0] - 1
        local_pos = local_pos * self.cmip_res
        # construct the process curve, must be the same as in WalkProcessCircularMIP()
        proc_coor = self.ntree[1][self.processes[id_proc],:3]
        curve = SmoothCurve(proc_coor, spl_smooth=None)
        # output position info
        print(f'id_proc: {id_proc}, local_pos: {local_pos:.1f} um')
        r = curve(local_pos)
        print(f'interpolated position: [{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}]')
        idx_min = np.argmin(norm(proc_coor - r, axis=1))
        node_id = self.ntree[0][self.processes[id_proc][idx_min],0]
        print(f'nearest tree node id: {node_id} xyz: {proc_coor[idx_min]}')

        info = {
            'id_proc': id_proc,
            'cmip_local_pos': local_pos,
            'interpolated_pos': r.tolist(),
            'nearest_node_id': int(node_id),
            'nearest_node_pos': proc_coor[idx_min].tolist(),
        }
        return info
    
    def LocalCurveToNTree(self, id_proc, local_cmip_pos, length_range):
        proc_coor = self.ntree[1][self.processes[id_proc],:3]
        curve = SmoothCurve(proc_coor, spl_smooth=None)
        bg_pos = 0 if local_cmip_pos < length_range else local_cmip_pos - length_range
        ed_pos = curve.length() if curve.length() - local_cmip_pos < length_range else local_cmip_pos + length_range
        s_idx = _ai(range(int(bg_pos/self.cmip_res), int(ed_pos/self.cmip_res)))
        coor = curve(s_idx * self.cmip_res)
        nt =(
                _ai([(s_idx[k], s_idx[k-1] if k>=1 else -1, 3) for k in range(len(s_idx))]),
                _a ([(coor[k,0], coor[k,1], coor[k,2], 1.0)  for k in range(len(s_idx))])
            )
        return nt
    
    def ConstructCMIP(self, pos0):
        # pos0 is in pixel unit
        screen_size = self.view_length
        n_screen_rows = self.view_n_rows
        proc_img_s = self.proc_img_s
        row_idxs = self.row_idxs
        gap_size = self.gap_size
        img_height = self.img_height
        res = self.cmip_res
        row_size = screen_size/n_screen_rows

        screen_img = np.zeros((screen_size, img_height), dtype=np.uint16)
        id_img = np.searchsorted(row_idxs, pos0, 'right') - 1
        screen_img_pos = 0
        i_bg = pos0 - row_idxs[id_img]
        # fill screen image
        while id_img < len(proc_img_s) and screen_img_pos < screen_size:
            i_len = proc_img_s[id_img].shape[0]
            if screen_size - screen_img_pos < i_len - i_bg:
                i_ed = screen_size - screen_img_pos + i_bg
            else:
                i_ed = i_len
            #print('id_img', id_img, 'i_len', i_len, 'i_bg', i_bg, 'i_ed', i_ed)
            img = proc_img_s[id_img][i_bg:i_ed, :]
            screen_img[screen_img_pos:screen_img_pos + max(i_ed - i_bg, 0), :] = img
            screen_img_pos += i_ed - i_bg + gap_size
            id_img += 1
            i_bg = 0

        clicked_pos = _a(self.logger.clicked_pos_um()) / res
        b_click_in_local = (clicked_pos > pos0) & (clicked_pos < pos0 + screen_size)
        local_clicked_pos = clicked_pos[b_click_in_local]

        figure(301).clear()
        fig, axs = plt.subplots(n_screen_rows, num=301)
        fig.suptitle(f'Neuron#{self.neu_id} circular MIP\n pos {pos0} - {pos0+screen_size} (of {row_idxs[0]} - {row_idxs[-1]})')
        # show the screen_img in split figure rows
        axshow = lambda axs, k, n, im, **kwval: \
            axs[k].imshow(f_l_gamma( \
                    im[int(screen_size/n*k) : int(screen_size/n*(k+1)), :].T,
                    self.screen_img_gamma),
                cmap='gray', origin='lower', **kwval)
        for id_s in range(n_screen_rows):
            extent = [pos0 + id_s*row_size, pos0 + (id_s+1)*row_size,
                      0, img_height]
            extent = res * _a(extent)
            axshow(axs, id_s, n_screen_rows, screen_img, extent=extent)
            ck_idx = (pos0 + row_size*id_s <= local_clicked_pos) & \
                     (local_clicked_pos < pos0 + row_size*(id_s+1))
            ck_pos = local_clicked_pos[ck_idx]
            axs[id_s].plot(res*ck_pos, res*img_height/2 * np.ones(len(ck_pos)), 'r+')
            if id_s == n_screen_rows - 1:
                axs[id_s].set_xlabel('um')
                axs[id_s].set_ylabel('um')
        fig.canvas.mpl_connect('key_press_event', self.on_cmip_key)
        fig.canvas.mpl_connect('button_press_event', self.on_cmip_mouse)
        fig.canvas.mpl_connect('scroll_event', self.on_cmip_scroll)

        # bad design
        self.fig = fig
        self.axs = axs
        self.last_pos0 = pos0
        self.screen_size = screen_size
        self.n_screen_row = n_screen_rows

    def on_cmip_key(self, event):
        print('key pressed:', event.key)
        if event.key == 'pagedown':
            self.ConstructCMIP(self.last_pos0 + self.screen_size)
            #plt.show()
            self.fig.canvas.draw()
        elif event.key == 'pageup':
            if self.last_pos0 >= int(self.screen_size/2):
                self.ConstructCMIP(self.last_pos0 - self.screen_size)
            #plt.show()
            self.fig.canvas.draw()
        if event.key == ' ':
            self.ConstructCMIP(self.last_pos0 + int(self.screen_size/2))
            #plt.show()
            self.fig.canvas.draw()
        if event.key == 'home':
            self.ConstructCMIP(0)
            #plt.show()
            self.fig.canvas.draw()
        if event.key == 'end':
            self.ConstructCMIP(self.row_idxs[-1] - self.screen_size)
            #plt.show()
            self.fig.canvas.draw()
        elif event.key == '*':
            self.screen_img_gamma += 0.5
            self.ConstructCMIP(self.last_pos0)
            #plt.show()
            self.fig.canvas.draw()
        elif event.key == '/':
            self.screen_img_gamma = max(self.screen_img_gamma - 0.5, 0.5)
            self.ConstructCMIP(self.last_pos0)
            #plt.show()
            self.fig.canvas.draw()
        elif event.key == 'z':
            # revoke last clicked position
            if len(self.logger) > 0:
                self.logger.Pop()
                self.ConstructCMIP(self.last_pos0)
                self.fig.canvas.draw()
                print(f'(revoked, total {len(self.logger)})')

    def on_cmip_scroll(self, event):
        #print(event.button, event.step)
        step = int(self.screen_size / self.n_screen_row)
        if event.button == 'up':
            if self.last_pos0 > step/2:
                self.ConstructCMIP(self.last_pos0 - step)
        else:
            self.ConstructCMIP(self.last_pos0 + step)
        self.fig.canvas.draw()

    def on_cmip_mouse(self, event):
        """
        Left key: view in  3D
        Right key: record
        Middle key: show info in cmd
        """
        if not ((event.button == 1 or event.button == 2 or event.button == 3) and event.inaxes):
            return
        print('=== Clicked ===')
        cmip_pos = event.xdata / self.cmip_res
        #print(f' pos: {event.xdata}, {event.ydata}; screen pos {event.x}, {event.y}')
        #print(' ax id', id_ax)
        print(f'cmip_pos: {cmip_pos:.1f} pixel')
        info = self.cmip_pos_to_coordinate(cmip_pos)
        if event.button == 3:
            # add to log
            t_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            self.logger.Append({
                    'clicked_pos_um':cmip_pos * self.cmip_res,
                    't_str':t_str
                } | info)
            self.ConstructCMIP(self.last_pos0)
            self.fig.canvas.draw()
            print(f'(recorded, total {len(self.logger)})')
        if event.button == 1:          # event.key == 'v':
            self.fig.canvas.flush_events()
            print('Opening Neu3DViewer...')
            nt = {
                f'neuron#{self.neu_id}':self.ntree,
                'smoothed': self.LocalCurveToNTree(
                    info['id_proc'],
                    info['cmip_local_pos'],
                    self.cmip_res * self.img_height * 0.6),
            }
            ViewByNeu3DViewer(nt, self.image_block_path, info['interpolated_pos'])

def SaveSWC(fout_path, ntree, comments=''):
    with open(fout_path, 'w', encoding="utf-8") as fout:
        if len(comments) > 0:
            fout.write(comments)
            fout.write('\n')
        for j in range(len(ntree[0])):
            nid      = ntree[0][j,0]          # node id
            pa       = ntree[0][j,1]          # parent
            ty       = 2 if pa!=-1 else 0     # type
            x,y,z,di = ntree[1][j,:]          # x,y,z, diameter
            fout.write('%d %d %.1f %.1f %.1f %.3f %d\n' % \
                       (nid, ty, x, y, z, di, pa))

def ViewByNeu3DViewer(named_ntree, zarr_dir, r_center):
    """
    Open Neu3DViewer to view the swc tree and image block.
    named_ntree: named ntree in dict
    """
    tmp_dir = '.tmp/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    # remove all swc before hand
    for f in glob.glob(os.path.join(tmp_dir, '*.swc')):
        os.remove(f)
    # save neuron trees to swc files
    for name, ntree in named_ntree.items():
        save_name = os.path.join(tmp_dir, name + '.swc')
        SaveSWC(save_name, ntree)
    
    blk_sz = 128
    r_center = _a(r_center)
    r0 = list(map(int, r_center - blk_sz/2))
    r1 = list(map(int, r_center + blk_sz/2))
    look_distance = blk_sz*3

    # see help of Neu3DViewer for possible options
    cmd_obj_desc = {
        'swc_dir' : tmp_dir,
        'img_path': zarr_dir,
        'range'   : f'[{r0[0]}:{r1[0]}, {r0[1]}:{r1[1]}, {r0[2]}:{r1[2]}]',
        'origin'  : str(list(r0)),
        #'look_at' : str(list(r_center)),
        #'look_distance': look_distance,
    }
    # call Neu3DViewer
    neu3dviewer.utils.debug_level = 2
    gui = GUIControl()
    gui.EasyObjectImporter(cmd_obj_desc)
    gui.Set3DCursor(r_center)
    fn1 = lambda gui: gui.interactor.style.ui_action. \
                        scene_look_at(r_center, look_distance)
    fn2 = lambda gui: gui.interactor.style.ui_action. \
                        auto_brightness('')
    def fn3(gui):
        #gui.scene_objects['swc.2'].color = 'blue'
        swc_obj_dict = gui.GetObjectsByType('swc')
        swcs = ArrayfyList(list(swc_obj_dict.values()))
        if 'smoothed' in swcs.obj_dict:
            swcs['smoothed'].color = 'blue'
        #gui.render_window.Render()
        gui.LazyRender()

    gui.Start([fn1, fn2, fn3])

def get_program_options():
    parser = argparse.ArgumentParser(
        description="Walk a neuron tree, generate and show its circular maximum intencity projection(MIP)."
    )
    parser.add_argument('swc_file_path', nargs='*')  # nargs='*' or '+'
    parser.add_argument('--zarr_dir',
                        help='Path to the zarr directory.')
    parser.add_argument('--cmip_dir',
                        help='Path to the circular MIP directory, for write or read (view mode).')
    parser.add_argument('--view', action='store_true',
                        default=argparse.SUPPRESS,  # for later filled by config
                        help='Enable view mode.')
    parser.add_argument('--res', type=float,
                        help='resolution, e.g. "1.0".')
    parser.add_argument('--view_length',
                        help='view length in screen, total length / rows, default "1000/4".')
    parser.add_argument('--filter',
                        help="""
                        string for filtering the processes.
                        Example: '(branch_depth(processes)<=3) & (path_length_to_root(end_point(processes))>10000)' 
                        """)
    parser.add_argument('--test',
                        help='Test mode, not for general use.')
    parser.add_argument('--config_path',
                        help='Path to the configuration file in json format.'
                        'supply the same arguments as commandline options.'
                        'Commandline options have higher priority.')
    parser.add_argument('--verbose', action='store_true',
                        default=argparse.SUPPRESS,  # for later filled by config
                        help='Show more information.')
    args = parser.parse_args()
    
    if getattr(args, 'verbose', False):
        print('From command-line.')
        print(args)

    # read config from file, if any
    if args.config_path:
        # read from config if not presented in commandline.
        opt = json.load(open(args.config_path, 'r', encoding='utf-8'))
        for k, v in opt.items():
            if (k in args) and (getattr(args, k) is None):
                vars(args)[k] = v
        # special rule for swc path
        if (args.swc_file_path == []) and ('swc_file_path' in opt):
            args.swc_file_path = opt['swc_file_path']
        # special rule for view
        k_switches = ['view', 'verbose']
        for k in k_switches:
            if (not hasattr(args, k)) and (k in opt):
                vars(args)[k] = opt[k]

    # set default values
    default_opt = {
        'verbose': False,
        'view': False,
        'view_length': "1000/4",
    }
    for k, v in default_opt.items():
        if (not hasattr(args, k)) or (getattr(args, k) is None):
            vars(args)[k] = v

    if args.view is None:
        raise ValueError('Unknown view mode. Specify --view true or --view false')

    if args.verbose:
        print("Program arguments:")
        print(args)

    return args

def check_cmd_options(args, *opt_names):
    ok = True
    for opt_name in opt_names:
        opt = opt_name.lstrip('-')
        if (not hasattr(args, opt)) or (getattr(args, opt) is None) or \
            (isinstance(getattr(args, opt), list) and (len(getattr(args, opt)) == 0)):
            print(f'Option "{opt_name}" is not specified.')
            ok = False
    if not ok:
        raise ValueError(f'Lack of options. See `python {os.path.basename(__file__)} -h` for help.')

if __name__ == '__main__':
    args = get_program_options()

    img_block_path = args.zarr_dir
    s_swc_path = args.swc_file_path

    if args.test:
        #swc_path = 'neuron#255.lyp.swc'
        # node_idx = 1936, node_id = 932514
        # xyz: [52785.  28145.6 55668.9]
        # Branch depth: 1
        # Node depth: 1254
        # Path length to root: 23228.5
        if len(s_swc_path) == 0:
            swc_path = 'neuron#122.lyp.swc'
        else:
            swc_path = s_swc_path[0]
        print(f'Testing on SWC "{swc_path}"')

        if args.test == 'slicing':
            plt.ion()
            Test3dImageSlicing()
            plt.show()
        elif args.test == 'tangent':
            #IPython %run ./neu_walk.py 'neuron#255.lyp.swc' --test tangent
            node_idx = 1936
            plt.ion()
            WalkTreeTangent(swc_path, img_block_path, node_idx)
            plt.show()
        elif args.test == 'neu3dviewer':
            ntree = LoadSWCTree(swc_path)
            #swc_path = 'neuron#255.lyp.swc'
            #r_c = _a([52785., 28145.6, 55668.9])
            r_c = _a([46090.5, 35027.2, 37534.6])
            ViewByNeu3DViewer({'abc': ntree}, img_block_path, r_c)
        elif args.test == 'ntreeops':
            test_ntreeops()
        elif args.test == 'tree_filter':
            test_tree_filter(swc_path)
        else:
            raise ValueError('Unknown test mode.')
    elif args.view:
        check_cmd_options(args, 'swc_file_path', '--cmip_dir', '--zarr_dir', '--res')
        for swc_path in s_swc_path:
            cmip_viewer = TreeCircularMIPViewer(
                swc_path, img_block_path, args.cmip_dir,
                args.res, args.view_length, args.filter)
            cmip_viewer.ConstructCMIP(0)
            plt.show()
    else:
        check_cmd_options(args, 'swc_file_path', '--cmip_dir', '--zarr_dir', '--res')
        for swc_path in s_swc_path:
            WalkTreeCircularMIP(swc_path, img_block_path,
                                args.cmip_dir, args.res)
