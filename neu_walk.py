#!/usr/bin/env python3

## Before run:
# * Prepare a directory named "pic_tmp" to hold output cMIP images
# * Put external dependencies in directory "external", such as external/neu3dviewer
# * See requirements.txt for required python packages.
# * need python3.11

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
#   See https://matplotlib.org/stable/gallery/widgets/menu.html
# * Show smoothed curve in 3D view
# * Fix neu3dviewer parallel load bug in windows

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
pkg_path_neu3dviewer = 'external/neu3dviewer'
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path, pkg_path_neu3dviewer))
# add neu3dviewer to the path, we need some helper functions in it
import neu3dviewer.utils
from neu3dviewer.img_block_viewer import GUIControl
from neu3dviewer.data_loader import (
    LoadSWCTree, SplitSWCTree, SWCDFSSort, SWCNodeRelabel, GetUndirectedGraph,
    OnDemandVolumeLoader
)

# utility functions
_a  = lambda x: np.array(x, dtype=np.float64)
_ai = lambda x: np.array(x, dtype=int)
_va = lambda *a: np.vstack(a)
_ha = lambda *a: np.hstack(a)   # concatenate along horizontal axis
f_l_gamma = lambda a, g: np.uint16(((np.float64(a) - a.min()) / (a.max()-a.min())) **(1/g) * (a.max()-a.min()) + a.min())
imgshow = lambda im, **kwval: plt.imshow(f_l_gamma(im, 3.0), cmap='gray', origin='lower', **kwval)

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
            print('Warning: redundant point(s), ignoring.')
            # get only the unique points
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
    # interpolation the process by a smooth curve
    curve = SmoothCurve(process_pos, spl_smooth=None)
    blk_sz = 128
    zimg = zarr.open(image_block_path, mode='r')

    # parameters for getting points to be inspected
    t_step = interp_resolution
    n_t = int(curve.length() / t_step) + 1
    t_interp = np.linspace(0, curve.length(), n_t)

    radius_max_soft = blk_sz / 2
    radius_step = interp_resolution
    n_radius = int(radius_max_soft / radius_step) + 1

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
        p, frame = curve.FrenetFrame(t_interp[idx_c_k])
        #print('p =', p)

        if 0:
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
    
def LoadSWCTreeProcess(swc_path):
    ntree = LoadSWCTree(swc_path)
    processes = SplitSWCTree(ntree)
    ntree, processes = SWCDFSSort(ntree, processes)
    #tr_idx = SWCNodeRelabel(ntree)
    #ntree = (tr_idx, ntree[1])
    #ngraph = GetUndirectedGraph(ntree)
    return ntree, processes

class FileLogger:
    """
    Format:
    [
        {
            'clicked_pos': cmip_pos,
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
    
    def clicked_pos(self):
        return [log['clicked_pos'] for log in self.logs]
    
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
    print('Number of porceeses:', len(processes))
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
    def __init__(self, swc_path, image_block_path, pic_path):
        self.swc_path = swc_path
        self.image_block_path = image_block_path
        self.pic_path = pic_path
        # extract neuron id, e.g. 255 in 'neuron#255.lyp.swc'
        self.neu_id = int(os.path.basename(swc_path).split('neuron#')[1].split('.')[0])
        print(f'neuron id: {self.neu_id}')

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
        self.cmip_pixel_size_um = 2.0
        self.gap_size = 3

        # load images and construct index (lookup table)
        self.proc_img_s = [tifffile.imread(s).T for s in self.tif_pathes]
        self.row_idxs = np.cumsum(_ha(0, _ai(
            [i.shape[0] + self.gap_size for i in self.proc_img_s])))
        self.img_height = self.proc_img_s[0].shape[1]

        # default "brightness"
        self.screen_img_gamma = 3.0

        print('Loading swc...', end='')
        self.ntree, self.processes = LoadSWCTreeProcess(self.swc_path)
        print('done.')

        #t_now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #f_pos_path = f'neuron#{self.neu_id}_cmip_marks_{t_now_str}.json'
        f_pos_path = f'neuron#{self.neu_id}_cmip_marks.json'
        self.logger = FileLogger(f_pos_path)
    
    def cmip_pos_to_coordinate(self, cmip_pos):
        # get position in terms of process id (id_proc) and path distance to starting point (local_pos)
        idx_proc = np.searchsorted(self.row_idxs, cmip_pos, 'right') - 1
        id_proc = self.proc_ids[idx_proc]
        local_pos = cmip_pos - self.row_idxs[id_proc]
        if local_pos > self.proc_img_s[id_proc].shape[0]:
            print('Warning: Clicked in the gap.')
            local_pos = self.proc_img_s[id_proc].shape[0] - 1
        local_pos = local_pos * self.cmip_pixel_size_um
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
    
    def ConstructCMIP(self, pos0, screen_size = 1000):
        proc_img_s = self.proc_img_s
        row_idxs = self.row_idxs
        gap_size = self.gap_size
        img_height = self.img_height
        n_screen_rows = 4

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

        clicked_pos = _a(self.logger.clicked_pos())
        b_click_in_local = (clicked_pos > pos0) & (clicked_pos < pos0 + screen_size)
        local_clicked_pos = clicked_pos[b_click_in_local] - pos0

        #print(clicked_pos)
        #print(local_clicked_pos)
        #print(len(self.logger))

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
            axshow(axs, id_s, n_screen_rows, screen_img)
            step = screen_size/n_screen_rows
            ck_idx = (step*id_s <= local_clicked_pos) & (local_clicked_pos < step*(id_s+1))
            ck_pos = local_clicked_pos[ck_idx] - step*id_s
            #print(ck_pos)
            axs[id_s].plot(ck_pos, img_height/2 * np.ones(len(ck_pos)), 'r+')
        fig.canvas.mpl_connect('key_press_event', self.on_cmip_key)
        fig.canvas.mpl_connect('button_press_event', self.on_cmip_mouse)

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

    def on_cmip_mouse(self, event):
        if (event.button == 1 or event.button == 3) and event.inaxes:
            print('=== Clicked ===')
            id_ax = list(self.axs).index(event.inaxes)
            cmip_pos = self.last_pos0 + id_ax * self.screen_size / self.n_screen_row + event.xdata
            #print(f' pos: {event.xdata}, {event.ydata}; screen pos {event.x}, {event.y}')
            #print(' ax id', id_ax)
            print(f'cmip_pos: {cmip_pos:.1f} pixel')
            info = self.cmip_pos_to_coordinate(cmip_pos)
            if event.button == 3:
                t_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                self.logger.Append({'clicked_pos':cmip_pos, 't_str':t_str} | info)
                self.ConstructCMIP(self.last_pos0)
                self.fig.canvas.draw()
                print(f'(recorded, total {len(self.logger)})')
            if event.key == 'v':
                self.fig.canvas.flush_events()
                print('Opening Neu3DViewer...')
                nt = {f'neuron#{self.neu_id}':self.ntree}
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
    gui.Start([fn1, fn2])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Walk a neuron tree, generate and show its circular maximum intencity projection(MIP)."
    )
    parser.add_argument('swc_file_path', nargs='*',  # nargs='*' or '+'
                        default='neuron#122.lyp.swc')
    parser.add_argument('--zarr_dir',
                        default='/mnt/xiaoyy/dataset/zarrblock',
                        help='Path to the zarr directory')
    parser.add_argument('--cmip_dir',
                        default="pic_tmp/",
                        help='Path to the circular MIP directory, for write or read (view mode)')
    parser.add_argument('--view', action='store_true',
                        default=False,
                        help='Enable view mode')
    parser.add_argument('--test',
                        help='Test mode, not for general use')
    parser.add_argument('--verbose', action='store_true',
                        default=False,
                        help='Show more information')
    args = parser.parse_args()
    if args.verbose:
        print("Program arguments:")
        print(args)

    img_block_path = args.zarr_dir
    s_swc_path = args.swc_file_path

    if args.test:
        #swc_path = 'neuron#255.lyp.swc'
        # node_idx = 1936, node_id = 932514
        # xyz: [52785.  28145.6 55668.9]
        # Branch depth: 1
        # Node depth: 1254
        # Path length to root: 23228.5
        if args.test == 'slicing':
            plt.ion()
            Test3dImageSlicing()
            plt.show()
        elif args.test == 'tangent':
            #IPython %run ./neu_walk.py 'neuron#255.lyp.swc' --test tangent
            swc_path = s_swc_path[0]
            node_idx = 1936
            plt.ion()
            WalkTreeTangent(swc_path, img_block_path, node_idx)
            plt.show()
        elif args.test == 'neu3dviewer':
            #swc_path = 'neuron#255.lyp.swc'
            swc_path = s_swc_path[0]
            ntree = LoadSWCTree(swc_path)
            r_c = _a([52785., 28145.6, 55668.9])
            ViewByNeu3DViewer({'abc': ntree}, img_block_path, r_c)
    elif args.view:
        for swc_path in s_swc_path:
            cmip_viewer = TreeCircularMIPViewer(swc_path, img_block_path, args.cmip_dir)
            cmip_viewer.ConstructCMIP(0)
            plt.show()
    else:
        for swc_path in s_swc_path:
            WalkTreeCircularMIP(swc_path, img_block_path, args.cmip_dir, 2)
