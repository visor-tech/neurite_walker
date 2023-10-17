#!/usr/bin/env python3
# add neu3dviewer to the path, we need some helper function in it

# tips in ipython
#%reload_ext autoreload
#%autoreload 2

import os
import sys
import glob   # for list files
import numpy as np
from numpy import diff, sin, cos, pi, linspace
from numpy.linalg import norm
import scipy.ndimage
import scipy.interpolate as interpolate

import tifffile
import SimpleITK as sitk
import zarr

import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel, ylabel, title, figure

# add path of current py file
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path, 'external/neu3dviewer'))
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
    q = p + b   # if p is np array, b can be a number of array
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
        p = self(t)
        # get tanget vector
        dp = self(t, der=1)  # approximately we have ||dp|| = 1
        l2 = norm(dp, axis=-1)
        if np.any(l2):
            dp = dp / l2.reshape(list(l2.shape)+[1])   # reshape to help auto broadcasting
        return p, dp

    def PointTangentNormal(self, t):
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
    
def WalkTreeCircularMIP(swc_path, image_block_path, resolution):
    # get an ordered and continuous node index tree and its graph
    print("WalkTreeCircularMIP")
    ntree = LoadSWCTree(swc_path)
    processes = SplitSWCTree(ntree)
    ntree, processes = SWCDFSSort(ntree, processes)
    #tr_idx = SWCNodeRelabel(ntree)
    #ntree = (tr_idx, ntree[1])
    #ngraph = GetUndirectedGraph(ntree)

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
        
        img_out_name = f'pic_tmp/{swc_name}_cmip_proc{idx_processes}.tif'
        tifffile.imwrite(img_out_name, axon_circular_mip.T)

def ViewTreeCircularMIP(neu_id, pos0, swc_path, image_block_path, pic_path):
    # list and sort pic path
    tif_pathes = glob.glob(os.path.join(pic_path, f'neuron#{neu_id}_rmip_*.tif'))
    get_proc_id = lambda s: int(s.split('proc')[1].split('.')[0])
    tif_pathes = sorted(tif_pathes, key=get_proc_id)
    proc_ids = [get_proc_id(os.path.basename(s)) for s in tif_pathes]
    
    #print('\n'.join(tif_pathes))
    #print(proc_ids)

    gap_size = 3
    screen_size = 1000

    proc_img_s = [tifffile.imread(s).T for s in tif_pathes]
    row_idxs = np.cumsum(_ha(0, _ai([i.shape[0]+gap_size for i in proc_img_s])))
    img_height = proc_img_s[0].shape[1]

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
        img = proc_img_s[id_img][i_bg:i_ed, :]
        screen_img[screen_img_pos:screen_img_pos + i_ed - i_bg, :] = img
        screen_img_pos += i_ed - i_bg + gap_size
        id_img += 1
        i_bg = 0

    figure(301).clear()
    fig, axs = plt.subplots(4, num=301)
    fig.suptitle(f'cMIP, neuron {neu_id}, pos {pos0}')
    axshow = lambda axs, k, n, im, **kwval: \
        axs[k].imshow(f_l_gamma( \
                im[int(screen_size/n*k) : int(screen_size/n*(k+1)), :].T,
                3.0),
            cmap='gray', origin='lower', **kwval)
    axshow(axs, 0, 4, screen_img)
    axshow(axs, 1, 4, screen_img)
    axshow(axs, 2, 4, screen_img)
    axshow(axs, 3, 4, screen_img)
    fig.canvas.mpl_connect('key_press_event', on_cmip_key)
    fig.canvas.mpl_connect('button_press_event', on_cmip_mouse)

def on_cmip_key(event):
    if event.key == 'y':
        print('hello y')

def on_cmip_mouse(event):
    if event.button == 1:
        print(f'left axis {event.inaxes} pos: {event.xdata}, {event.ydata}; screen pos {event.x}, {event.y}')

if __name__ == '__main__':
    #swc_path = 'neuron#255.lyp.swc'
    # node_idx = 1936, node_id = 932514
    # xyz: [52785.  28145.6 55668.9]
    # Branch depth: 1
    # Node depth: 1254
    # Path length to root: 23228.5

    interactive_mode = False
    #block_lym_path = 'RM009_traced_blocks/full_set/block.lym'
    img_block_path = '/mnt/xiaoyy/dataset/zarrblock'

    if len(sys.argv) == 1:
        s_swc_path = ['neuron#122.lyp.swc']
    elif sys.argv[1].startswith('--'):
        plt.ion()
        if sys.argv[1] == '--test_slicing':
            Test3dImageSlicing()
        elif sys.argv[1] == '--tangent':
            swc_path = 'neuron#255.lyp.swc'
            node_idx = 1936
            WalkTreeTangent(swc_path, img_block_path, node_idx)
        elif sys.argv[1] == '--view':
            swc_path = 'neuron#122.lyp.swc'
            ViewTreeCircularMIP(122, 100, swc_path, img_block_path, 'pic_rm009_1.6.6')
        else:
            print('Hello?')
        plt.show()
        sys.exit(0)
    else:
        s_swc_path = sys.argv[1:]

    for swc_path in s_swc_path:
        WalkTreeCircularMIP(swc_path, img_block_path, 2)
