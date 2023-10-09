import numpy as np
import tifffile
import SimpleITK as sitk
import matplotlib.pyplot as plt

_a  = lambda x: np.array(x, dtype=np.float64)
_af = lambda x: np.array(x, dtype=np.float32)

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
    # use resampling filter to get the normal-plane image
    trfm = sitk.Transform()  # default to identity
    # trfm.SetParameters((1, 0, 0, 0, 1, 0, 0, 0, 1, p_center[0], p_center[1], p_center[2]))
    sz = img3d.shape
    sz = (sz[0], sz[1], 1)
    outOrigin = (0, 0, 0)
    outSpacing = (1, 1, 1)
    outDirection = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    img_normal = sitk.Resample(img3ds, sz, trfm, sitk.sitkLinear,
                               outOrigin, outSpacing, outDirection,
                               0, sitk.sitkUInt16)
    # convert the image to numpy array
    img_normal = sitk.GetArrayFromImage(img_normal)
    img_normal = img_normal[0, :, :]
    return img_normal

if __name__ == '__main__':
    # load the 3D image
    tif_path = "/home/xyy/code/py/neurite_walker/52224-30976-56064.tif"
    img3d = tifffile.imread(tif_path)

    # set the center point and normal vector
    p_center   = _a([0, 0, 0])
    vec_normal = _a([0, 0, 1])  # vectors always follow (x,y,z)
    vec_up     = _a([0, 1, 0])
    # get the normal-plane image
    img_normal = NormalSlice3DImage(img3d, p_center, vec_normal, vec_up)

    # show the normal-plane image by matplotlib
    #plt.ion()
    plt.figure(9)
    # reference image at z==0, facing -z direction, with horizon-x, vertial-y axis.
    plt.imshow(img3d[0,:,:], cmap='gray', origin='lower')

    plt.figure(10)
    # rescale the image by min and max
    img_normal = (img_normal - img_normal.min()) / (img_normal.max() - img_normal.min())
    plt.imshow(img_normal, cmap='gray', origin='lower')
    plt.show()
