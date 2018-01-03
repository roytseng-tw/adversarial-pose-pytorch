import random
import numpy as np
import skimage.transform as sktf
import matplotlib.pyplot as plt


""" Random """
def randn():
    return random.gauss(0, 1)

def rand():
    return random.random()

def rnd(x):
    '''umich hourglass mpii random function'''
    return max(-2 * x, min(2 * x, randn() * x))


""" Visualization """
def show_sample(img, label):  # FIXME: color blending is not right, diff color for each joint
    nJoints = label.shape[0]
    white = np.ones((4,) + img.shape[1:3])
    new_img = white.copy()
    new_img[:3] = img * 0.5
    for i in range(nJoints):
        new_img += 0.5 * white * sktf.resize(label[i], img.shape[1:3], preserve_range=True)
        # print(label[i].max())
        # plt.subplot(121)
        # plt.imshow(np.transpose(new_img, [1, 2, 0]))
        # plt.subplot(122)
        # plt.imshow(label[i])
        # plt.show()
    return np.transpose(new_img, [1, 2, 0])


""" Label """
def create_label(imsize, pt, sigma, distro_type='Gaussian'):
    label = np.zeros(imsize)
    # Check that any part of the distro is in-bounds
    ul = np.math.floor(pt[0] - 3 * sigma), np.math.floor(pt[1] - 3 * sigma)
    br = np.math.floor(pt[0] + 3 * sigma), np.math.floor(pt[1] + 3 * sigma)
    # If not, return the blank label
    if ul[0] >= imsize[1] or ul[1] >= imsize[0] or br[0] < 0 or br[1] < 0:
        return label

    # Generate distro
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    '''Note:
    original torch impl: `local g = image.gaussian(size)`
    equals to `gaussian(size, sigma=0.25*size)` here
    '''
    if distro_type == 'Gaussian':
        distro = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif distro_type == 'Cauchy':  # IS THIS CORRECT ???
        distro = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)
        # distro = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) * np.pi)

    # Usable distro range
    distro_x = max(0, -ul[0]), min(br[0], imsize[1]) - ul[0]
    distro_y = max(0, -ul[1]), min(br[1], imsize[0]) - ul[1]
    assert (distro_x[0] >= 0 and distro_y[0] >= 0), '{}, {}'.format(distro_x, distro_y)
    # label range
    label_x = max(0, ul[0]), min(br[0], imsize[1])
    label_y = max(0, ul[1]), min(br[1], imsize[0])
    label[label_y[0]:label_y[1], label_x[0]:label_x[1]] = \
        distro[distro_y[0]:distro_y[1], distro_x[0]:distro_x[1]]
    return label


""" Flip """
def fliplr_labels(labels, matchedParts, joint_dim=1, width_dim=3):
    """fliplr the joint labels, defaults (B, C, H, W)
    """
    # flip horizontally
    labels = np.flip(labels, axis=width_dim)
    # Change left-right parts
    perm = np.arange(labels.shape[joint_dim])
    for i, j in matchedParts:
        perm[i] = j
        perm[j] = i
    labels = np.take(labels, perm, axis=joint_dim)
    return labels

def fliplr_coords(pts, width, matchedParts):
    # Flip horizontally (only flip valid points)
    pts = np.array([(width - x, y) if x > 0 else (x, y) for x, y in pts])
    # Change left-right parts
    perm = np.arange(pts.shape[0])
    for i, j in matchedParts:
        perm[i] = j
        perm[j] = i
    pts = pts[perm]
    return pts


""" Transform, Crop """
def get_transform(center, scale, rot, res, invert=False):
    '''Prepare transformation matrix (scale, rot).
    '''
    h = 200 * scale
    t = np.eye(3)  # transformation matrix
    # scale
    t[0, 0] = res[1] / h
    t[1, 1] = res[0] / h
    # translation
    t[0, 2] = res[1] * (-center[0] / h + .5)
    t[1, 2] = res[0] * (-center[1] / h + .5)
    # rotation
    if rot != 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[:2, :2] = [[cs, -sn],
                           [sn, cs]]
        rot_mat[2, 2] = 1
        # Need to make sure rotation is around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    if invert:
        t = np.linalg.inv(t)
    return t

def transform(pts, center, scale, rot, res, invert=False):
    """ Transform points from original coord to new coord
    pts: 2 * n array
    """
    t = get_transform(center, scale, rot, [res, res], invert)
    pts = np.array(pts)
    assert pts.shape[0] == 2, pts.shape
    if pts.ndim == 1:
        pts = np.array([pts[0], pts[1], 1])
    else:
        pts = np.concatenate([pts, np.ones((1, pts.shape[1]))], axis=0)
    new_pt = np.dot(t, pts)
    return new_pt[:2].astype(int)

def crop(img, center, scale, rot, res):
    '''
    res: single value of targeted output image resolution
    rot: in degrees
    '''
    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    # print(center, scale, rot, ht, wd)
    sf = scale * 200.0 / res
    # print(sf)
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            # Zoomed out so much that the image is now a single pixel or less
            return np.zeros(res, res) if img.ndim == 2 \
                else np.zeros(res, res, img.shape[2])
        else:
            img = sktf.resize(img, [new_ht, new_wd], preserve_range=True)
            ht, wd = img.shape[0], img.shape[1]
    # print(ht, wd)
    # Calculate upper left and bottom right coordinates defining crop region
    center = center / sf
    scale = scale / sf
    # print(center, scale)
    ul = transform([0, 0], center, scale, 0, res, invert=True)
    br = transform([res, res], center, scale, 0, res, invert=True)
    if sf >= 2:
         br += - (br - ul - res)
    # print(ul, br)
    # Padding so that when rotated proper amount of context is included
    pad = np.math.ceil(np.linalg.norm(br - ul) / 2 - (br[0] - ul[0]) / 2)
    # print(pad)
    if rot != 0:
        ul -= pad
        br += pad
    # print(ul, br)
    # Define the range of pixels to take from the old image
    old_x = max(0, ul[0]), min(br[0], wd)
    old_y = max(0, ul[1]), min(br[1], ht)
    # print(old_x, old_y)
    # And where to put them in the new image
    new_x = max(0, -ul[0]), min(br[0], wd) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], ht) - ul[1]
    # print(new_x, new_y)
    # Initialize new image and copy pixels over
    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    # print(new_shape)
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if rot != 0:
        # Rotate the image and remove padded area
        new_img = sktf.rotate(new_img, rot, preserve_range=True)
        new_img = new_img[pad:-pad, pad:-pad]

    if sf < 2:
        new_img = sktf.resize(new_img, [res, res], preserve_range=True)

    return new_img
