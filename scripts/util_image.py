#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2021-11-24 16:54:19

import sys
import cv2
import math
import torch
import random
import numpy as np
from scipy import fft
from pathlib import Path
from einops import rearrange
from skimage import img_as_ubyte, img_as_float32

# --------------------------Metrics----------------------------
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(im1, im2, border=0, ycbcr=False):
    '''
    SSIM the same outputs as MATLAB's
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    if im1.ndim == 2:
        return ssim(im1, im2)
    elif im1.ndim == 3:
        if im1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(im1[:,:,i], im2[:,:,i]))
            return np.array(ssims).mean()
        elif im1.shape[2] == 1:
            return ssim(np.squeeze(im1), np.squeeze(im2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0, ycbcr=False):
    '''
    PSNR metric.
    im1, im2: h x w x , [0, 255], uint8
    '''
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def batch_PSNR(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    h, w = Iclean.shape[2:]
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return PSNR

def batch_SSIM(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return SSIM

def normalize_np(im, mean=0.5, std=0.5, reverse=False):
    '''
    Input:
        im: h x w x c, numpy array
        Normalize: (im - mean) / std
        Reverse: im * std + mean

    '''
    if not isinstance(mean, (list, tuple)):
        mean = [mean, ] * im.shape[2]
    mean = np.array(mean).reshape([1, 1, im.shape[2]])

    if not isinstance(std, (list, tuple)):
        std = [std, ] * im.shape[2]
    std = np.array(std).reshape([1, 1, im.shape[2]])

    if not reverse:
        out = (im.astype(np.float32) - mean) / std
    else:
        out = im.astype(np.float32) * std + mean
    return out

def normalize_th(im, mean=0.5, std=0.5, reverse=False):
    '''
    Input:
        im: b x c x h x w, torch tensor
        Normalize: (im - mean) / std
        Reverse: im * std + mean

    '''
    if not isinstance(mean, (list, tuple)):
        mean = [mean, ] * im.shape[1]
    mean = torch.tensor(mean, device=im.device).view([1, im.shape[1], 1, 1])

    if not isinstance(std, (list, tuple)):
        std = [std, ] * im.shape[1]
    std = torch.tensor(std, device=im.device).view([1, im.shape[1], 1, 1])

    if not reverse:
        out = (im - mean) / std
    else:
        out = im * std + mean
    return out

# ------------------------Image format--------------------------
def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: uint8 [0,255] or float [0,1]
        only_y: only return Y channel
    '''
    # transform to float64 data type, range [0, 255]
    if im.dtype == np.uint8:
        im_temp = im.astype(np.float64)
    else:
        im_temp = (im * 255).astype(np.float64)

    # convert
    if only_y:
        rlt = np.dot(im_temp, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im_temp, np.array([[65.481,  -37.797, 112.0  ],
                                           [128.553, -74.203, -93.786],
                                           [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if im.dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(im.dtype)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    # transform to range [0,255.0]
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    flag_tensor = torch.is_tensor(tensor)
    if flag_tensor:
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1 and flag_tensor:
        result = result[0]
    return result

def img2tensor(imgs, out_type=torch.float32):
    """Convert image numpy arrays into torch tensor.
    Args:
        imgs (Array or list[array]): Accept shapes:
            3) list of numpy arrays
            1) 3D numpy array of shape (H x W x 3/1);
            2) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.

    Returns:
        (array or list): 4D ndarray of shape (1 x C x H x W)
    """

    def _img2tensor(img):
        if img.ndim == 2:
            tensor = torch.from_numpy(img[None, None,]).type(out_type)
        elif img.ndim == 3:
            tensor = torch.from_numpy(rearrange(img, 'h w c -> c h w')).type(out_type).unsqueeze(0)
        else:
            raise TypeError(f'2D or 3D numpy array expected, got{img.ndim}D array')
        return tensor

    if not (isinstance(imgs, np.ndarray) or (isinstance(imgs, list) and all(isinstance(t, np.ndarray) for t in imgs))):
        raise TypeError(f'Numpy array or list of numpy array expected, got {type(imgs)}')

    flag_numpy = isinstance(imgs, np.ndarray)
    if flag_numpy:
        imgs = [imgs,]
    result = []
    for _img in imgs:
        result.append(_img2tensor(_img))

    if len(result) == 1 and flag_numpy:
        result = result[0]
    return result

# ------------------------Image I/O-----------------------------
def imread(path, chn='rgb', dtype='float32'):
    '''
    Read image.
    chn: 'rgb', 'bgr' or 'gray'
    out:
        im: h x w x c, numpy tensor
    '''
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    try:
        if chn.lower() == 'rgb':
            if im.ndim == 3:
                im = bgr2rgb(im)
            else:
                im = np.stack((im, im, im), axis=2)
        elif chn.lower() == 'gray':
            assert im.ndim == 2
    except:
        print(str(path))

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype ==  'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    return im

def imwrite(im_in, path, chn='rgb', dtype_in='float32', qf=None):
    '''
    Save image.
    Input:
        im: h x w x c, numpy tensor
        path: the saving path
        chn: the channel order of the im,
    '''
    im = im_in.copy()
    if isinstance(path, str):
        path = Path(path)
    if dtype_in != 'uint8':
        im = img_as_ubyte(im)

    if chn.lower() == 'rgb' and im.ndim == 3:
        im = rgb2bgr(im)

    if qf is not None and path.suffix.lower() in ['.jpg', '.jpeg']:
        flag = cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    else:
        flag = cv2.imwrite(str(path), im)

    return flag

def jpeg_compress(im, qf, chn_in='rgb'):
    '''
    Input:
        im: h x w x 3 array
        qf: compress factor, (0, 100]
        chn_in: 'rgb' or 'bgr'
    Return:
        Compressed Image with channel order: chn_in
    '''
    # transform to BGR channle and uint8 data type
    im_bgr = rgb2bgr(im) if chn_in.lower() == 'rgb' else im
    if im.dtype != np.dtype('uint8'): im_bgr = img_as_ubyte(im_bgr)

    # JPEG compress
    flag, encimg = cv2.imencode('.jpg', im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    assert flag
    im_jpg_bgr = cv2.imdecode(encimg, 1)    # uint8, BGR

    # transform back to original channel and the original data type
    im_out = bgr2rgb(im_jpg_bgr) if chn_in.lower() == 'rgb' else im_jpg_bgr
    if im.dtype != np.dtype('uint8'): im_out = img_as_float32(im_out).astype(im.dtype)
    return im_out

# ------------------------Augmentation-----------------------------
def data_aug_np(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out.copy()

def inverse_data_aug_np(image, mode):
    '''
    Performs inverse data augmentation of the input image
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image, axes=(1,0))
    elif mode == 3:
        out = np.flipud(image)
        out = np.rot90(out, axes=(1,0))
    elif mode == 4:
        out = np.rot90(image, k=2, axes=(1,0))
    elif mode == 5:
        out = np.flipud(image)
        out = np.rot90(out, k=2, axes=(1,0))
    elif mode == 6:
        out = np.rot90(image, k=3, axes=(1,0))
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(image)
        out = np.rot90(out, k=3, axes=(1,0))
    else:
        raise Exception('Invalid choice of image transformation')

    return out

class SpatialAug:
    def __init__(self):
        pass

    def __call__(self, im, flag=None):
        if flag is None:
            flag = random.randint(0, 7)

        out = data_aug_np(im, flag)
        return out

# ----------------------Visualization----------------------------
def imshow(x, title=None, cbar=False):
    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

# -----------------------Covolution------------------------------
def imgrad(im, pading_mode='mirror'):
    '''
    Calculate image gradient.
    Input:
        im: h x w x c numpy array
    '''
    from scipy.ndimage import correlate  # lazy import
    wx = np.array([[0, 0, 0],
                   [-1, 1, 0],
                   [0, 0, 0]], dtype=np.float32)
    wy = np.array([[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=np.float32)
    if im.ndim == 3:
        gradx = np.stack(
                [correlate(im[:,:,c], wx, mode=pading_mode) for c in range(im.shape[2])],
                axis=2
                )
        grady = np.stack(
                [correlate(im[:,:,c], wy, mode=pading_mode) for c in range(im.shape[2])],
                axis=2
                )
        grad = np.concatenate((gradx, grady), axis=2)
    else:
        gradx = correlate(im, wx, mode=pading_mode)
        grady = correlate(im, wy, mode=pading_mode)
        grad = np.stack((gradx, grady), axis=2)

    return {'gradx': gradx, 'grady': grady, 'grad':grad}

def imgrad_fft(im):
    '''
    Calculate image gradient.
    Input:
        im: h x w x c numpy array
    '''
    wx = np.rot90(np.array([[0, 0, 0],
                            [-1, 1, 0],
                            [0, 0, 0]], dtype=np.float32), k=2)
    gradx = convfft(im, wx)
    wy = np.rot90(np.array([[0, -1, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=np.float32), k=2)
    grady = convfft(im, wy)
    grad = np.concatenate((gradx, grady), axis=2)

    return {'gradx': gradx, 'grady': grady, 'grad':grad}

def convfft(im, weight):
    '''
    Convolution with FFT
    Input:
        im: h1 x w1 x c numpy array
        weight: h2 x w2 numpy array
    Output:
        out: h1 x w1 x c numpy array
    '''
    axes = (0,1)
    otf = psf2otf(weight, im.shape[:2])
    if im.ndim == 3:
        otf = np.tile(otf[:, :, None], (1,1,im.shape[2]))
    out = fft.ifft2(fft.fft2(im, axes=axes) * otf, axes=axes).real
    return out

def psf2otf(psf, shape):
    """
    MATLAB psf2otf function.
    Borrowed from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py.
    Input:
        psf : h x w numpy array
        shape : list or tuple, output shape of the OTF array
    Output:
        otf : OTF array with the desirable shape
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

# ----------------------Patch Cropping----------------------------
def random_crop(im, pch_size):
    '''
    Randomly crop a patch from the give image.
    '''
    h, w = im.shape[:2]
    if h == pch_size and w == pch_size:
        im_pch = im
    else:
        assert h >= pch_size or w >= pch_size
        ind_h = random.randint(0, h-pch_size)
        ind_w = random.randint(0, w-pch_size)
        im_pch = im[ind_h:ind_h+pch_size, ind_w:ind_w+pch_size,]

    return im_pch

class RandomCrop:
    def __init__(self, pch_size):
        self.pch_size = pch_size

    def __call__(self, im):
        return random_crop(im, self.pch_size)

class ImageSpliterNp:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: h x w x c, numpy array, [0, 1], low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        if im.ndim == 2:
            im = im[:, :, None]

        height, width, chn = im.shape
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = np.zeros([height*sf, width*sf, chn], dtype=im.dtype)
        self.pixel_count = np.zeros([height*sf, width*sf, chn], dtype=im.dtype)

    def extract_starts(self, length):
        starts = list(range(0, length, self.stride))
        if starts[-1] + self.pch_size > length:
            starts[-1] = length - self.pch_size
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_pchs < self.length:
            w_start_idx = self.num_pchs // len(self.height_starts_list)
            w_start = self.width_starts_list[w_start_idx] * self.sf
            w_end = w_start + self.pch_size * self.sf

            h_start_idx = self.num_pchs % len(self.height_starts_list)
            h_start = self.height_starts_list[h_start_idx] * self.sf
            h_end = h_start + self.pch_size * self.sf

            pch = self.im_ori[h_start:h_end, w_start:w_end,]
            self.w_start, self.w_end = w_start, w_end
            self.h_start, self.h_end = h_start, h_end

            self.num_pchs += 1
        else:
            raise StopIteration(0)

        return pch, (h_start, h_end, w_start, w_end)

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: pch_size x pch_size x 3, [0,1]
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert np.all(self.pixel_count != 0)
        return self.im_res / self.pixel_count

class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution
        '''
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf

        bs, chn, height, width= im.shape
        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.length = self.__len__()
        self.num_pchs = 0

        self.im_ori = im
        self.im_res = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)
        self.pixel_count = torch.zeros([bs, chn, height*sf, width*sf], dtype=im.dtype, device=im.device)

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [0,]
        else:
            starts = list(range(0, length, self.stride))
            for i in range(len(starts)):
                if starts[i] + self.pch_size > length:
                    starts[i] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_pchs < self.length:
            w_start_idx = self.num_pchs // len(self.height_starts_list)
            w_start = self.width_starts_list[w_start_idx]
            w_end = w_start + self.pch_size

            h_start_idx = self.num_pchs % len(self.height_starts_list)
            h_start = self.height_starts_list[h_start_idx]
            h_end = h_start + self.pch_size

            pch = self.im_ori[:, :, h_start:h_end, w_start:w_end,]

            h_start *= self.sf
            h_end *= self.sf
            w_start *= self.sf
            w_end *= self.sf

            self.w_start, self.w_end = w_start, w_end
            self.h_start, self.h_end = h_start, h_end

            self.num_pchs += 1
        else:
            raise StopIteration()

        return pch, (h_start, h_end, w_start, w_end)

    def update(self, pch_res, index_infos):
        '''
        Input:
            pch_res: n x c x pch_size x pch_size, float
            index_infos: (h_start, h_end, w_start, w_end)
        '''
        if index_infos is None:
            w_start, w_end = self.w_start, self.w_end
            h_start, h_end = self.h_start, self.h_end
        else:
            h_start, h_end, w_start, w_end = index_infos

        self.im_res[:, :, h_start:h_end, w_start:w_end] += pch_res
        self.pixel_count[:, :, h_start:h_end, w_start:w_end] += 1

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)

# ----------------------Patch Cropping----------------------------
class Clamper:
    def __init__(self, min_max=(-1, 1)):
        self.min_bound, self.max_bound = min_max[0], min_max[1]

    def __call__(self, im):
        if isinstance(im, np.ndarray):
            return np.clip(im, a_min=self.min_bound, a_max=self.max_bound)
        elif isinstance(im, torch.Tensor):
            return torch.clamp(im, min=self.min_bound, max=self.max_bound)
        else:
            raise TypeError(f'ndarray or Tensor expected, got {type(im)}')

if __name__ == '__main__':
    im = np.random.randn(64, 64, 3).astype(np.float32)

    grad1 = imgrad(im)['grad']
    grad2 = imgrad_fft(im)['grad']

    error = np.abs(grad1 -grad2).max()
    mean_error = np.abs(grad1 -grad2).mean()
    print('The largest error is {:.2e}'.format(error))
    print('The mean error is {:.2e}'.format(mean_error))
