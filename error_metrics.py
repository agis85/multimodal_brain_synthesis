import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


def ErrorMetrics(vol_s, vol_t):
    # calculate various error metrics.
    # vol_s should be the synthesized volume (a 3d numpy array) or an array of these volumes
    # vol_t should be the ground truth volume (a 3d numpy array) or an array of these volumes

    vol_s = np.squeeze(vol_s)
    vol_t = np.squeeze(vol_t)

    assert len(vol_s.shape) == len(vol_t.shape) == 3
    assert vol_s.shape[0] == vol_t.shape[0]
    assert vol_s.shape[1] == vol_t.shape[1]
    assert vol_s.shape[2] == vol_t.shape[2]

    vol_s[vol_t == 0] = 0
    vol_s[vol_s < 0] = 0

    errors = {}

    errors['MSE'] = np.mean((vol_s - vol_t) ** 2.)
    errors['SSIM'] = ssim(vol_t, vol_s)
    dr = np.max([vol_s.max(), vol_t.max()]) - np.min([vol_s.min(), vol_t.min()])
    errors['PSNR'] = psnr(vol_t, vol_s, dynamic_range=dr)

    # non background in both
    non_bg = (vol_t != vol_t[0, 0, 0])
    errors['SSIM_NBG'] = ssim(vol_t[non_bg], vol_s[non_bg])
    dr = np.max([vol_t[non_bg].max(), vol_s[non_bg].max()]) - np.min([vol_t[non_bg].min(), vol_s[non_bg].min()])
    errors['PSNR_NBG'] = psnr(vol_t[non_bg], vol_s[non_bg], dynamic_range=dr)

    vol_s_non_bg = vol_s[non_bg].flatten()
    vol_t_non_bg = vol_t[non_bg].flatten()
    errors['MSE_NBG'] = np.mean((vol_s_non_bg - vol_t_non_bg) ** 2.)

    return errors
