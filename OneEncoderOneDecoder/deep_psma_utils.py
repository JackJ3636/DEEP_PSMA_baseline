import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
from skimage.segmentation import watershed

def plot_mip(image, lab_ttb, lab_norm, outpath, title='Early Fusion MIP View - TTB & Normal', clim_max=2.5, show=False):
    nar = sitk.GetArrayFromImage(lab_norm)
    tar = sitk.GetArrayFromImage(lab_ttb)    
    plt.figure(figsize=[12, 6])
    spacing = image.GetSpacing()
    aspect = spacing[2] / spacing[0]
    ar = sitk.GetArrayFromImage(image)

    plt.subplot(121)
    plt.imshow(np.flipud(np.amax(ar, 1)), aspect=aspect, cmap='Greys', clim=[0, clim_max])
    plt.contour(np.flipud(np.amax(tar, 1)), colors='r', levels=[0.5], linewidths=0.8)
    plt.contourf(np.flipud(np.amax(nar, 1)), colors='b', levels=[0.5, 1.5], linewidths=0.8, alpha=0.7)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(np.flipud(np.amax(ar, 2)), aspect=aspect, cmap='Greys', clim=[0, clim_max])
    plt.contour(np.flipud(np.amax(tar, 2)), colors='r', levels=[0.5], linewidths=0.8)
    plt.contourf(np.flipud(np.amax(nar, 2)), colors='b', levels=[0.5, 1.5], linewidths=0.8, alpha=0.7)
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close('all')

def get_expansion_sphere(radius, spacing):
    xlim = np.ceil(radius / spacing[0])
    ylim = np.ceil(radius / spacing[1])
    zlim = np.ceil(radius / spacing[2])
    x, y, z = np.meshgrid(np.arange(-xlim, xlim + 1),
                          np.arange(-ylim, ylim + 1),
                          np.arange(-zlim, zlim + 1), indexing='ij')
    sphere = (np.sqrt((x * spacing[0])**2 + (y * spacing[1])**2 + (z * spacing[2])**2) <= radius).astype(np.float32)
    return sphere

def create_subregion_labels(image, threshold_array, local_maxima_threshold, sphere_radius):
    img_array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    sphere = get_expansion_sphere(sphere_radius, spacing)
    peaks = morphology.h_maxima(img_array, local_maxima_threshold, sphere)
    numbered_peaks, n_peaks = ndimage.label(peaks)
    labels_ws = watershed(-img_array, numbered_peaks, mask=threshold_array)
    
    missing_peaks_ar = threshold_array - (labels_ws > 0).astype('uint8')
    if missing_peaks_ar.sum() > 0:
        missing_peaks_ar_numbered, n_missing = ndimage.label(missing_peaks_ar)
        for i in range(n_missing):
            labels_ws[missing_peaks_ar_numbered == (i + 1)] = n_peaks + i
    
    ws_im = sitk.GetImageFromArray(labels_ws.astype(np.int16))
    ws_im.CopyInformation(image)
    return ws_im
