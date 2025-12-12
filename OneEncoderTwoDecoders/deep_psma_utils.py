import SimpleITK as sitk, numpy as np

def resample_to_target(moving, target, default=-1000):
    tfm = sitk.TranslationTransform(3)
    return sitk.Resample(moving, target, tfm, sitk.sitkLinear, default)