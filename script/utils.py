import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import try_all_threshold, threshold_li, threshold_mean, threshold_triangle, threshold_otsu
from skimage.morphology import binary_erosion, disk, dilation, closing, opening
from skimage.filters import roberts, sobel
from skimage.filters import gaussian as gauss_blur

def normalize(a):
	return ((a-np.min(a))/(np.max(a)-np.min(a)))

def clean(binary_mask, pixel_size):
	return opening(binary_mask, disk(pixel_size))

def close_gap(binary_mask, pixel_size):
	return closing(binary_mask, disk(pixel_size))

def refine(binary_mask, close=True, clean_=False, c_size=2, s_blur=1):
	if close:
		binary_mask = closing(binary_mask, disk(c_size))
	if clean_:
		binary_mask = clean(binary_mask, 2)
	blurred = gauss_blur(binary_mask, sigma=s_blur)
	refined = blurred > threshold_otsu(blurred)
	return refined

def dilate(binary_mask, pixel_size):
	return dilation(binary_mask, disk(pixel_size))

def binarize(image):
	return image > threshold_otsu(image)

def overlay_mask(mask, cmap='viridis', alpha=1):
	masked = np.ma.masked_where(mask == 0, mask)
	cmap = plt.cm.get_cmap(cmap)
	cmap.set_bad(alpha=0)
	return plt.imshow(masked, alpha=alpha, cmap=cmap, interpolation='none')

def get_edge(mask):
	return dilation(roberts(mask) > 0, disk(2))

def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]