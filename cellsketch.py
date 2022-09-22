import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.measure import label
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import gan
import utils

class CellSketch:
	def __init__(self, weight):
		self.generator = gan.GAN()
		self.generator.load_weight(weight)

	def prepare_image(self, image):
		if len(image.shape) == 2:
			image = np.dstack((image,)*3)

		if image.shape[0] != image.shape[1]:
			ps = np.min(image.shape[:2])
			image = image[:ps,:ps]

		if image.shape[:2] != (512, 512):
			image = cv2.resize(image, (512, 512))

		image = utils.normalize(image) * 255

		return image.astype('uint8')

	def sketch(self, image):
		if image.shape != (512, 512, 3):
			image = self.prepare_image(image)

		input_image = utils.normalize(image)
		output = self.generator.generate(input_image)
		return output

	def postprocess(self, sktch, ws_type='single'):
		RGB_mask = [utils.binarize(sktch[:,:,i])for i in range(sktch.shape[-1])]

		R_mask = utils.refine(RGB_mask[0])
		G_mask = utils.clean(RGB_mask[1], 6)
		B_mask = utils.refine(utils.clean(RGB_mask[2], 6))

		R_G = np.logical_or(R_mask, G_mask)
		R_G = utils.close_gap(R_G, 3)
		R_B = np.where(R_mask, 0, B_mask)

		assert ws_type in ['single', 'repeat']

		center = label(R_B)

		if ws_type == 'single':
			edt = ndi.distance_transform_edt(G_mask)
			instance_mask = watershed(-edt, center, mask=R_G)

		elif ws_type == 'repeat':
			init_instances = label(G_mask)

			lbl_ws = []
			for i in np.unique(init_instances)[1:]:
				_mask = init_instances == i
				_edt = ndi.distance_transform_edt(_mask)
				l_ws = watershed(-_edt, center, mask=_mask)
				lbl_ws.append(l_ws)

			lbl_ws_combined = np.add.reduce(lbl_ws)
			final_mask = utils.dilate(lbl_ws_combined.astype(bool), 3)
			edt = ndi.distance_transform_edt(final_mask)

			instance_mask = watershed(-edt, lbl_ws_combined, mask=final_mask)

		return instance_mask

	def segment(self, image, ws_type='single', show_result=False):
		if image.shape != (512, 512, 3):
			image = self.prepare_image(image)

		sketching = self.sketch(image)
		segmentation = self.postprocess(sketching, ws_type=ws_type)

		if show_result:
			plt.figure(figsize=(15,5.5))

			plt.subplot(131)
			plt.title('Input image')
			plt.imshow(image)
			plt.axis('off')

			plt.subplot(132)
			plt.title('Sketching')
			plt.imshow(sketching)
			plt.axis('off')

			plt.subplot(133)
			plt.title('Segmentation')
			plt.imshow(image, interpolation='none')
			utils.overlay_mask(utils.get_edge(segmentation), cmap='viridis_r')
			plt.axis('off')

			plt.show()

		return segmentation, sketching