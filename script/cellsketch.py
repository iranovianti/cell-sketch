import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from skimage.measure import label
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import model
import utils

class GAN:
	def __init__(self, optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5)):
		self.generator = model.Generator()
		self.discriminator = model.Discriminator()

		self.checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer,
										discriminator_optimizer=optimizer,
										generator=self.generator,
										discriminator=self.discriminator)

	def load_weight(self, weight_path):
		self.checkpoint.restore(weight_path)

	def generate(self, input_image):
		output = self.generator(tf.cast(tf.expand_dims(input_image, axis=0), tf.float32), training=True)[0]
		return output.numpy()

class CellSketch:
	def __init__(self, weight):
		self.generator = GAN()
		self.generator.load_weight(weight)

	def load_weight(self, weight):
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

	def postprocess(self, _sketch, remove_small_objects=False, min_size=100):
		RGB_mask = [utils.binarize(_sketch[:,:,i])for i in range(_sketch.shape[-1])]

		R_mask = utils.refine(RGB_mask[0])
		G_mask = utils.clean(RGB_mask[1], 6)
		B_mask = utils.refine(utils.clean(RGB_mask[2], 6))

		RnG = np.logical_or(R_mask, G_mask)
		RnG = utils.close_gap(RnG, 3)
		R_B = np.where(R_mask, 0, B_mask)

		center = label(R_B)

		edt = ndi.distance_transform_edt(G_mask)

		initial_markers = watershed(-edt, center, mask=G_mask)
		other_markers = (G_mask ^ initial_markers.astype(bool)).astype('int32')
		other_markers += other_markers * initial_markers.max()

		final_markers = np.add.reduce([initial_markers, other_markers])

		instance_mask = watershed(-(ndi.distance_transform_edt(RnG)), final_markers, mask=RnG) #final watershed/marker expansion

		if remove_small_objects:
			#filter instances based on their size (number of nonzero pixels)
			cell_sizes = {i: np.count_nonzero(instance_mask == i) for i in np.unique(instance_mask)[1:]}
			filtered_mask = [(instance_mask == i) for i,size in cell_sizes.items() if size > min_size]
			instance_masks = [filtered_mask[i]*(i+1) for i in range(len(filtered_mask))]
			instance_mask = np.add.reduce(instance_masks)
		
		nuc_mask = utils.dilate(R_B, 2)
		nuc_mask = label(nuc_mask)
		
		return instance_mask, nuc_mask

	def segment(self, image, show_result=False, **kwargs):
		if image.shape != (512, 512, 3):
			image = self.prepare_image(image)

		sketching = self.sketch(image)
		segmentation, nuc_mask = self.postprocess(sketching, **kwargs)

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

		return segmentation, nuc_mask, sketching