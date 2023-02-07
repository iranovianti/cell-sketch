import tensorflow as tf
import os
import glob

H = 512
W = 512

def resize(input_image, real_image, height, width):
	input_image = tf.image.resize(input_image, [H, W],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	real_image = tf.image.resize(real_image, [H, W],
								method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return input_image, real_image

def random_crop(input_image, real_image):
	stacked_image = tf.stack([input_image, real_image], axis=0)
	cropped_image = tf.image.random_crop(stacked_image, size=[2, H, W, 3])
	return cropped_image[0], cropped_image[1]

def random_scale(input_image, real_image, s_range=(1,1.5)):
	scale = tf.random.uniform([], minval=s_range[0], maxval=s_range[1], dtype=tf.float32)
	input_image, real_image = resize(input_image, real_image, H*scale, W*scale)
	input_image, real_image = random_crop(input_image, real_image)

	if tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		real_image = tf.image.flip_left_right(real_image)

	return input_image, real_image

def load_data(image_path, rescale=False, contrast=False):
	image = tf.io.read_file(image_path)
	image = tf.image.decode_jpeg(image)

	width = tf.shape(image)[1] // 2
	inp = image[:, :width, :]
	target = image[:, width:, :]

	if contrast:
		inp = tf.image.random_brightness(inp, 0.2)
		inp = tf.image.random_contrast(inp, 0.5, 2.0)

	if rescale:
		inp, target = random_scale(inp, target, s_range=(1,2))

	inp = inp / 255
	target = target / 255

	inp = tf.cast(inp, tf.float32)
	target = tf.cast(target, tf.float32)

	return inp, target


def Dataset(path, ds_type="train", bs=1, rescale=False, contrast=False):
	assert ds_type in ["train", "val", "test"]

	folder = os.path.join(path, ds_type)
	file_list = os.path.join(folder, '*.jpg')
	len_data = len(glob.glob(file_list))

	dataset = tf.data.Dataset.list_files(file_list)
	dataset = dataset.map(lambda x: load_data(x, rescale=rescale, contrast=contrast))
	if ds_type == "train":
		dataset = dataset.shuffle(len_data)
	dataset = dataset.batch(bs)

	return dataset