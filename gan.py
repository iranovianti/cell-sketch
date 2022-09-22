import tensorflow as tf
import numpy as np

def downsample(filters, size, apply_batchnorm=True):
	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
										kernel_initializer=initializer, use_bias=False))

	if apply_batchnorm:
		result.add(tf.keras.layers.BatchNormalization())

	result.add(tf.keras.layers.LeakyReLU())

	return result

def upsample(filters, size, apply_dropout=False):
	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
												padding='same',
												kernel_initializer=initializer,
												use_bias=False))
	result.add(tf.keras.layers.BatchNormalization())

	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result

OUTPUT_CHANNELS = 3

def Generator():
	inputs = tf.keras.layers.Input(shape=[512, 512, 3])

	down_stack = [
		downsample(64, 4, apply_batchnorm=False),
		downsample(128, 4),
		downsample(256, 4),
		downsample(512, 4),
		downsample(512, 4),
		downsample(512, 4),
		downsample(512, 4),
		downsample(512, 4),
		]

	up_stack = [
		upsample(512, 4, apply_dropout=True),
		upsample(512, 4, apply_dropout=True),
		upsample(512, 4, apply_dropout=True),
		upsample(512, 4),
		upsample(256, 4),
		upsample(128, 4),
		upsample(64, 4),
		]

	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
											strides=2,
											padding='same',
											kernel_initializer=initializer,
											activation='sigmoid')

	x = inputs

	skips = []
	for down in down_stack:
		x = down(x)
		skips.append(x)

	skips = reversed(skips[:-1])

	for up, skip in zip(up_stack, skips):
		x = up(x)
		x = tf.keras.layers.Concatenate()([x, skip])

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
	initializer = tf.random_normal_initializer(0., 0.02)

	inp = tf.keras.layers.Input(shape=[512, 512, 3], name='input_image')
	tar = tf.keras.layers.Input(shape=[512, 512, 3], name='target_image')

	x = tf.keras.layers.concatenate([inp, tar])

	down1 = downsample(64, 4, False)(x)
	down2 = downsample(128, 4)(down1)
	down3 = downsample(256, 4)(down2)

	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
	conv = tf.keras.layers.Conv2D(512, 4, strides=1,
									kernel_initializer=initializer,
									use_bias=False)(zero_pad1)

	batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

	leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

	last = tf.keras.layers.Conv2D(1, 4, strides=1,
									kernel_initializer=initializer)(zero_pad2)

	return tf.keras.Model(inputs=[inp, tar], outputs=last)

class GAN:
	def __init__(self, optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5)):
		self.generator = Generator()
		self.discriminator = Discriminator()

		self.checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer,
										discriminator_optimizer=optimizer,
										generator=self.generator,
										discriminator=self.discriminator)

	def load_weight(self, weight_path):
		self.checkpoint.restore(weight_path)

	def generate(self, input_image):
		output = self.generator(tf.cast(tf.expand_dims(input_image, axis=0), tf.float32), training=True)[0]
		return output.numpy()
