import tensorflow as tf

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

def Generator(apply_dropout=False):
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
		upsample(512, 4, apply_dropout=apply_dropout),
		upsample(512, 4, apply_dropout=apply_dropout),
		upsample(512, 4, apply_dropout=apply_dropout),
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

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss(disc_generated_output, gen_output, target, a_gan, a_1):
	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

	total_gen_loss = (a_gan * gan_loss) + (a_1 * l1_loss)

	return total_gen_loss, gan_loss, l1_loss

def disc_loss(disc_real_output, disc_generated_output):
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss