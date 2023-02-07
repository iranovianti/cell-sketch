import tensorflow as tf
import argparse
import os
import datetime
import json

import model
import data_prep

def train_par():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, required=True, help="name of the training method, folder name to store logs and checkpoint")
	parser.add_argument('--datadir', type=str, default='Dataset', help="directory containing train and test dataset (dir/train/*.jpg and dir/test/*.jpg)")
	parser.add_argument('--train_steps', type=int, default=100000, help="training steps, 1 step: 1 image, weight saved every 5k steps")
	parser.add_argument('--a_1', type=float, default=1, help="weight of L1 loss (MAE)")
	parser.add_argument('--a_g', type=float, default=0.01, help="weight of adversarial loss")
	parser.add_argument('--batch_s', type=int, default=1, help="batch size")
	parser.add_argument('--rescale', action='store_true', default=False, help="augmentation: random rescale, range: (1,1.5)")
	parser.add_argument('--contrast', action='store_true', default=False, help="augmentation: random constrast and brightness")
	parser.add_argument('--dropout', action='store_true', default=False, help="dropout for the deeper three upsampling layers of the generator")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	tp = train_par()
	
	batch_size = tp.batch_s
	steps = tp.train_steps

	train_dataset = data_prep.Dataset(tp.datadir, ds_type="train", bs=tp.batch_s, rescale=tp.rescale, contrast=tp.contrast)
	try:
		test_dataset = data_prep.Dataset(tp.datadir, ds_type="val", bs=tp.batch_s)
	except tf.errors.InvalidArgumentError:
		test_dataset = data_prep.Dataset(tp.datadir, ds_type="test", bs=tp.batch_s)

	generator = model.Generator(apply_dropout=tp.dropout)
	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discriminator = model.Discriminator()
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

	checkpoint_dir = "checkpoint/" + tp.name
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									discriminator_optimizer=discriminator_optimizer,
									generator=generator,
									discriminator=discriminator)

	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

	log_dir = "logs/" + tp.name
	summary_writer = tf.summary.create_file_writer(log_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

	def train_step(input_image, target, step):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			gen_output = generator(input_image, training=True)

			disc_real_output = discriminator([input_image, target], training=True)
			disc_generated_output = discriminator([input_image, gen_output], training=True)

			gen_total_loss, gen_gan_loss, gen_l1_loss= model.gen_loss(disc_generated_output, gen_output, target,
																		a_1=tp.a_1, a_gan=tp.a_g)
			disc_loss = model.disc_loss(disc_real_output, disc_generated_output)

		generator_gradients = gen_tape.gradient(gen_total_loss,
												generator.trainable_variables)
		discriminator_gradients = disc_tape.gradient(disc_loss,
													discriminator.trainable_variables)

		generator_optimizer.apply_gradients(zip(generator_gradients,
												generator.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
													discriminator.trainable_variables))

		with summary_writer.as_default():
			tf.summary.scalar("gen_total_loss", gen_total_loss, step=step//1000)
			tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step//1000)
			tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step//1000)
			tf.summary.scalar("disc_loss", disc_loss, step=step//1000)

	with open((tp.name + "_par.txt"), "w") as file:
		json.dump(tp.__dict__, file, indent=2)

	for step, (input_image, target) in train_dataset.repeat().take(steps).enumerate():
		train_step(input_image, target, step)

		if (step + 1) % 5000 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)
