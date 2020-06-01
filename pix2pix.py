import tensorflow as tf
import numpy as np
import datetime
import time

class Pix2Pix:
    """TF2.0 version of Pix2Pix model with inspiration from TChollet
    https://github.com/keras-team/keras-io/blob/master/examples/generative/dcgan_overriding_train_step.py
    and https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    OUTPUT_CHANNELS = 3
    LAMBDA = 100
    def __init__(self, img_shape=(256,256,3)):
        self.img_shape = img_shape
        self.discriminator = self.build_discrimiator()
        self.generator = self.build_generator()
        self.log_dir = 'logs/'
        self.writer = tf.summary.create_file_writer(
            self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def build_discrimiator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def build_generator(self):
        inputs = tf.keras.layers.Input(shape=[*self.img_shape])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            self.downsample(128, 4), # (bs, 64, 64, 128)
            self.downsample(256, 4), # (bs, 32, 32, 256)
            self.downsample(512, 4), # (bs, 16, 16, 512)
            self.downsample(512, 4), # (bs, 8, 8, 512)
            self.downsample(512, 4), # (bs, 4, 4, 512)
            self.downsample(512, 4), # (bs, 2, 2, 512)
            self.downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            self.upsample(512, 4), # (bs, 16, 16, 1024)
            self.upsample(256, 4), # (bs, 32, 32, 512)
            self.upsample(128, 4), # (bs, 64, 64, 256)
            self.upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh') # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def compile(self, d_optimizer, g_optimizer, loss_object):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_object = loss_object

    def discrim_loss(self, discrim_real, discrim_generated):
        loss = self.loss_object(
                tf.ones_like(discrim_real), discrim_real)

        gen_loss = self.loss_object(tf.zeros_like(discrim_generated), 
                        discrim_generated)

        return loss + gen_loss

    def gen_loss(self, disc_output, gen_output, target):
        gan_loss = self.loss_object(
                    tf.ones_like(disc_output), disc_output)

        l1_loss = tf.math.reduce_mean(tf.math.abs(target - gen_output))

        total = gan_loss + (self.LAMBDA * l1_loss)

        return total, gan_loss, l1_loss

    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real = self.discriminator([input_image, target], training=True)
            disc_gen = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.gen_loss(disc_gen, gen_output, target)
            disc_loss = self.discrim_loss(disc_real, disc_gen)

        gen_grad = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_grad, self.discriminator.trainable_variables))

        with self.writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    

    def fit(self, train_ds, epochs, size):
        for epoch in range(epochs):
            start = time.time()
            progbar = tf.keras.utils.Progbar(size)
            for n, (input_image, target) in train_ds.enumerate():
                progbar.update(n.numpy()+1)
                self.train_step(input_image, target, epoch)

            tf.print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))