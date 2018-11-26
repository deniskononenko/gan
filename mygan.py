#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:35:04 2018

@author: denyskononenko
"""

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, Flatten
from keras.datasets import fashion_mnist
import matplotlib as mplt
from keras.optimizers import Adam
mplt.use('TkAgg')
import matplotlib.pyplot as plt

class GAN():
    def __init__(self):
        # dimensions of dimensions of input image for discriminator, image data (image_height, image_width, channels)
        self.image_height = 28
        self.image_width = 28
        self.channels = 1
        self.image_shape = (self.image_height, self.image_width, self.channels)
        # build generator
        self.generator = self.build_gen()
        # build discriminator
        self.discriminator = self.build_discr()

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discr()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_gen()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        
    def build_gen(self):
        """ Set up generator neural network.  """
        noise_shape = (100,)

        model = Sequential()
        model.add(Dense(256, input_shape=noise_shape, activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(np.prod(self.image_shape), activation="sigmoid"))
        model.add(Reshape(self.image_shape))
        print("\nGenerator parameters:")
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)
    
    def build_discr(self):
        """ Set up discriminator neural network. """
        model = Sequential()

        model.add(Flatten(input_shape=self.image_shape))
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        print("\nDiscriminator parameters:")
        model.summary()

        img = Input(shape=self.image_shape)
        perception = model(img)

        return Model(img, perception)
    
    def train_generator(self, sub_batch_size):
        """ 
        Train Generator Neural Network. 
        During generator training discriminator is static.
        """


        pass

    def train_discr(self, sub_batch_size):
        """
        Train Discriminator Neural Network.
        During discriminator training generator is static.
        """

        pass

    def complex_train(self, epochs, batch_size, save_interval):
        """
        Train full GAN including serial train of
        discriminator and generator.
        """
        # get image training set with number batch_size of images
        (x_train, _), (_, _) = fashion_mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_img(epoch)

    def save_img(self, epoch):
        """Save image in dir at certain epoch."""
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("/Users/denyskononenko/Documents/PythonProjects/gan/fig/image_%d.png" % epoch)
        plt.close()


if __name__ == "__main__":
    """
    print((100,))
    test_gan = GAN()
    print(test_gan.discriminator)
    print(test_gan.generator)

    (x_train, _), (_, _) = fashion_mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=3)
    print(x_train[1])

    # plt.imshow(x_train, cmap="gray")
    # plt.title("Image resolution %d x %d" % (len(x_train[0,1]), len(x_train[1,1])))
    # plt.show()
    testarr = [[[1,1,1],[1,1,1],[1,1,1]], [[1,1,1],[1,1,1],[1,1,1]]]
    print(testarr)
    print(np.expand_dims(testarr, axis=3).shape[0])
    
    """
    gan = GAN()
    print(gan.discriminator)
    print(gan.generator)
    gan.complex_train(epochs=30000, batch_size=32, save_interval=200)