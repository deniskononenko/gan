#
#
#   U-net architecture neural network trained on pairs (image, mask)
#   is used for mask generation for distinct image.
#
#   Used scientific paper:  arXiv:1505.04597
#
#   Input image 256x256 bw image (channel = 1)
#

import numpy as np
import datetime
import os
import re
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,  Dense, Reshape, Input, Flatten, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import save_model, load_model
import matplotlib as mplt
mplt.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class UNet():
    def __init__(self):
        self.image_height = 512
        self.image_width = 512
        self.channels = 1
        self.input_shape = (self.channels, self.image_height, self.image_width)
        self.network = self.gen_layers()
        # compile model
        self.network.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    def gen_layers(self):
        """Generates  convolutional neural network's layers (left and right)"""
        model = Sequential()

        # feed forward
        model.add(Conv2D(512, kernel_size=(4, 4), input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(1024, kernel_size=(4, 4), padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(2048, kernel_size=(4, 4), padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((1, 2), padding='same'))

        # feed backward (convolution on top)
        model.add(Conv2D(2048, kernel_size=(4, 4), padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(UpSampling2D((1, 2)))
        model.add(Conv2D(1024, kernel_size=(4, 4), padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(UpSampling2D((1, 2)))
        model.add(Conv2D(512, kernel_size=(4, 4), padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(UpSampling2D((1, 1)))
        model.add(Conv2D(256, kernel_size=(4, 4), padding="same"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(UpSampling2D((1, 1)))

        print("\nParameters of U-net")
        model.summary()

        img = Input(shape=self.input_shape)
        output_image = model(img)

        return Model(img, output_image)

    def get_image_batch(self):
        """
        Get appropriate number of images for training an preprocess them.
        Output an array of preprocessed images.
        Returns array with two elements: [0] - training set, [1] - training forms of images
        """
        # get images
        image_form_arr = []
        train_images = []
        train_images_masks = []

        print("\n %s" % os.getcwd())
        work_dir = os.getcwd()+"/train_imgs/"
        os.chdir(work_dir)
        for files in os.walk(work_dir):
            for file in files:
                if type(file) == list and len(file) !=0:
                    for imgs_items in file:
                        if re.search(r".png", imgs_items) != None and re.search(r"cont", imgs_items) == None and imgs_items != ".png":
                            name = imgs_items.split(".")[0]
                            print(imgs_items)
                            img = mpimg.imread(imgs_items, 0)
                            img_cont = mpimg.imread(name+"cont.png", 0)
                            image_form_arr.append([imgs_items, img, img_cont])
                            train_images.append(img)
                            train_images_masks.append(img_cont)
        return [train_images, train_images_masks]

    @staticmethod
    def rename(imgs):
        """Renames files with imgs names in current directory"""
        print(imgs)
        new_name = str(imgs.split(".")[0]) + ".png"
        os.rename(imgs, new_name)

    def train(self, batch_size, epochs, train_images, train_image_masks):
        """Train model. Make fit of train_images with train_images_masks. Returns history model."""
        trained_model = self.network.fit(train_images, train_image_masks, batch_size=batch_size, epochs=epochs)
        trained_model.save("/Users/denyskononenko/Documents/PythonProjects/gan/trained_models/trained_model.h5py")
        #return trained_model


    def generate_mask(self, image):
        """Generates image with help of trained model"""
        pass


if __name__  == "__main__":
    print("\nstart time: " + str(datetime.datetime.today()))
    unet = UNet()
    images_forms_array = unet.get_image_batch()
    #im_train = (images_forms_array[0] - 127.5) / 127.5
    im_nd_arr = np.ndarray([])
    #print(im_nd_arr)
    for i in range(len(images_forms_array)):
        np.append(im_nd_arr, images_forms_array[0][i])
    # print(im_nd_arr)
    print(images_forms_array[0][0].shape)
    print(images_forms_array[1][0].shape)
    print(images_forms_array[1][0])
    #print(np.concatenate(images_forms_array[0][0], images_forms_array[0][1]))
    #unet.train(2, 20, images_forms_array[0], images_forms_array[1])


    (x_train, _), (_, _) = fashion_mnist.load_data()
    #print(x_train)
    #x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    print(x_train)
