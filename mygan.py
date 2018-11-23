#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:35:04 2018

@author: denyskononenko
"""

import numpy as np
import tensorflow

class GAN():
    
    def __init__(self):
        # dimensions of dimensions of input image for discriminator
        # image data (image_height, image_width, channels)
        self.image_height = 32
        self.image_width = 32
        self.channels = 1
        self.generator = self.build_gen()
        # build generator
        self.discriminator = self.build_discr()
        # build discriminator
        
    def build_gen():
        """ Set up generator neural network. """
        pass
    
    def build_discr():
        """ Set up discriminator neural network. """
        pass
    
    def train_generator():
        """ 
        Train Generator Neural Network. 
        During generator training discriminator is static.
        """
        pass
    
    def train_discriminator():
        """
        Train Discriminator Neural Network 
        During discriminator training generator is static.
        """
        pass
    
    



if __name__ == "__main__":
    pass