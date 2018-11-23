#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:35:04 2018

@author: denyskononenko
"""

import numpy as np
import tensorflow

class GAN():
    
    def __init__():
        # dimensions of dimensions of input image for discriminator
        # image data (image_height, image_width, channels)
        self.image_height = 32
        self.image_width = 32
        self.channels = 1
        
        self.generator = self.build_gen()
        self.discriminator = self.build_discr()
        


if __name__ == "__main__":
    pass