from __future__ import division
import ismrmrd
import ismrmrd.xsd
import numpy as np
from gadgetron import Gadget
import copy 
import math
import os, sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import tensorflow as tf
import scipy as sp
from auto_contouring import segment_all_images, put_contours_in_headers

class CineContouring(Gadget):

    def process_config(self, cfg):
        print("Process config of cine contouring ... ")
        self.images = []
        self.headers = []
        self.metas = []

        self.header = ismrmrd.xsd.CreateFromDocument(cfg)

        # first encoding space
        self.enc = self.header.encoding[0]

        #Parallel imaging factor
        self.acc_factor = self.enc.parallelImaging.accelerationFactor.kspace_encoding_step_1

        self.slc = self.enc.encodingLimits.slice.maximum+1
        self.phs = self.enc.encodingLimits.phase.maximum+1
        self.phs_retro = int(self.params["phase"])

        print("CineContouring, maximal number of slice ", self.slc)

        n = len(self.header.userParameters.userParameterLong)
        for kk in range(0, n-1):
            ss = self.header.userParameters.userParameterLong[kk].content()
            if ss[0] == 'RetroGatedImages':
                print('Found retro phase from xml protocol', ss[1])
                self.phs_retro = ss[1]

        print("CineContouring, number of retro-gated phases ", self.phs_retro)
        
    def process(self, header, image, metadata=None):

        print("Receiving image__+_, phase ", header.phase, ", slice ", header.slice)

        # buffer incoming images
        self.images.append(image)
        self.headers.append(header)

	pixel_spacing = header.field_of_view[0]/header.matrix_size[0] # cheat with a single dimension for now...
	sa_slice_index=header.slice
	sa_phase_index=header.phase

        if metadata is not None:
            # deserialize metadata
	    curr_meta = ismrmrd.Meta.deserialize(metadata)
            self.metas.append(curr_meta)


        # if all images are received
        if header.slice<self.slc-1 or header.phase<self.phs_retro-1:
            self.put_next(header,image,curr_meta)
            return 0

        # send out the last image
        self.put_next(header,image,curr_meta)

	# enough images received
        print("Sufficient images are received ... ")
        print(len(self.headers))

	# call segmentation code -- do all at once for now. To do: do each slice individually and call the code as soon as the image 'comes in'
	ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list, ctr_RV_y_list = segment_all_images (self.images, pixel_spacing)

	#write the contours to the meta data to allow export to ARGUS
	self.metas= put_contours_in_headers (self.metas, ctr_endo_x_list, ctr_endo_x_list, ctr_epi_x_list, ctr_epi_x_list)

	for i in range(0,len(self.metas)):
            self.headers[i].image_series_index += 100 
	    self.put_next(self.headers[i],self.images[i],self.metas[i])


        return 0




