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
from segmentation_tools import segment_single_image, add_contours_to_single_header, SegmentationModel

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

	# instansiate tensorflow model to avoid delays in loading parameters later
	if len(self.images)==1: 	# hard-code so that it's only called once
	    self.segmentation_model = SegmentationModel()

        ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y, _, _ = segment_single_image (image, pixel_spacing, sa_slice_index, sa_phase_index, self.segmentation_model)
        
	if metadata is not None:
            # deserialize metadata
	    curr_meta = ismrmrd.Meta.deserialize(metadata)
	    updated_meta = add_contours_to_single_header (curr_meta, ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y)
            self.metas.append(updated_meta)

        # if all images are received
        if header.slice<self.slc-1 or header.phase<self.phs_retro-1:
            self.put_next(header,image,curr_meta)
            return 0

        # send out the last image
        self.put_next(header,image,curr_meta)

	# send out copy of image without contour (will switch this to the front when I have more time)
	for i in range(0,len(self.metas)):
            self.headers[i].image_series_index += 2000 
	    this_meta=self.metas[i]
	    this_meta.pop('EPI')
            this_meta.pop('ENDO')
            self.put_next(self.headers[i],self.images[i],this_meta)


        # enough images received
        print("Sufficient images are received ... ")
        print(len(self.headers))

        return 0



