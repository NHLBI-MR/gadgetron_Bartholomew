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
import time

class CineContouring(Gadget):

    def process_config(self, cfg):
        print("Process config of cine contouring ... ")

        try:
            self.header = ismrmrd.xsd.CreateFromDocument(cfg)
        except:
            print('CineContouring, parse xml header failed ... ')
            return 1

        # first encoding space
        self.enc = self.header.encoding[0]

        #Parallel imaging factor
        self.acc_factor = self.enc.parallelImaging.accelerationFactor.kspace_encoding_step_1

        self.slc = self.enc.encodingLimits.slice.maximum+1
        self.phs = self.enc.encodingLimits.phase.maximum+1
        self.phs_retro = int(self.params["phase"])
        self.send_origin = bool(self.params["send_original"])

        print("CineContouring, maximal number of slice ", self.slc)

        n = len(self.header.userParameters.userParameterLong)
        for kk in range(0, n-1):
            ss = self.header.userParameters.userParameterLong[kk].content()
            if ss[0] == 'RetroGatedImages':
                print('Found retro phase from xml protocol', ss[1])
                self.phs_retro = ss[1]

        print("CineContouring, number of retro-gated phases ", self.phs_retro)

        # instansiate tensorflow model to avoid delays in loading parameters later
        try:
            start = time.time()
            self.segmentation_model = SegmentationModel()
            end = time.time()
            print('CineContouring, loading model : ', end-start)
        except:
            print('CineContouring, load model failed ... ')
            return 1

    def process(self, header, image, metadata=None):
        print("Receiving image__+_, phase ", header.phase, ", slice ", header.slice, ', data size', image.shape)

        pixel_spacing = header.field_of_view[0]/header.matrix_size[0] # cheat with a single dimension for now...
        sa_slice_index=header.slice
        sa_phase_index=header.phase

        try:
            start = time.time()
            # row-wise to column-wise
            image_column_wise = np.transpose(image, (1,0,2,3))
            ctr_endo_y, ctr_endo_x, ctr_epi_y, ctr_epi_x, _, _ = segment_single_image (image_column_wise, pixel_spacing, sa_slice_index, sa_phase_index, self.segmentation_model)
            end = time.time()
            print('Cine contouring, segment image : ', end-start)
        except:
            print('CineContouring, segment failed ... ')
            return 1

        # deserialize metadata
        curr_meta = ismrmrd.Meta.deserialize(metadata)

        # attach contours
        updated_meta = add_contours_to_single_header (curr_meta, ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y)

        header_sent = header
        image_sent = image

        # if send original image
        try:
            if self.send_original:
                self.put_next(header,image,curr_meta)
        except:
            print('CineContouring, send original image to downstream failed ... ')
            return 1

        try:
            # send out copy of image with contour
            header_sent.image_series_index += 2050 
            if(updated_meta.has_key('GADGETRON_ImageComment')):
                image_comment = updated_meta['GADGETRON_ImageComment']
                image_comment.append('ENDO_EPI')
                updated_meta['GADGETRON_ImageComment']=image_comment

            self.put_next(header_sent,image_sent,updated_meta)
        except:
            print('CineContouring, send image with contours to downstream failed ... ')
            return 1

        return 0
