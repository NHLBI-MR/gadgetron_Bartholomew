### location of model files
segmentation_options.model_fname = os.environ["GADGETRON_HOME"]+"/share/gadgetron/python/FCN_sa"
###

### whether the images are flipped (updisde-down and left-to-right) from the 'normal' orientation
segmentation_options.flipped_images = True
###

### the percentile above which intensity values are saturated -- 99% works well for DICOM images at Barts, but seems to be closer to 95% on the h5 images I used at NIH
segmentation_options.intensity_percentile_to_saturate = 95
###

### choose True if segmented images should be written to a local file -- useful for debugging###
segmentation_options.write_out_debug_images = True
segmentation_options.dirname_out = '/tmp/'
###

