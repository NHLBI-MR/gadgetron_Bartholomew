import os

### location of model files
model_fname = os.environ["GADGETRON_HOME"]+"/share/gadgetron/python/FCN_sa"
###

### whether the images are flipped (updisde-down and left-to-right) from the 'normal' orientation
flipped_images = False
###

### the percentile above which intensity values are saturated -- 99% works well for DICOM images at Barts, but seems to be closer to 95% on the h5 images I used at NIH
intensity_percentile_to_saturate = 95
###

### choose True if segmented images should be written to a local file -- useful for debugging###
write_out_debug_images = False #True
dirname_out = '/tmp/'
###

segmentation_options = {"model_fname": model_fname, "flipped_images": flipped_images, "intensity_percentile_to_saturate": intensity_percentile_to_saturate, "write_out_debug_images":write_out_debug_images, "dirname_out": dirname_out}
