from __future__ import division
import ismrmrd
import ismrmrd.xsd
import numpy as np
from  skimage.measure import label, regionprops
from gadgetron import Gadget
import copy 
import math
import cv2
import os, sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import tensorflow as tf
import scipy as sp
from scipy.ndimage import generate_binary_structure, binary_erosion, grey_erosion
from scipy.spatial import ConvexHull

###############################################################################
#########hard-coded variables -- we could pass these in #######################
###############################################################################
### location of model files
model_fname = os.environ["GADGETRON_HOME"]+"/share/gadgetron/python/FCN_sa" 
###

### whether the images are flipped (updisde-down and left-to-right) from the 'normal' orientation
flipped_images = False
###

### the percentile above which intensity values are saturated -- 99% works well for the images at Barts, but seems to be closer to 95% on the h5 images I used at NIH
intensity_percentile_to_saturate = 95
###

### choose True if segmented images should be written to a local file -- useful for debugging###
write_out_debug_images = False
dirname_out='/tmp'
###
###############################################################################
###############################################################################


def preprocess_images (sa_image, pixel_size):
    """ preprocess each iimage in the list sa_image -- just resize at present"""
    sz=pixel_size/1.8 
    processed_images=[]
    for i, img in enumerate(sa_image):
	if flipped_images:
	    img2 = np.flipud(np.fliplr(cv2.resize(img.copy(),(0,0),fx=sz,fy=sz)))
        else:
            img2 = cv2.resize(img.copy(),(0,0),fx=sz,fy=sz)
        processed_images.append(normalise_intensity(img2, (1,intensity_percentile_to_saturate)))
    
    return processed_images


def normalise_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to range [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2 [image < val_l] = val_l
    image2 [image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2



def segment_images (processed_images, weight_fn):
    """ generate a list of processed_images  """
    with tf.Session() as sess:
        #sess=tf.Session()
        sess.run(tf.global_variables_initializer())
    
        saver = tf.train.import_meta_graph(weight_fn+'.meta')
        saver.restore(sess, weight_fn)
        segmentation_masks = []
        for i in range(0,len(processed_images)):
            image=processed_images[i]#194,146
            image=cv2.transpose(image)#146,194
            #image = normalise_intensity(image, (1,99))
            X, Y = image.shape[:2]#x 146
            if image.ndim == 2:
                image = np.expand_dims(image, axis=2)#146 194 1
            X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16#160,208
            x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
            x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
            image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')#160,208,1
            image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)#1,160,208
            image = np.expand_dims(image, axis=-1)#1,160,208,1
    
            prob, pred = sess.run(['prob:0', 'pred:0'],
                  feed_dict={'image:0': image, 'training:0': False})#1,160,208
    
            pred = np.transpose(pred, axes=(1, 2, 0))#160,208,1
            pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]
            segmentation_masks.append(cv2.transpose(pred[:,:,0]))

    #plt.imshow(image[0,:,:,0]);plt.imshow(pred[0,:,:],alpha=0.5)
    
    return segmentation_masks


def multiply_array_if_not_empty (arr, multiplier): # must be an easier way!
    return arr * multiplier if len(arr)>0 else []

def subtract_array_if_not_empty (arr, subtractor): # must be an easier way!
    return subtractor-arr if len(arr)>0 else []

def flood_fill(test_array,h_max=255):
    """ code from https://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale """
    input_array = np.copy(test_array) 
    el = generate_binary_structure(2,2).astype(np.int)
    inside_mask = binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array


def postprocess_epicardial_masks (M1):
    """ takes an epicardial mask (ie 1 for each pixel inside epicardium; 0 outside """

    #step 1. get rid of any 'holes'
    M1=flood_fill(M1)
    
    #plt.imshow(M1)
    #plt.imshow(np.logical_and(np.logical_not(M1),Mepf))
    
    
    #step 2. keep only the biggest patch and make sure it's >64 pixels
    label_img = label (M1, connectivity=2)
    if np.sum(label_img)==0:
        M2 = M1 * 0
    else:
        props = regionprops(label_img)
        mx=-1;maxix=0;
        for i,prop in enumerate(props):
            if prop.area>mx:
                mx=prop.area;maxix=i;
    
        if mx>64:
            M2 = label_img==maxix+1
            M2 = flood_fill(M2)
        else:
            M2=M1*0
        
        # step 3: use a solidity index of 0.9 as crude measure of poor segmentations
        if props[maxix].solidity<0.90:   
            M2=M1*0
        
    return M2

def postprocess_masks_endocardium (mask_epicardium, mask_myocardium, mask_endocardium):
    
    # step 1. ensure that there are no background pixels inside blood pool
    
    #derive endocardium (ie the blood pool) by taking the myocardium away from the epicardial mask
    mask_endocardium_derived = np.logical_xor (mask_epicardium, mask_myocardium)
    
    mask_endocardium = flood_fill(mask_endocardium)
    mask_endocardium = np.logical_or(mask_endocardium_derived, mask_endocardium)
    
    #step 2. ensure no holes in blood pool
    mask_endocardium=flood_fill(mask_endocardium)
    
    #step 3. keep biggest mask
    label_img = label(mask_endocardium, connectivity=2)
    if np.sum(label_img)==0:
        mask_endocardium=mask_endocardium*0
    else:
        props = regionprops(label_img)
        mx=-1;maxix=0;
        for i,prop in enumerate(props):
            if prop.area>mx:
                mx=prop.area;maxix=i;

        mask_endocardium = label_img==maxix+1
    
    #step 4. ensure that >90% of blood pool is inside area of epicardium
    n_endo_in_epi = np.sum(mask_endocardium & mask_epicardium)
    n_total_endo = np.sum(mask_endocardium)
    
    if (n_endo_in_epi/n_total_endo)<=0.9:
        mask_endocardium=mask_endocardium*0;
    
    return mask_endocardium


def smooth_contours (contour_x, contour_y, tokeep=None):
    """ takes contour_x,contour_y the cartesian coordinates of a contour, then procdues a smoothed more circular contour smoothed_contour_x,smoothed_contour_y"""
    #to do: tidy this up
    if tokeep is None:
        tokeep=12 # slightly arbitary number,  but seems to work well
    
    npts=400
    
    # get the contour points that form a convex hull
    contour_pts = np.transpose(np.stack([contour_x,contour_y]))
    hull = sp.spatial.ConvexHull(contour_pts)
    hv = hull.vertices
    hv = np.hstack([hv,hv[0]])#wrap around
    convex_pts = contour_pts[hv,:]
         
    # sample each curve at uniform distances according to arc length parameterisation
    dist_between_pts  = np.diff(convex_pts,axis=0)
    cumulative_distance = np.sqrt(dist_between_pts[:,0]**2 + dist_between_pts[:,1]**2)
    cumulative_distance = np.insert(cumulative_distance,0,0,axis=0)
    cumulative_distance = np.cumsum(cumulative_distance)
    cumulative_distance = cumulative_distance/cumulative_distance[-1]
    contour_x=np.interp(np.linspace(0,1,npts),cumulative_distance,convex_pts[:,0],period=360)
    contour_y=np.interp(np.linspace(0,1,npts),cumulative_distance,convex_pts[:,1],period=360)
    contour_x = contour_x[:-1]
    contour_y = contour_y[:-1]
    
    # smooth out contour by keeping the lowest nkeep Fourier components
    n = int(len (contour_x)/2)*2
    
    nfilt = int((n-tokeep-1)/2)*2
    f = np.fft.fft(contour_x)
    f[int(n/2+1-nfilt/2):int(n/2+nfilt/2)] = 0.0;
    smoothed_contour_x = np.fft.ifft(f).astype(np.float)
    f = np.fft.fft(contour_y)
    f[int(n/2+1-nfilt/2):int(n/2+nfilt/2)] = 0.0;
    smoothed_contour_y = np.fft.ifft(f).astype(np.float)
     
    return smoothed_contour_x, smoothed_contour_y
   
    
def mask_to_contour (M1):
    """ takes a mask, M1, and converts into smoothed/circularised contours """
    if np.sum(M1)>0:
        M2=M1.astype(np.uint8)
        M2=M2/np.max(M2)
	M2=M2.astype(np.uint8)

        c = cv2.findContours(M2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours=c[1]
        cc=contours[0];
                       
        xx=np.zeros((len(cc)))
        yy=np.zeros((len(cc)))
        for ix,ccc in enumerate(cc):
            xx[ix]=ccc[0,0]
            yy[ix]=ccc[0,1]
        
        xx,yy=smooth_contours(xx,yy)
    else:
        xx = []
        yy = []
        
    return xx, yy



def create_separate_masks (mask):
    """ hard-coded for Wenjia's network """
    WM = mask
    
    # if using Wenjia's code, we'll get rid of the RV for now. Note notation is slightly different with my network
    WM[WM==3]=0 # get rid of RV
    ## change from Wenjia's format to mine
    #bb format: LVBP 1 myo 2 bg 0
    #need 0 LV blood pool, 1 myocardium, 2 background
    bg=WM==0;  WM[WM==1] = 0; WM[bg]=1 #LVBP 0 myo 2 bg 1
    bg=WM==2;  WM[WM==1] = 2; WM[bg]=1 #LVBP 0 myo 1 bg 2
            
    mask_epicardium = WM!=2
    mask_myocardium = WM==1        
    mask_endocardium = WM==0
    
    return mask_epicardium, mask_myocardium, mask_endocardium




def postprocess_masks (mask,pixel_size):
    """ do some morphological operations to tidy up the segmentation masks """

    mask_epicardium, mask_myocardium, mask_endocardium = create_separate_masks (mask)
                 
    mask_epicardium = postprocess_epicardial_masks (mask_epicardium)
    mask_endocardium = postprocess_masks_endocardium (mask_epicardium, mask_myocardium, mask_endocardium)
        
    ctr_epi_x, ctr_epi_y = mask_to_contour (mask_epicardium)
    ctr_endo_x, ctr_endo_y = mask_to_contour (mask_endocardium)
        
    #to do: stop open contours from 'sticking out' into the LVOT

    if flipped_images:
        offset_x = mask.shape[1]
        offset_y = mask.shape[0]
        ctr_endo_x = subtract_array_if_not_empty (ctr_endo_x, offset_x)
        ctr_endo_y = subtract_array_if_not_empty (ctr_endo_y, offset_y)
        ctr_epi_x = subtract_array_if_not_empty (ctr_epi_x, offset_x)
        ctr_epi_y = subtract_array_if_not_empty (ctr_epi_y, offset_y)

    return ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y



def postprocess_all_masks (segmentation_masks, sa_slice_index, sa_phase_index, pixel_size):
    """postprocesses each mask and puts into list of x/y coordinates for epi and endo contours"""
    sz = 1.8/pixel_size
    
    ctr_endo_x_list = []
    ctr_endo_y_list = []
    ctr_epi_x_list = []
    ctr_epi_y_list = []
    i=0
    for mask in segmentation_masks:
        #print(i);i=i+1
        ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y = postprocess_masks (mask,pixel_size)
        ctr_endo_x_list.append(multiply_array_if_not_empty(ctr_endo_x,sz))
        ctr_endo_y_list.append(multiply_array_if_not_empty(ctr_endo_y,sz))
        ctr_epi_x_list.append(multiply_array_if_not_empty(ctr_epi_x,sz))
        ctr_epi_y_list.append(multiply_array_if_not_empty(ctr_epi_y,sz))
    
    return ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list 
     

def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def pad_number_with_zero (numst, lngth): return '0'+numst if len(numst)==1 else numst
                         

def display_result_as_image (image, ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y):
    """ produces an image (suitable for OP to png) where the contours are overlaid on the image -- just for illustration """
    out_image = cv2.resize(image.copy(), (0,0), fx=4, fy=4)
    
    #overlay contours (just 'white out' contour pixels)
    if len(ctr_endo_x)>0:
        mx = np.max(out_image)
        xi = (4.*(ctr_endo_x-1.)).astype('int')
        yi = (4.*(ctr_endo_y-1.)).astype('int')
        out_image [yi, xi] = mx
        xi = (4.*(ctr_epi_x-1.)).astype('int')
        yi = (4.*(ctr_epi_y-1.)).astype('int')
        out_image [yi, xi] = mx
    
    # normalise images for writing to png
    out_image = normalise_intensity(out_image, thres=(1.0, 95.0))
    out_image = (255*out_image).astype('uint8')
    
    return out_image


def write_overlaid_images_to_dir (dirname_out, sa_image, sa_slice_index,sa_phase_index,ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list):
    """ writes out results as crude pnds in the directory dirnmae_out"""    
    all_ok = True;
    for i in range(0, len (sa_image)):
        #print(i)
        out_image = (display_result_as_image (sa_image[i], ctr_endo_x_list[i], ctr_endo_y_list[i], ctr_epi_x_list[i], ctr_epi_y_list[i]))
        
        sli=pad_number_with_zero(str(sa_slice_index[i]),2);
        phi=pad_number_with_zero(str(sa_phase_index[i]),2);       
        
        fname = 'I_sl'+sli+'_ph'+phi+'.png'
        
        retval = cv2.imwrite(dirname_out+fname, out_image)
        all_ok = all_ok & retval
    
    return all_ok #True if all saved correctly


def segment_one_image_at_a_time (image, pixel_spacing, sa_slice_index, sa_phase_index):

    image_in = np.squeeze(np.abs(image))

    sa_image=[image_in]
    processed_image = preprocess_images (sa_image, pixel_spacing)

    segmentation_masks = segment_images (processed_image, model_fname)

    ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list   = postprocess_all_masks (segmentation_masks, [],[], pixel_spacing)

    if write_out_debug_images:
        write_overlaid_images_to_dir (dirname_out, sa_image, [sa_slice_index],[sa_phase_index],ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list)

    return ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list


def segment_all_images (images, pixel_spacing):
    """ mast function to segment all images in the list 'images' """

    print("segmenting "+str(len(images))+" images")
    sa_image =[]
    for img in images:
    	image_in = np.squeeze(np.abs(img))
	sa_image.append(image_in)

    print("preprocessing images")
    processed_images = preprocess_images (sa_image, pixel_spacing)


    print("segmenting images")
    segmentation_masks = segment_images (processed_images, model_fname)

    print("postprocessing images")
    ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list   = postprocess_all_masks (segmentation_masks, [],[], pixel_spacing)

    print("writing images")

    if write_out_debug_images: # change this if you want to output some pngs to give an idea of how 
       	write_overlaid_images_to_dir (dirname_out, sa_image, range(0,len(images)), range(0,len(images)),ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list)

    return ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list


def flatten_contours_to_argus_format (x_list, y_list):
    contour_list=[]; n=len(x_list)
    for i in range(0,n): 
       	contour_list.append(x_list[i])
       	contour_list.append(y_list[i])
    return contour_list


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

	ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list = segment_all_images (self.images, pixel_spacing)

	#write the contours to the meta data to allow export to ARGUS
	all_metas=self.metas
	n_images = len(all_metas)

	for i in range (0, n_images):
	    endo_list = flatten_contours_to_argus_format (ctr_endo_x_list[i], ctr_endo_y_list[i])
	    epi_list = flatten_contours_to_argus_format (ctr_endo_x_list[i], ctr_endo_y_list[i])

	    all_metas[i]['ENDO'] = endo_list
	    all_metas[i]['EPI'] = epi_list

            self.headers[i].image_series_index += 2000 
	    self.put_next(self.headers[i],self.images[i],all_metas[i])

        return 0




