from __future__ import division
import ismrmrd
import ismrmrd.xsd
import numpy as np
from  skimage.measure import label, regionprops
from gadgetron import Gadget
import copy 
import math
import cv2
import time
import os, sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import tensorflow as tf
import scipy as sp
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import binary_fill_holes
from segmentation_settings import segmentation_options


def normalise_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] -- from Wenjia's code """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2 [image < val_l] = val_l
    image2 [image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def preprocess_images (sa_image, pixel_size):
    """ preprocess each iimage in the list sa_image -- just resize at present"""
    sz=pixel_size/1.8 
    processed_images=[]
    for i, img in enumerate(sa_image):
        if segmentation_options['flipped_images']:
            img2 = np.flipud(np.fliplr(cv2.resize(img.copy(),(0,0),fx=sz,fy=sz)))
        else:
            img2 = cv2.resize(img.copy(),(0,0),fx=sz,fy=sz)
        processed_images.append(normalise_intensity(img2, (1,segmentation_options['intensity_percentile_to_saturate'])))
    return processed_images

class SegmentationModel(object):
  
  def __init__(self):
    start_time = time.time()
    model_fname = segmentation_options['model_fname']
    f = tf.gfile.GFile(model_fname, "rb")
    graph_str = tf.GraphDef()
    graph_str.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_str, name="var")
    self.session  = tf.Session(graph=graph)
    op_nodes = graph.get_tensor_by_name('var/pred:0')
    self.op_nodes = op_nodes
    end_time = time.time()
    print("Time taken to load = "+str(end_time-start_time)+" secs")


  def segment_image(self, image):
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
    
    pred = self.session.run(self.op_nodes,
               feed_dict={'var/image:0': image, 'var/training:0': False})#1,160,208
    
    pred = np.transpose(pred, axes=(1, 2, 0))#160,208,1
    pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]
    return cv2.transpose(pred[:,:,0])


def segment_images (processed_tensors):
    """ generate a list of processed_images  """
    segmentation_model = SegmentationModel()
    segmentation_masks = []
    for i in range(0,len(processed_tensors)):
    	segmentation_masks.append(segmentation_model.segment_image(processed_tensors[i]))
    return segmentation_masks


def multiply_array_if_not_empty (arr, multiplier): # must be an easier way!
    return arr * multiplier if len(arr)>0 else []

def subtract_array_if_not_empty (arr, subtractor): # must be an easier way!
    return subtractor-arr if len(arr)>0 else []

def flood_fill(test_array,h_max=255):
    return binary_fill_holes(test_array).astype(int)


def filter_masks_by_size_and_solidity (M1, min_area = 48, min_solidity=0.9):
    """ pick the largest region, make sure it's pigger than min_area (pixels) with solidity more than solidity"""
    #to do: make area relative to area (ie * pixel_spacing)
        
    M2=M1*0  
    if np.sum(M1)>0: #make sure it contains something
        #step 1. get rid of any 'holes'
        M1=flood_fill(M1)
    
        label_img, n_labels = label (M1, connectivity=2, return_num=True)
        mx=-1; maxix=0;
        for i in range(1, n_labels+1):
            area = np.sum(label_img==i)
            if area>mx:
                mx=area; maxix=i;

        M2 = label_img==maxix
        if mx<=min_area:
            M2=M1*0
            #M2 = flood_fill (M2)
        else:        
            # step 3: use a solidity index of 0.9 as crude measure of poor segmentations
            if min_solidity>0: 
                props = regionprops(M2.astype('uint8'))
                if props[0].solidity<min_solidity:   
                    M2=M1*0
        
    return M2


def postprocess_epicardial_masks (mask_epicardium):
    """ takes an epicardial mask (ie 1 for each pixel inside epicardium; 0 outside """
    mask_epicardium = filter_masks_by_size_and_solidity (mask_epicardium, 64, 0.9)
    return mask_epicardium

def postprocess_masks_RV (mask_epicardium, mask_myocardium, mask_RV):
    mask_RV = filter_masks_by_size_and_solidity(mask_RV, 32, 0.)
    return mask_RV
    
def postprocess_masks_endocardium (mask_epicardium, mask_myocardium, mask_endocardium):
    #step 1. derive endocardium (ie the blood pool) by taking the myocardium away from the epicardial mask
    mask_endocardium_derived = np.logical_xor (mask_epicardium, mask_myocardium)
    
    mask_endocardium = flood_fill(mask_endocardium)
    mask_endocardium = np.logical_or(mask_endocardium_derived, mask_endocardium)
    
    #step 2. ensure no holes in blood pool
    mask_endocardium = filter_masks_by_size_and_solidity (mask_endocardium, 32, 0.9)
    
    #step 3. ensure that >90% of blood pool is inside area of epicardium
    n_endo_in_epi = np.sum(mask_endocardium & mask_epicardium)
    n_total_endo = np.sum(mask_endocardium)
    
    if ((n_endo_in_epi+1)/(n_total_endo+1))<=0.9:
        mask_endocardium=mask_endocardium*0;
    
    is_open_contour=False # todo
        
    return mask_endocardium, is_open_contour


def smooth_contours (contour_x, contour_y, n_components=None, circularise=True):
    """ takes contour_x,contour_y the cartesian coordinates of a contour, then procdues a smoothed more circular contour smoothed_contour_x,smoothed_contour_y"""
    #to do: tidy this up
    if n_components is None:
        n_components=12 # slightly arbitary number,  but seems to work well
    
    npts=400+1
    contour_pts = np.transpose(np.stack([contour_x,contour_y]))
    
    if circularise:
        # get the contour points that form a convex hull
        hull = sp.spatial.ConvexHull(contour_pts)
        to_sample = hull.vertices
    else:
        to_sample = range(0,len(contour_x))
    
       
    #wrap around cirlce
    to_sample = np.hstack([to_sample,to_sample[0]])
    sample_pts = contour_pts[to_sample,:]
    
          
             
    # sample each curve at uniform distances according to arc length parameterisation
    dist_between_pts  = np.diff(sample_pts,axis=0)
    cumulative_distance = np.sqrt(dist_between_pts[:,0]**2 + dist_between_pts[:,1]**2)
    cumulative_distance = np.insert(cumulative_distance,0,0,axis=0)
    cumulative_distance = np.cumsum(cumulative_distance)
    cumulative_distance = cumulative_distance/cumulative_distance[-1]
    contour_x=np.interp(np.linspace(0,1,npts),cumulative_distance,sample_pts[:,0],period=360)
    contour_y=np.interp(np.linspace(0,1,npts),cumulative_distance,sample_pts[:,1],period=360)
    contour_x = contour_x[:-1]
    contour_y = contour_y[:-1]

        
    # smooth out contour by keeping the lowest nkeep Fourier components
    n = len (contour_x)
    nfilt=n-n_components-1
    f = np.fft.fft(contour_x)
    f[int(n/2+1-nfilt/2):int(n/2+nfilt/2)] = 0.0;
    smoothed_contour_x = np.abs(np.fft.ifft(f))
    f = np.fft.fft(contour_y)
    f[int(n/2+1-nfilt/2):int(n/2+nfilt/2)] = 0.0;
    smoothed_contour_y = np.abs(np.fft.ifft(f))
    
    return smoothed_contour_x, smoothed_contour_y
   
    
def mask_to_contour (M1, n_components=12, circularise = True):
    
    """ takes a mask, M1, and converts into smoothed/circularised contours """
    if np.sum(M1)>0:
        M2=M1.astype(np.uint8)
        M2=M2/np.max(M2)
        M2=M2.astype('uint8') # this seems overkill, but it's the only way it works

        c = cv2.findContours(M2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours=c[1]
        cc=contours[0];
                       
        xx=np.zeros((len(cc)))
        yy=np.zeros((len(cc)))
        for ix,ccc in enumerate(cc):
            xx[ix]=ccc[0,0]
            yy[ix]=ccc[0,1]
        
        xx,yy=smooth_contours(xx, yy, n_components, circularise)
    else:
        xx = []
        yy = []
        
    return xx, yy


def create_separate_masks (mask):
    """ assumes: backgroun=0; endo (ie LV blood pool)=1;mocardium=2;RV=3"""
    WM = mask
    
    mask_RV = WM==3
    mask_myocardium = WM==2        
    mask_endocardium = WM==1
    mask_epicardium = np.logical_or(mask_myocardium, mask_endocardium)
    
    return mask_epicardium, mask_myocardium, mask_endocardium, mask_RV


def postprocess_masks (mask):
    """ do some morphological operations to tidy up the segmentation masks """
        
    mask_epicardium, mask_myocardium, mask_endocardium, mask_RV = create_separate_masks (mask)
                 
    mask_epicardium = postprocess_epicardial_masks (mask_epicardium)
    mask_endocardium, is_open_contour = postprocess_masks_endocardium (mask_epicardium, mask_myocardium, mask_endocardium)
    mask_RV = postprocess_masks_RV (mask_epicardium, mask_myocardium, mask_RV)
    
    ctr_epi_x, ctr_epi_y = mask_to_contour (mask_epicardium)
    ctr_endo_x, ctr_endo_y = mask_to_contour (mask_endocardium)
    ctr_RV_x, ctr_RV_y = mask_to_contour (mask_RV, 20, False)    
    #to do: stop open contours from 'sticking' out into the LVOT

    #to do: stop RV/LV overlapping etc...
    if segmentation_options['flipped_images']:
        offset_x = mask.shape[1]
        offset_y = mask.shape[0]
        ctr_endo_x = subtract_array_if_not_empty (ctr_endo_x, offset_x)
        ctr_endo_y = subtract_array_if_not_empty (ctr_endo_y, offset_y)
        ctr_epi_x = subtract_array_if_not_empty (ctr_epi_x, offset_x)
        ctr_epi_y = subtract_array_if_not_empty (ctr_epi_y, offset_y)
        ctr_RV_x = subtract_array_if_not_empty (ctr_RV_x, offset_x)
        ctr_RV_y = subtract_array_if_not_empty (ctr_RV_y, offset_y)

    return ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y, ctr_RV_x, ctr_RV_y



def postprocess_all_masks (segmentation_masks, sa_slice_index, sa_phase_index, pixel_size):
    """postprocesses each mask and puts into list of x/y coordinates for epi and endo contours"""
    sz = 1.8/pixel_size
    
    ctr_endo_x_list = []
    ctr_endo_y_list = []
    ctr_epi_x_list = []
    ctr_epi_y_list = []
    ctr_RV_x_list = []
    ctr_RV_y_list = []
    i=0

    for mask in segmentation_masks:
        #print(i);i=i+1
        ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y, ctr_RV_x, ctr_RV_y = postprocess_masks (mask)
        ctr_endo_x_list.append(multiply_array_if_not_empty(ctr_endo_x,sz))
        ctr_endo_y_list.append(multiply_array_if_not_empty(ctr_endo_y,sz))
        ctr_epi_x_list.append(multiply_array_if_not_empty(ctr_epi_x,sz))
        ctr_epi_y_list.append(multiply_array_if_not_empty(ctr_epi_y,sz))
        ctr_RV_x_list.append(multiply_array_if_not_empty(ctr_RV_x,sz))
        ctr_RV_y_list.append(multiply_array_if_not_empty(ctr_RV_y,sz))
    
    return ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list, ctr_RV_y_list 
     
   

def poly_area(x,y):
    """ from """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))




def get_area_per_phase (index_array, mask_list):
    nslices,nphases = index_array.shape    
    ar = np.zeros((nslices,nphases))
    
    for slc in range (0, nslices):
        for phs in range (0, nphases):
            msk = mask_list[index_array[slc,phs]]
            ar[slc, phs] = np.sum(msk)
    phase_area =  np.sum (ar, axis = 0)
    if 0:
        print("area for each phase")
        print(phase_area)
    pa=np.argsort(phase_area)[::-1]
    diastole_ix = pa[0]
    systole_ix = pa[-1]
    return diastole_ix, systole_ix, phase_area


#def get_diastole_and_systole_of_masks(index_array, segmentation_masks):
#    """ get the diastolic (largest volume) and systolic (smallest_volume) index
#        note that this is done on pre-processed images (which may be inaccruate) for the purpose of speed)"""
#    diastole_LV_ix, systole_LV_ix, phase_LV_area = get_area_per_phase (index_array, segmentation_masks==1)   
#    diastole_RV_ix, systole_RV_ix, phase_RV_area = get_area_per_phase (index_array, segmentation_masks==3)
#   
#    ## need to consider the consequences of this properly
#    diastole_RV_ix = diastole_LV_ix 
#    if np.abs(systole_LV_ix - systole_RV_ix)<=1:
#        systole_RV_ix = systole_LV_ix
#    
#    #quick check -- assuming no regirgitatn lesion or shunt...
#    LVSV = phase_LV_area[diastole_LV_ix]-phase_LV_area[systole_LV_ix]
#    RVSV = phase_RV_area[diastole_RV_ix]-phase_RV_area[systole_RV_ix]
#    SVdiff = LVSV-RVSV
#    print(np.abs(SVdiff/LVSV*100))
#    
#    return diastole_LV_ix, systole_LV_ix, diastole_LV_ix, systole_LV_ix


def get_diastoe_and_systole (sa_slice_index, sa_phase_index, ctr_epi_x_list, ctr_epi_y_list):
    """ get the diastolic (largest volume) and systolic (smallest_volume) index"""
    # to do: get RV indices
    slices = np.unique(sa_slice_index)
    phases = np.unique(sa_phase_index)
    nslices = len(slices)
    nphases = len(phases)
    ar = np.zeros((nslices,nphases))
        
    for slc in range(0,nslices):
        this_slice_ix = np.where(sa_slice_index==slc)
        for phs in range (0,nphases):
            sa_index = np.intersect1d(np.where(sa_phase_index==phs),this_slice_ix)[0]
            x = ctr_epi_x_list[sa_index]
            y = ctr_epi_y_list[sa_index]
            ar [slc, phs] = poly_area (x, y)
    phase_ar =  np.sum(ar,axis=0)
    
    print("area for each phase")
    print(phase_ar)
    
    pa=np.argsort(phase_ar)[::-1]
    diastole_ix = pa[0]
    systole_ix = pa[-1]
    return diastole_ix, systole_ix


def pad_number_with_zero (numst, lngth): 
    return '0'+numst if len(numst)==1 else numst
                         

def display_result_as_image (image, ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y, ctr_RV_x, ctr_RV_y, to_draw):
    """ produces an image (suitable for OP to png) where the contours are overlaid on the image -- just for illustration 
        to_draw is one of 'left', 'right' or 'both'"""
    
    out_image = cv2.resize(image.copy(), (0,0), fx=4, fy=4)
    
    #overlay contours (just 'white out' contour pixels)
    if to_draw == 'both' or to_draw == 'left':
        if len(ctr_endo_x)>0:
            mx = np.max(out_image)
        
            xi = (4.*(ctr_endo_x-1.)).astype('int')
            yi = (4.*(ctr_endo_y-1.)).astype('int')
            out_image [yi, xi] = mx
        
            xi = (4.*(ctr_epi_x-1.)).astype('int')
            yi = (4.*(ctr_epi_y-1.)).astype('int')
            out_image [yi, xi] = mx
    
    if to_draw == 'both' or to_draw == 'right':
        if len(ctr_RV_x)>0:
            mx = np.max(out_image)
            xi = (4.*(ctr_RV_x-1.)).astype('int')
            yi = (4.*(ctr_RV_y-1.)).astype('int')
            out_image [yi, xi] = mx
            
    # normalise images for writing to png
    out_image = normalise_intensity(out_image, thres=(1.0, 99.0))
    out_image = (255*out_image).astype('uint8')
    
    return out_image


def write_overlaid_images_to_dir (dirname_out, sa_image, sa_slice_index, sa_phase_index, ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list, ctr_RV_y_list):
    """ writes out results as crude pnds in the directory dirnmae_out"""    
    all_ok = True;
    for i in range(0, len (sa_image)):
        #print(i)
        out_image = display_result_as_image (sa_image[i], ctr_endo_x_list[i], ctr_endo_y_list[i], ctr_epi_x_list[i], ctr_epi_y_list[i], ctr_RV_x_list[i], ctr_RV_y_list[i], 'both')
        
        sli=pad_number_with_zero(str(sa_slice_index[i]),2);
        phi=pad_number_with_zero(str(sa_phase_index[i]),2);       
        
        fname = 'I_sl'+sli+'_ph'+phi+'.png'
        
        retval = cv2.imwrite (dirname_out + fname, out_image)
        all_ok = all_ok & retval
        #print(all_ok)

    return all_ok #True if all saved correctly



def flatten_contours_to_argus_format (x_list, y_list): 
    contour_list=[]; n=len(x_list)
    for i in range(0,n): 
       	contour_list.append(x_list[i])
       	contour_list.append(y_list[i])
    return contour_list


def add_contours_to_headers (all_metas, ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list):
    """ puts the epi- and endo-cardial contours into the header files ('metas') so that they can be sent to the appropriate dicom fields for display in ARGUS"""
    n_images = len(all_metas)
    for i in range (0, n_images):
        endo_list = flatten_contours_to_argus_format (ctr_endo_x_list[i], ctr_endo_y_list[i])
        epi_list = flatten_contours_to_argus_format (ctr_epi_x_list[i], ctr_epi_y_list[i])

        all_metas[i]['ENDO'] = endo_list
        all_metas[i]['EPI'] = epi_list

    return all_metas


def add_contours_to_single_header (meta, ctr_endo_x, ctr_endo_y, ctr_epi_x, ctr_epi_y):
    """ puts the epi- and endo-cardial contours into the header files ('metas') so that they can be sent to the appropriate dicom fields for display in ARGUS"""

    endo_list = flatten_contours_to_argus_format (ctr_endo_x, ctr_endo_y)
    epi_list = flatten_contours_to_argus_format (ctr_epi_x, ctr_epi_y)

    meta['ENDO'] = endo_list
    meta['EPI'] = epi_list

    return meta


def segment_all_images (images, pixel_spacing):
    """ master function to segment all images in the list 'images' """

    print("segmenting "+str(len(images))+" images")
    sa_image =[]
    for img in images:
        image_in = np.squeeze(np.abs(img))
        sa_image.append(image_in)

    print("preprocessing images")
    processed_images = preprocess_images (sa_image, pixel_spacing)


    print("segmenting images")
    segmentation_masks = segment_images (processed_images)

    print("postprocessing images")
    ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list,  ctr_RV_y_list  = postprocess_all_masks (segmentation_masks, [],[], pixel_spacing)

    print("writing images")
    if segmentation_options['write_out_debug_images']: # change this if you want to output some pngs to give an idea of how the segmentation performs
       	write_overlaid_images_to_dir (segmentation_options['dirname_out'], sa_image, range(0,len(images)), range(0,len(images)),ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list,  ctr_RV_y_list)

    return ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list,  ctr_RV_y_list


def segment_single_image (image, pixel_spacing, sa_slice_index, sa_phase_index, segmentation_model):
    """ master function to segment all images in the list 'images' """

    print("segmenting one image at a time")
    image_in = np.squeeze(np.abs(image))
    sa_image =[image_in]

    print("preprocessing image")
    processed_images = preprocess_images (sa_image, pixel_spacing)

    print("segmenting image")
    #segmentation_masks = segment_images (processed_images[0])
    segmentation_mask=[segmentation_model.segment_image(processed_images[0])]

    print("postprocessing images")
    ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list,  ctr_RV_y_list  = postprocess_all_masks (segmentation_mask, [],[], pixel_spacing)

    print("writing images")
    if segmentation_options['write_out_debug_images']: # change this if you want to output some pngs to give an idea of how the segmentation performs
       	write_overlaid_images_to_dir (segmentation_options['dirname_out'], sa_image, [sa_slice_index], [sa_phase_index], ctr_endo_x_list, ctr_endo_y_list, ctr_epi_x_list, ctr_epi_y_list, ctr_RV_x_list,  ctr_RV_y_list)

    return ctr_endo_x_list[0], ctr_endo_y_list[0], ctr_epi_x_list[0], ctr_epi_y_list[0], ctr_RV_x_list[0],  ctr_RV_y_list[0]



