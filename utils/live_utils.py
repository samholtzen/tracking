import nd2
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.measure import regionprops
from scipy.ndimage import grey_dilation, shift
import glob
import pickle
import os
import copy
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture

# Set the logging level to ERROR


def get_mpp_from_nd2(file_to_open):
    
    '''
    Extracts microns per pixel from image unstructured metadata
    '''

    with nd2.ND2File(file_to_open) as file:
        image_sizes = file.sizes
        metadata = file.unstructured_metadata()

    mpp = metadata['ImageCalibrationLV|0']['SLxCalibration']['Calibration']
    return mpp, image_sizes

def get_outdirs(file):
    
    filename_ext = os.path.basename(file)
    filename = os.path.splitext(filename_ext)[0].split('_')[0]
    
    return filename
    
def get_npy(file, metadata=None):
    
    '''
    Extracts images from ND2s and exports them as TIFs.
    '''

    if metadata is None:
        raise FileNotFoundError("Specify experiment metadata.")
        
        
    with nd2.ND2File(file) as f:
        img = f.asarray()

    # handle the case where there is no "sites" axis
    if img.ndim == 4:
        img = img[:,np.newaxis,]
        img = np.moveaxis(img, 2, -1)

    return img

def get_global_offset(metadata):

    image_directory = metadata['output_dir']
    global_offset = []
    folder_to_open = glob.glob(f'{image_directory}/*.nd2')
    registration_channel = metadata['registration_channel']

    for file in tqdm(folder_to_open):

        full_image = get_npy(file, metadata=metadata)

        n_frames = full_image.shape[0]
        n_sites = full_image.shape[1]
        n_channels = full_image.shape[-1]

        curr_offset = []
        reference_frame = full_image[0, 0,...,registration_channel].copy()

        for i in range(0,n_frames):

            next_frame = full_image[i, 0,...,registration_channel].copy()
            offset, error, diffphase = phase_cross_correlation(reference_frame, next_frame, normalization=None)

            curr_offset.append(offset)

        global_offset.append(curr_offset)

    return np.median(np.array(global_offset), axis=0)

def register_image_stack(full_image, metadata=None):
    '''
    Registers each frame of the time lapse and corresponding mask using phase cross correlation.
    No normalization works best for this since these images are "noisy"
    Shifts, stacks images together and saves them to .npy files

    image sizes is the size of each axis of the image, where:
    0: T - time
    1: P - site
    2: X - X coordinate
    3: Y - Y coordinate
    4: C - Channel

    This will change if we do live cell imaging with Z stacks
    '''
    
    n_frames = full_image.shape[0]

    offsets = []

    
    for i in tqdm(range(0,n_frames)):
        
        reference_frame = np.log(np.sum(full_image[0, 0,], axis=-1).copy())
        next_frame = np.log(np.sum(full_image[i, 0,], axis=-1).copy())
        offset, _, _ = phase_cross_correlation(reference_frame, next_frame, normalization=None)
        offsets.append(offset)

    # ensure offsets is a numpy array
    offsets = np.array(offsets)

    registered_image = crop_images_to_overlap(full_image, offsets, (2,3))

    return registered_image

def segment_and_clean(full_image, mpp, nuc_area=300, nuc_channel = None, segment_app=None):

    if segment_app is None:
        raise ValueError("Please include the model you want to use for segmentation.")
        
    '''
    image_directory (str): directory (relative or absolute) to the project folder containing
        your raw images.

    nuc_area (int or float): the area in pixels below which segmented areas are removed. 
        Should be less than 0.5 times the total area of your cells in pixels.

    mpp (float): is microns per pixel from metadata
    '''
            

    n_frames = full_image.shape[0]
    n_sites = full_image.shape[1]

    mask_out = np.zeros((full_image.shape[0],
                         full_image.shape[1],
                         full_image.shape[2],
                         full_image.shape[3],
                         1), dtype=np.int16)
        
    for site in range(n_sites):
        print(f'Segmenting site {site}.')

        x = full_image[:,site,:,:,nuc_channel:nuc_channel+1]
        y_pred = segment_app.predict(x, image_mpp=mpp)
        
        for frame in tqdm(range(n_frames)):
            
            curr_frame = y_pred[frame,]
            
            for props in regionprops(curr_frame):
            
                if props.area < nuc_area:
                    curr_frame[curr_frame == props.label] = 0
                    
            y_pred[frame,] = curr_frame
            
        mask_out[:,site,:,:,0] = y_pred.squeeze()
        
    
    return mask_out

def track_and_pickle(registered_image, registered_mask, nuc_channel=None, tracker=None, dirname=None, filename=None, dilation_size=None):

    if tracker is None:
        raise ValueError("Please specify the model you want to use for tracking.")
    if dirname is None:
        raise ValueError("Please specify the output directory.")

    
    n_sites = registered_mask.shape[1]

    for site in range(n_sites):

        tracked_images = tracker.track(registered_image[:,site,...,nuc_channel:nuc_channel+1],
                                     registered_mask[:,site,])

        tracked_images.pop('X')
        tracked_images['registered_image'] = registered_image[:,site,]
        tracked_images['cyto_mask'] = segment_cytoplasm(tracked_images['y_tracked'], dilation_size=dilation_size).astype('int16')
        tracks = tracked_images.pop('tracks')

        postprocess_tracks = postprocess(tracked_images, tracks)

        pickle_out = f'{dirname}/{filename}_site_{site}_images.pickle'
        
        with open(pickle_out, 'wb') as file:
            pickle.dump(tracked_images, file)

        tracks_pickle = f'{dirname}/{filename}_site_{site}_tracks.pickle'
        
        with open(tracks_pickle, 'wb') as file:
            pickle.dump(postprocess_tracks, file)
    
    return None

def gmm_filter_tracks(track_dict, n_channels=None):
    

    for i in tuple(track_dict.keys()):
        track_dict[i]['channel_exclude'] = np.full((n_channels), 0)

    for channel in range(n_channels):

        int_store = np.array([], ndmin=1)

        for track_idx in tuple(track_dict.keys()):

            if not track_dict[track_idx]['exclude']:

                int_mean = np.mean(track_dict[track_idx]['cell_intensity'][channel,:4])
                int_store = np.hstack((int_store, int_mean))
                
        
        gm = GaussianMixture(n_components=2, random_state=42)
        labels = gm.fit_predict(np.log(int_store.reshape(-1,1)))
        gm_means = gm.means_.squeeze()

        # Flip labels if out of order
        sorted_labels = np.argsort(gm_means)

        if any(np.not_equal(gm_means[sorted_labels], gm_means)):
            labels=np.logical_not(labels)

        for j, track_idx in enumerate(tuple(track_dict.keys())):
            
            if not track_dict[track_idx]['exclude']:
                track_dict[track_idx]['channel_exclude'][channel] = labels[j]


    return track_dict

def segment_cytoplasm(nuc_labels, dilation_size=30):
    '''
    From deepcell-dynamics by eemerson@caltech.edu and elaubsch@caltech.edu
    '''
    cyto_labels = np.zeros(np.shape(nuc_labels))
    for i in range(np.shape(cyto_labels)[0]):
        
        nuc_mask = nuc_labels.copy()[i,...,0] > 0
        wc_labels = grey_dilation(nuc_labels[i,...,0],
                                size=(dilation_size, dilation_size))
        
        wc_mask = wc_labels > 0
        cyto_mask = np.array(nuc_mask.astype(int) + wc_mask.astype(int)) == 1
        cyto_labels[i,...,0] = cyto_mask * wc_labels

    return cyto_labels.astype(int)

def shuffle_colors(ymax, cmap):
    """Utility function to generate a colormap for a labeled image"""
    cmap = mpl.colormaps[cmap].resampled(ymax)
    nmap = cmap(range(ymax))
    np.random.shuffle(nmap)
    cmap = ListedColormap(nmap)
    cmap.set_bad('black')
    return cmap

def plot(x, y, cmap=None, vmax=None):
    yy = copy.deepcopy(y)
    yy = np.ma.masked_equal(yy, 0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(x, cmap='pink', vmax=500)
    ax[0].axis('off')
    ax[0].set_title('Raw')
    ax[1].imshow(yy, cmap=cmap, vmax=vmax)
    ax[1].set_title('Tracked')
    ax[1].axis('off')

    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return image

def crop_images_to_overlap(images, offsets, xy_indices):

    x, y = xy_indices
    image_shape = images.shape
    W = image_shape[x]
    H = image_shape[y]

    offsets = np.array(offsets)

    tl_x = offsets[:, 0]
    tl_y = offsets[:, 1]

    br_x = offsets[:, 0] + W
    br_y = offsets[:, 1] + H

    x_min_overlap = np.max(tl_x)
    y_min_overlap = np.max(tl_y)

    x_max_overlap = np.min(br_x)
    y_max_overlap = np.min(br_y)

    # 3. Calculate the Dimensions of the Overlapping Region
    cropped_width = x_max_overlap - x_min_overlap
    cropped_height = y_max_overlap - y_min_overlap

    cropped_images = []
    for i in range(image_shape[0]):

        img = images[i]
        ox, oy = offsets[i]

        # Calculate top-left crop coordinates relative to the current image's original frame
        crop_x_start = x_min_overlap - ox
        crop_y_start = y_min_overlap - oy

        # Define the cropping box (left, upper, right, lower)
        # Note: Pillow's box is (left, upper, right, lower) where right and lower are exclusive
        crop_box = (int(crop_x_start), int(crop_y_start),
                    int(crop_x_start + cropped_width), int(crop_y_start + cropped_height))

        # Perform the crop
        cropped_img = img[:,crop_box[0]:crop_box[2], crop_box[1]:crop_box[3],:]
        cropped_images.append(cropped_img)

    return np.array(cropped_images)
    
def postprocess(image_data, tracks):


    n_frames = image_data['y_tracked'].shape[0]
    n_channels = image_data['registered_image'].shape[-1]

    for i, track in enumerate(tracks):

        tracks[i+1]['nuc_intensity'] = np.full((n_channels, n_frames), np.nan)
        tracks[i+1]['cyto_intensity'] = np.full((n_channels, n_frames), np.nan)
        tracks[i+1]['cell_intensity'] = np.full((n_channels, n_frames), np.nan)
        tracks[i+1]['exclude'] = False

    for i in range(n_frames):

        cytoring = image_data['cyto_mask'][i].squeeze()
        nuc_mask = image_data['y_tracked'][i].squeeze()
        curr_frame = image_data['registered_image'][i].copy()

        cyto_props = regionprops(cytoring, curr_frame)
        nuc_props = regionprops(nuc_mask, curr_frame)
        cell_props = regionprops(nuc_mask+cytoring, curr_frame)

        for nuc_idx in range(len(nuc_props)):

            cyto_label = cyto_props[nuc_idx].label
            nuc_label = nuc_props[nuc_idx].label

            tracks[cyto_label]['cyto_intensity'][:,i] = cyto_props[nuc_idx].intensity_mean
            tracks[nuc_label]['nuc_intensity'][:,i] = nuc_props[nuc_idx].intensity_mean
            tracks[nuc_label]['cell_intensity'][:,i] = cell_props[nuc_idx].intensity_mean

    # tracks_list = tuple(track_dict.keys())

    # for track_idx in tracks_list:
    #     track_len = len(track_dict[track_idx]['frames'])

    #     # Has to exist for the whole movie
    #     if track_len < n_frames:
    #         track_dict.pop(track_idx)
        
    
    return tracks





