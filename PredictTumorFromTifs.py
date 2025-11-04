# internal imports 
import os
import numpy as np
import subprocess

# external imports
from skimage import io
from scipy import misc # to load bmp
import nibabel as nb
import glob
import cv2
from PIL import Image
import pandas as pd

def get_voxel_size(path):
    # grab the log file
    log_file = glob.glob(os.path.join(path,'*.log'))
    
    if len(log_file) != 1:
        return 'Log File Not Found'
    else:
        infile = log_file[0]
        keep_phrases = ['Image Pixel Size']
        important = []
        with open(infile) as f:
            f = f.readlines()
        
        for line in f:
            for phrase in keep_phrases:
                if phrase in line:
                    important.append(line)
    
        if len(important) == 0:
            return 'Pixel Size Not Found'
        else:
            return important

def reorient(image):
    image = np.flip(image,axis=2)
    image = np.rot90(image,2)
    image = np.flip(image,axis=1)
    
    return image

def convert2HU(image):

    rescale_min = -1000
    rescale_range = 2200

    # Because these images have intensities from 
    current_range = np.max(image.flatten())

    # Scale all values to be withing a 3200 range
    rescale_factor = rescale_range/current_range
    return (image * rescale_factor) + rescale_min
    
def squarecrop(im):
    squared = im.crop((0,0,min(im.size),min(im.size)))
    return squared

def pad256(image):
    # this take 3d array and pads up to 256x256x256
    current_shape = image.shape
    pad = []
    for i in [0,1,2]:
        if current_shape[i]%2 == 1:
            pad.append([int((256-current_shape[i]-1)/2),int((256-current_shape[i]+1)/2)])
        else:
            pad.append([int((256-current_shape[i])/2),int((256-current_shape[i])/2)])
    return np.pad(image,((pad[0][0],pad[0][1]),(pad[1][0],pad[1][1]),(pad[2][0],pad[2][1])))

def downsample(image,scaling_factor):

    # preallocate first 
    holder_xy = np.zeros((int(scaling_factor*image.shape[0]),int(scaling_factor*image.shape[1]), image.shape[2]))

    # Reshape x,y
    for z in range(holder_xy.shape[2]):
        holder_xy[:,:,z] = cv2.resize(image[:,:,z],(holder_xy.shape[1],holder_xy.shape[0]),interpolation=cv2.INTER_CUBIC)
        #holder_xy[:,:,z] = cv2.GaussianBlur(holder_xy[:,:,z],(5,5),0)
    holder_z = np.zeros((holder_xy.shape[0], holder_xy.shape[1],int(scaling_factor*image.shape[2])))

    #reshape z
    for x in range(holder_z.shape[0]):
        holder_z[x,:,:] =cv2.resize(holder_xy[x,:,:],(holder_z.shape[2],holder_z.shape[1]),interpolation=cv2.INTER_CUBIC)
    
    return holder_z

def preprocess_microCT(filepath, filt='*rec0000*.*'):
    
    print('starting '+filepath)
    new_name = 'img_000_0000.nii.gz'
    output_folder = filepath+'-Nifti/img'
    new_name = os.path.join(output_folder,new_name)
    
    to_glob = os.path.join(filepath,filt)

    # Grab all the files
    filenames = glob.glob(to_glob)
    print(f'{len(filenames)} files found')
    
    # if < 2 filenames (likely 0) then data wasn't reconstructed
    if len(filenames) < 2:
        return []
        
    # Make the ouptut directory
    if not os.path.exists(filepath+'-Nifti'):
        os.mkdir(filepath+'-Nifti')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Sort the filenames so image is reconstructed in order
    filenames.sort()
    filenames.sort(key=len)

    # Check to make sure this sample image isn't 512x512x800 (ie a full tif stack) 
    initial_size = io.imread(filenames[0])
    if len(initial_size.shape) > 2:
        if initial_size.shape[2] > 3:
            return []

    #try:
    # Preallocate the room 
    im = np.zeros((initial_size.shape[0],initial_size.shape[1],len(filenames)))

    # Collect all the slices into a full stack (either .bmp so we use Image.open or .tif so we use io.imread)
    for i,filename in enumerate(filenames): # iterate over the mouse scans
        if filename.split('.')[-1] == 'bmp':
            im[:,:,i] = np.array(Image.open(filename))
        else:
            try:
                im[:,:,i] = np.array(io.imread(filename))
            except:
                im[:,:,i] = im[:,:,i-1]
    
    # Calculate how much you need to downsample
    image_size = [im.shape[0],im.shape[1],im.shape[2]]
    largest_dim = np.max(im.shape)
    resample_rate = 256/largest_dim
    voxel_size_log = get_voxel_size(filepath)
    
    if 'Not Found' not in voxel_size_log:
         log_size = float(voxel_size_log[-1].split('=')[-1])
         voxel_size_log.append(f'Nifti Pixel Scaling : {largest_dim/256}')
         voxel_size_log.append(f'Nifti Pixel Size : {log_size * largest_dim/256}')
         newlog='\n'.join(listitem for listitem in voxel_size_log)
         
         with open(os.path.join(output_folder,"Voxel_Spacing.txt"), "w") as text_file:
         	text_file.write(newlog)
		
    # Downsample so the max length of any dimension is 256
    dwnsmpled_img = downsample(im,resample_rate)
    
    # Pad out to 256x256x256
    new_image = pad256(dwnsmpled_img)
    
    # Reorient and convert to houndsfield units
    new_image_HU = reorient(convert2HU(new_image))

    if 'supine' in filepath:
        new_image_HU = np.flip(new_image_HU, axis=0)
        new_image_HU = np.flip(new_image_HU, axis=1)
        
    # Needed for nii header
    affine = np.eye(4)
    affine[0][0] = -1
    
    # Turn numpy array into nii image
    img_nii = nb.Nifti1Image(new_image_HU,affine)
        
    # Saving the file
    print('saving '+new_name)
    try:
        nb.save(img_nii,new_name)
    except:
        print(f'{filename} failed to build')

def predict_microCT(filepath, model_location):

	#variables that have to be exported
	training_data_raw = '/mnt/bmc-lab6-archive/nnUnet_LauraMTrainingData'
	training_data_processed = '/mnt/bmc-lab6-archive/nnUnet_LauraMTrainingData/data_preprocessed'	

	if not os.path.exists(os.path.join(tif_folder+'-Nifti','prediction')):
	    os.mkdir(os.path.join(tif_folder+'-Nifti','prediction'))

	# Set up environment
	os.environ['nnUNet_results'] = model_location
	os.environ['nnUNet_raw'] = training_data_raw
	os.environ['nnUNet_preprocessed'] = training_data_processed

	subprocess.check_call(['nnUNetv2_predict', '-i', os.path.join(tif_folder+'-Nifti','img'), '-o',
		os.path.join(tif_folder+'-Nifti','prediction'), '-c','3d_fullres','-f', '0', '-d', '1'], 
		env=os.environ)


def create_and_save_histogram(lung_vol, tumor_vol, data, bins, tif_folder):
    """
    Creates a histogram from data and saves it to a file without plotting.

    Args:
        data (array_like): The input data for the histogram.
        bins (int or sequence of scalars): The number of bins or bin edges.
        output_file (str): The path to the output file.
    """

    
    hist, bin_edges = np.histogram(data, bins=bins)
    
    total_volume = len(data)
    aerated_volume = np.sum(data <= -600)
    percent_aerated = 100 * aerated_volume/total_volume
    
    output_file = tif_folder +'-Nifti/prediction/LungHist.txt'
    # Save the histogram data to a text file
    with open(output_file, "w") as f:
        f.write("Bin Edges: ")
        f.write(str(bin_edges.tolist()))
        f.write("\n")
        f.write("Histogram Counts: ")
        f.write(str(hist.tolist()))
        f.write("\n")
        f.write(f'Percent Aerated Volume: {percent_aerated:.2f}%')
        f.write("\n")
        f.write(f'Lung Vol : {lung_vol} voxels')
        f.write("\n")
        f.write(f'Tumor Vol : {tumor_vol} voxels')
        f.write("\n")
        f.write(f'Mean Intensity : {np.mean(data):.2f}')
        f.write("\n")
        f.write(f'Median Intensity : {np.median(data):.2f}')
	
def write_results(tif_folder):
	
	try:
		# Load img and segmentation
		img = nb.load(tif_folder + '-Nifti/img/img_000_0000.nii.gz').get_fdata()
		seg = nb.load(tif_folder + '-Nifti/prediction/img_000.nii.gz').get_fdata()
		lungs = img[seg > 0]
		lung_vol = np.sum(seg>0)
		tumor_vol = np.sum(seg == 2)
		create_and_save_histogram(lung_vol, tumor_vol, lungs.flatten(), 50, tif_folder)
	except:
		print('Nifti not found')
		
	
	

# Point to file
tif_folders = glob.glob('')
model_location = ''

for tif_folder in tif_folders:
	# Preprocess file
	preprocess_microCT(tif_folder, '*_rec00*.tif*')
	    
	# Predict
	predict_microCT(tif_folder,model_location)
	
	# Write Quick Results
	write_results(tif_folder)
	
