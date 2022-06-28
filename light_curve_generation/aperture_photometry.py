# Import all the toys
!pip install photutils
!pip install plottify
!pip install lightkurve
!pip install matplotlib --upgrade

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import and define plotting style
import seaborn as sb
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
sb.set_context("talk")

from lightkurve import LightCurve
from cmath import sqrt
from pandas import DataFrame, read_csv, concat
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder
from photutils.utils import calc_total_error
from astropy.table import join
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.modeling import models, fitting, functional_models
from astropy.utils.exceptions import AstropyWarning
from plottify import autosize
from google.colab import drive
from warnings import simplefilter, filterwarnings
from os import path, mkdir, rename
drive.mount('/content/drive')
simplefilter('ignore', category = AstropyWarning)
filterwarnings('ignore')
%matplotlib inline

def filesorter(filename, dataframedir, foldername):

    '''
    PURPOSE: 
            Checks if the directory (dir + foldername + filename) exists, then creates the directory if it doesn't.

    INPUTS:
                [filename; string]:  Name of the file
            [dataframedir; string]:  Directory where you want things to be saved (this is the same as the directory that holds the dataframe)
              [foldername; string]:  Name of the folder you want to check/generate

    RETURNS:
            Creates the specified folders and or directories

    AUTHOR:
            Julio M. Morales (adapted from Connor E. Robinson), November 04, 2021
    '''

    # The statement checks to see if the file exists.
    if path.exists(''.join([dataframedir,filename])):
        pass
    else:
        print(''.join([dataframedir, filename, ' does not exist or has already been moved.']))
        return

    # If the foldername (input) doesn't exist, then it creates a new directory with the name foldername.
    if path.exists(''.join([dataframedir, foldername])):
        pass
    else:
        print(''.join(["Making new directory: ", dataframedir, foldername]))
        mkdir(''.join([dataframedir, foldername]))

    # Move files to new folder
    print(''.join(['Moving ', filename, ' to:  ', dataframedir, foldername, '/', filename]))
    rename(''.join([dataframedir, filename]), ''.join([dataframedir, foldername, '/', filename]))
    
    return

def pad_to_square(image, pad_value=0):

    '''
    PURPOSE:
            Pad the image to maintain a squre shape to the array.  This is required to perform modeling and photometry on the central
            star and ghost dynamically.
            
    INPUTS:
            [image; np.array, float]:  FITS image to be padded.

    OPT.
    INPUTS:
            [pad_value; integer]:  Value you wish the image to be padded by (default is 0).

    RETURNS:
            [padded_image; np.array, float]:  Padded version of the FITS image.
        
    AUTHOR:
            Julio M. Morales March 1st, 2022
    '''

    # Reshape the array
    m = image.reshape((image.shape[0], -1))

    # Pad the image
    padded = pad_value * np.ones(2 * [max(m.shape)], dtype = m.dtype)

    # Reassign the values to the array
    padded[0:m.shape[0], 0:m.shape[1]] = m

    return padded

def fwhm_extractor(image, sat, i, print_psfs=False):

    '''
    PURPOSE:
            This function uses a 2-dimensional Moffat to model the PSF of the central star in a FITS image. 
            In the case of unsaturated data, it extracts FWHM from the central star, and in the case of saturated
            data, it subtracts the central star PSF from the data so that it can accurately model the PSF of the 
            ghost for FWHM extraction.

    INPUTS:
            [image; np.array, float]:  FITS image.
                       [sat; string]:  ('Y' or 'N') whether the dataset is saturated or not.
                            [i; int]:  Integer of the given image in the dataset sequence.

    OPT.
    INPUTS:
            [print_psfs; boolean]:  If True, prints out the density plots of the data, model, and residual PSFs (default is False)


    RETURNS:
            [f2.fwhm, f2.amplitude.value; list, float]:  Returns the fit values for the FWHM and the peak pixel.

    AUTHOR:
            Julio M. Morales, January 20, 2022    
    '''

    # Define dimensions of the cut around the PSFs
    width  = 31
    stmpsz = int((width - 1)/2)
    y2, x2 = np.mgrid[:width, :width]
    size = image.shape
    xcen, ycen = int(size[0]/2), int(size[1]/2)

    # True if the image is unsaturated
    if (sat == 'N'):

        plot_string = ''

    # True if the image is SATURATED
    else:
        
        # Define parameters used to cut around PSF
        xcen, ycen = int(xcen + 1), int(ycen + 5)
        plot_string = ' GHOST'

    # Cut out smaller box around PSF
    zoomed_image = image[ycen - stmpsz - 1: ycen + stmpsz, xcen - stmpsz - 1: xcen + stmpsz]

    # Declare what function you want to fit to your data with
    f_init2 = models.Moffat2D(np.nanmax(zoomed_image), stmpsz, stmpsz, 6, 1)

    # Declare what fitting method
    fit_f2  = fitting.LevMarLSQFitter()

    # Fit the model to your data
    f2 = fit_f2(f_init2, x2, y2, zoomed_image)

    # Subtract the model from the data
    residual_image = zoomed_image - f2(x2, y2)

    # Estimate an uncertainity on the peak pixel value
    peak_err = np.nanmedian(residual_image)
    
    # True if you want to print out the density plots
    if (print_psfs == True):

        # Dummy fwhm value
        fw = f2.fwhm
        
        # Checks if the FWHM is nonphysical
        if (np.isnan(fw) == True) or (fw <= 0) or (fw > 31) or (fw <= 2):

            # Set FWHM to NaN so we cannot use it
            fw = np.nan

        # Select proper label format for density plot legends
        if (np.isnan(fw) == True):
            label = 'FWHM Fit = NaN '
        else:
            label = f'FWHM Fit = {fw:.2f}'

        # Plot the ghost psf with the central star psf subtracted, its model, and its residual
        fig, axs = plt.subplots(1, 3, figsize=(25, 20))
        interval = ZScaleInterval()                     
        vmin, vmax = interval.get_limits(zoomed_image)
        im0 = axs[0].imshow(zoomed_image, vmin=vmin, vmax=vmax)
        axs[0].set_title(''.join(['(Image ', str(i), ')', plot_string, ' Data']), size=20)
        axs[0].text(22.5, 1, label, bbox={'facecolor': 'white', 'pad': 10})

        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0, ax=axs[0])

        im1 = axs[1].imshow(f2(x2, y2), vmin=vmin, vmax=vmax)
        axs[1].set_title(''.join(['(Image ', str(i), ')', plot_string, ' Model']), size=20)
        axs[1].text(22.5, 1, label, bbox={'facecolor': 'white', 'pad': 10})

        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size = "5%", pad=0.05)
        plt.colorbar(im1, cax=cax1, ax=axs[1])

        vmin2, vmax2 = interval.get_limits(residual_image)
        im2 = axs[2].imshow(residual_image, vmin=vmin2, vmax=vmax2)
        axs[2].set_title(''.join(['(Image ', str(i), ')', plot_string, ' Residuals']), size=20)

        divider2 = make_axes_locatable(axs[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax2, ax=axs[2])
        # plt.savefig('/content/drive/My Drive/Julios Thesis Codes/Data/source_psf_image_' + str(i) + '.jpg', dpi = 400, bbox_inches = 'tight', rasterize = False)
        # filesorter('source_psf_image_' + str(i) + '.jpg', '/content/drive/My Drive/Julios Thesis Codes/Data/', 'movies')
    
    return [f2.fwhm, f2.amplitude.value, peak_err]

def psf_model(image, sat, xcen, ycen, box_dim, i, print_psfs=False):

    '''
    PURPOSE:
            This function uses a 2-dimensional Moffat to model the PSF of the central star in a FITS image. 
            In the case of unsaturated data, it extracts FWHM from the central star, but in the case of saturated
            data, it subtracts the central star PSF from the data so that it can accurately model the PSF of the 
            ghost for the FWHM extraction.

    INPUTS:
            [image; np.array, float]:  Image to be modeled
                       [sat; string]:  ('Y' or 'N') whether the dataset is saturated or not
                         [xcen; int]:  x-centroid of the source
                         [ycen; int]:  y-centroid of the source
                      [box_dim; int]:  Size of the square reigon to be cut around the ghost
                            [i; int]:  Integer of the image in the sequence

    OPT.
    INPUTS:
               [print_psfs; boolean]:  If True, prints out the density plots of the data, model, and residual psfs (default is False)

    RETURNS:
            [psf_sub_image or sliced_image; np.array, float]:  FITS image around the ghost with the central star PSF subtracted or the image sliced around the central star.
                                               [peak; float]:  Peak Moffat fit pixel value.
                                               [fwhm; float]:  FWHM value extracted from the 2D Moffat model.

    AUTHOR:
            Julio M. Morales, January 10, 2022    
    '''

    # Define dimensions of the cut around the PSFs
    stmpsz = int((box_dim - 1)/2)
    size = image.shape
    xcen = int(size[0]/2) + 2
    ycen = xcen

    # True if the image is saturated
    if sat == 'Y':
        
        # Redefine the centroids needed to cut around the ghost
        xcen, ycen = int(xcen + 158.5) - 1, int(ycen - 7)

    # Convert all values in the image to floats (we need to do this to replace the pixels)
    float_image = image.astype("float")

    # This nested for loop replaces all of the pixels over the ghost with 0's
    for xpixels in range(0, float_image.shape[1]):

        for ypixels in range(0, float_image.shape[0]):

            # Define the range of pixels we want to be replaced with 0's                
            if (float_image.shape[1] - 60 >= xpixels >= 370) and (205 <= ypixels <= float_image.shape[1] - 225):
                float_image[ypixels, xpixels] = 0

    # Cut out smaller box around PSF and pad to maintain square array
    sliced_image = float_image[ycen - stmpsz - 1: ycen + stmpsz, xcen - stmpsz - 1: xcen + stmpsz]
    sliced_image_pad = pad_to_square(sliced_image, pad_value=0)
    yp, xp = sliced_image_pad.shape

    # Generate grid of same size like box to put the fit on
    y, x, = np.mgrid[:yp, :xp]

    # Declare what function you want to fit to your data to
    f_init = models.Gaussian2D()

    # Declare what fitting method 
    fit_f  = fitting.LevMarLSQFitter()
    
    # Fit the model to your data (box)
    f = fit_f(f_init, x, y, sliced_image_pad)

    # Define string to be used in plot title
    plot_string = 'Central Star'

    # True if the image is unsaturated
    if (sat == 'N'):

        # Calculate the parameters of the PSF model
        model = fwhm_extractor(sliced_image, sat, i, print_psfs)

    # True if the image is SATURATED
    else:

        # Subtract the model from the data and pad to maintain square array
        psf_sub_image = image[ycen - stmpsz - 1: ycen + stmpsz, xcen - stmpsz - 1: xcen + stmpsz]
        psf_sub_image_pad = pad_to_square(psf_sub_image, pad_value = 0) - f(x, y)

        # Calculate the parameters of the PSF model
        model = fwhm_extractor(psf_sub_image, sat, i, print_psfs)
    
    # Extract model fwhm and peak
    fwhm, peak, peak_err = model[0], model[1], model[2]
    
    # Return the image cut around the central source
    if (sat == 'N'):

        return [sliced_image_pad, fwhm, peak, peak_err]

    # True if the image is SATURATED
    else:

        # True if we want to show the density plots of the ghost data before subtracting the central psf,
        # the ghost model and the residuals
        if (print_psfs == True):

            # Plot the data with the best-fit model
            fig, axs = plt.subplots(1, 3, figsize=(25, 20))
            interval = ZScaleInterval()                     
            vmin, vmax = interval.get_limits(sliced_image_pad)
            im0 = axs[0].imshow(sliced_image_pad, vmin=vmin, vmax=vmax)
            axs[0].set_title(''.join([plot_string, ' Data']), size=20)

            divider0 = make_axes_locatable(axs[0])
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im0, cax=cax0, ax=axs[0])

            im1 = axs[1].imshow(f(x, y), vmin=vmin, vmax=vmax)
            axs[1].set_title(''.join([plot_string, ' Model']), size=20)

            divider1 = make_axes_locatable(axs[1])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax1, ax=axs[1])

            vmin2, vmax2 = interval.get_limits(psf_sub_image_pad)
            im2 = axs[2].imshow(psf_sub_image_pad, vmin=vmin2, vmax=vmax2)
            axs[2].set_title(''.join([plot_string, ' Residuals']), size=20)

            divider2 = make_axes_locatable(axs[2])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im2, cax=cax2, ax=axs[2])
            # plt.savefig('/content/drive/My Drive/Julios Thesis Codes/Data/central_psf_image_' + str(i) + '.jpg', dpi = 400, bbox_inches = 'tight', rasterize = False)
            # filesorter('central_psf_image_' + str(i) + '.jpg', '/content/drive/My Drive/Julios Thesis Codes/Data/', 'movies')

        return [psf_sub_image_pad, fwhm, peak, peak_err]

def bg_error_estimate(image, star_pos, box_dim, gain, sigma=4.0):

    '''
    PURPOSE:
            This function uses a sigma clipping algorthim to mask the brightests sources
            in the image.  It then stores this image as an error image to be used in the
            photometry extraction.

    INPUTS:
            [image; np.array, float]:  Image whose background is to be estimated
             [star_pos; list, float]:  x and y-centroid of the source
                      [box_dim; int]:  Size of the square reigon to be cut around the ghost
                         [gain; int]:  Gain of the detector.
    OPT. 
    INPUTS:
            [sigma; float]:  Threshold of the detections (default is 4.0)

    RETURNS:
            [error_image; np.array, float]:  Sigma clipped version of the image

    AUTHOR:
            Julio M. Morales, December 01, 2021
    '''
    
    # Define dimensions of the cut around the PSFs
    stmpsz = int((box_dim - 1)/2)
    xcen, ycen = star_pos[0][0], star_pos[0][1]

    # Slice the image around the source
    sliced_image = image[ycen - stmpsz - 1: ycen + stmpsz, xcen - stmpsz - 1: xcen + stmpsz]
    sliced_image_pad = pad_to_square(sliced_image, pad_value=0)

    # Remove pixels from the data that are above sigma times the standard deviation of the background
    filtered_data  = sigma_clip(sliced_image_pad, sigma=sigma, copy=False)

    # Take the filtered values and fill them with NaN's
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)

    # Find the variance of all the remaining values (the values that have not been replaced with NaN's)
    bkg_error = np.sqrt(bkg_values_nan)

    # Find the median of the variance (found in the previous line)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)
    
    # Calculate the error image and write to a new file
    error_image = calc_total_error(sliced_image_pad, bkg_error, gain)  
    
    return error_image

def photometry_extractor(image, error_image, sat, star_pos, model_fwhm, units, gain, read_noise, ghost_scale, un_gs, print_values=False):

    '''
    PURPOSE:
            This function calculates the aperture photometry using the FWHM measurements from the PSF modelling function.
            It uses the DAOStarFinder to locate the centroid of the star (or ghost) and calculates aperture and sky annuli
            according to r_aperture = 1.5*FWHM, r_in = r_aperture + 5, r_out = r_in + 10.  It also calculates the overall 
            uncertainity in the measurement

    INPUTS:
                  [image; np.array, float]:  FITS image.
            [error_image; np.array, float]:  Sigma clipped version of the image.
                             [sat; string]:  ('Y' or 'N') Saturation status of the image.
                   [star_pos; list, float]:  x and y-centroid of the source.
                       [model_fwhm; float]:  FWHM value extracted from the 2D Moffat model.
                           [units; string]:  Units that the dataset was taken in.
                               [gain; int]:  Gain of the detector (for the MAG AO system, this is 500 electrons/ADU).
                       [read_noise; float]:  Readnoise of the instrument used for observation.
                      [ghost_scale; float]:  Scale used to bring a ghost measurement to central star values.
                            [un_gs; float]:  Uncertainy in the ghost scale.

    OPT.
    INPUTS:
           [print_values; boolean]:  If True, prints out the image numbers, fitted FWHMs, and flux measurements (default is False).

    RETURNS:
               [return_sum; float]:  Final photometry measurement.
                    [noise; float]:  Error on the photometry measurement.
  
    AUTHOR:
            Julio M. Morales, November 23, 2021
    '''

    # Manually set centroids (in case DAO Fails)
    xpos_man = np.array([image.shape[0]/2])
    ypos_man = xpos_man
    star_pos_man = np.array([[image.shape[0]/2, image.shape[0]/2]])
    
    # Compute MAD of the background
    bkg_mad = DataFrame(np.hstack(error_image)).mad(skipna=True)[0]
        
    # Scan image for centroid
    daofind = DAOStarFinder(fwhm=model_fwhm, threshold=3*bkg_mad, brightest=1, xycoords=star_pos_man)
    sources = daofind(image)

    # True if the source isn't found by DAO
    if (sources == None):
        xpos0, ypos0 = xpos_man, ypos_man
    else:
        xpos0, ypos0 = np.array(sources['xcentroid']), np.array(sources['ycentroid'])

    # Pair up the positions as a tupple and stores them as a list
    star_pos0 = list(zip(xpos0, ypos0))

    # -------------------------------------------------------------------------------------- #
    # YOU CAN UNCOMMENT THE FOLLOWING CODE IF YOU WISH TO ENSURE THE STARFINDER IS DETECTING #
    # THE CENTRAL STAR                                                                       #
    # -------------------------------------------------------------------------------------- #

    # if (sat == 'N'):
    #     # Define the file that contains the circled source
    #     f = open('/content/drive/My Drive/Julios Thesis Codes/Data/central_star_green_cricles.reg', 'w') 
    #     for i in range(0, len(xpos0)):
    #         # Write the circled reigons to the file
    #         f.write('circle ' + str(xpos0[i]) + ' ' + str(ypos0[i]) + ' ' + str(1.5*model_fwhm) + '\n')
    #     # Close the file
    #     f.close()
    # else:         
    #     # Define the file that contains the circled source
    #     f = open('/content/drive/My Drive/Julios Thesis Codes/Data/ghost_green_cricles.reg', 'w') 
    #     for i in range(0, len(xpos0)):
    #         # Write the circled reigons to the file
    #         f.write('circle ' + str(xpos0[i]) + ' ' + str(ypos0[i]) + ' ' + str(1.5*model_fwhm) + '\n')
    #     # Close the file
    #     f.close()

    # Define the aperture and sky radii
    if (sat == 'N'):
        r_aperture = 19*model_fwhm
    else:
        r_aperture = 1.5*model_fwhm
    
    skyin  = r_aperture + 5
    skyout = skyin + 10

    # Create circular apertures and annuli with the specifed radii
    starapertures = CircularAperture(star_pos0, r=r_aperture)
    skyannuli  = CircularAnnulus (star_pos0, r_in=skyin, r_out=skyout)
    phot_apers = [starapertures, skyannuli]

    # Table containing photometry
    phot_table = aperture_photometry(image, phot_apers, error=error_image, method='subpixel', subpixels=50)
      
    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean  = phot_table['aperture_sum_1']/skyannuli.area
    bkg_starap_sum = bkg_mean*starapertures.area
    final_sum = (phot_table['aperture_sum_0'] - bkg_starap_sum)
    
    # Calculate the error on the photometry
    bkg_mean_err = phot_table['aperture_sum_err_1']/skyannuli.area
    bkg_sum_err  = bkg_mean_err*starapertures.area

    # True if used ghost
    if (sat == 'Y'):
        return_sum = final_sum
        noise = sqrt((phot_table['aperture_sum_err_0'])**2 + (bkg_sum_err)**2 + float(final_sum) + (read_noise/gain)**2).real
    else:
        return_sum = final_sum
        noise = sqrt((phot_table['aperture_sum_err_0'])**2 + (bkg_sum_err)**2 + float(final_sum) + (read_noise/gain)**2).real

    # True if the flux is unphysical
    if (np.isnan(return_sum) == True) or (return_sum <= 0):

        # Set flux and noise equal to NaN so we cannot use it
        return_sum, noise = np.nan, np.nan

        # True if you want to print the values
        if (print_values == True):
            print(''.join(['Flux = NaN +/- NaN \n']))

    # True if the flux is physical
    else:
        
        # True if you want to print the values
        if (print_values == True):
            if (np.isnan(noise) == True):
                print(''.join(['Flux = ', f'{int(float(return_sum)):,.0f}', ' +/- NaN \n']))
            else:
                print(''.join(['Flux = ', f'{int(float(return_sum)):,.0f}', ' +/- ', f'{int(float(noise)):,.0f} ', units, '\n']))

    return [float(return_sum), float(noise)]

def Aperture_Photometry(dataframedir, dataframename, timestampsdir, object_name=None, date=None, print_psfs=False, print_values=False):

    '''
    PURPOSE: 
            This function extracts photometry from every image in a datacube for both the line and continumn seqeuences in the 
            GAPlanetS Database via aperture photometry. It does so by first, loading in the images one-by-one and calculateing the
            centroid of the central star in the case of unsaturated data.  It then models the PSF of the central star and checks
            to see if the peak pixel value exceeds that of the VisAO saturation limit.  If it does not, the code extracts photo-
            metric measurments from the central star.  If it does exceed the saturation limit, the code goes through the exact
            same process to extract photometry from the ghost instead.  All photometric measurments are extracted via aperture
            photometry, and all aperture sizes are calculated by using the PSF models of either the central star or the ghost 
            with a 2D Moffatt. All of this data and uncertainties (as well as some plotting color conventions) are saved in the 
            form of a NPY dictionary.  This function can automatically extract photometry for all datasets at once, or one-by-one 
            if an object name and date is specified. The user can also specifiy whether or not they want to print the PSFs of each 
            image and its model, and if they want to print the individual FWHM and flux measurements.

    INPUTS:
             [dataframedir; string]:  Directory that cotains the dataframe needed to extract naming convention info
            [dataframename; string]:  Name of .csv dataframe that resides in [dataframedir]
            [timestampsdir; string]:  Directory where the timestamps dictionary resides

    OPT.
    INPUTS:
              [object_name; string]:  Name of the object in the database  (default is None)
                     [date; string]:  Date of the dataset in the database (default is None)
              [print_psfs; boolean]:  If True, prints out the density plots of the data, model, and residual psfs (default is False)
            [print_values; boolean]:  If True, prints out the image numbers, fitted FWHMs, and flux measurements  (default is False)

    RETURNS:  
            [dictionary, np.array, float]:  NPY file containing the dictionary with the photometry data.  This will be saved in the dataframedir.

    AUTHOR:
            Julio M. Morales, November 04, 2021
    '''
  
    # Define empty dictionary to append the photometry data to
    dictionary = {}

    # Load in dataframe
    dataframe  = read_csv(dataframedir + dataframename)

    # Load in the time data from the dictionary
    time_data  = np.load(''.join([timestampsdir, 'GAPlanetS_Survey_Timestamps.npy']), allow_pickle = 'TRUE').item()

    # Extract all of the relevant data from the CSV file
    object_list = dataframe['Object']
    date_list = dataframe['Date']
    dir_list  = dataframe['Directory']
    sat_list  = dataframe['Saturated?']
    imstring_list = dataframe['Preprocessed File Name']
    units_list = dataframe['Units']
    gain_list  = dataframe['Gain']
    ron_list = dataframe['RON']

    # Store all this info as a tupple
    tupple = list(zip(object_list, date_list, dir_list, sat_list, imstring_list, units_list, gain_list, ron_list))
    
    # True if the object name was not specified and the date wasn't
    if (object_name != None) and (date == None):

        print('ERROR:  You have to pick a date! \n')
        date = input('Write a date for the dataset (as it is listed in the database):  ')
        print('\n')
        print('- - ' * 25)
        print('\n')

    # True if the object name was not specified but the date was
    if (object_name == None) and (date != None):

        print('ERROR:  You have to pick an object name! \n')
        object_name = input('Write the name of the object for the dataset (as it is listed in the database):  ')
        print('\n')
        print('- - ' * 25)
        print('\n')

    # True if user has input a specific object
    if (object_name != None) and (date != None):

        # Define index and test name and date to extract the correct data
        i = 0

        name_test = dataframe['Object'][i]
        date_test = dataframe['Date'][i]

        # Select the correct info based on input name and date
        while (name_test != object_name) or (date_test != date):

            # Pull the all the relevant info from the tupple
            i += 1
            name_test = dataframe['Object'][i]
            date_test = dataframe['Date'][i]

        else:
            dir = ''.join([dataframe['Directory'][i], object_name, '/', date, '/'])
            sat = dataframe['Saturated?'][i]
            def_sat  = dataframe['Saturated?'][i]
            imstring = dataframe['Preprocessed File Name'][i]
            units = dataframe['Units'][i]
            gain  = dataframe['Gain'][i]
            ron = dataframe['RON'][i]
            one_dataset = True

    # True if the user has not specified a dataset
    if (object_name == None) and (date == None):

        one_dataset = False

    # Loop over all of the datasets
    for dataset in tupple:

        # True if no object or date was specified
        if (one_dataset == False):
            
            # Pull the all the relevant info from the tupple
            object_name = dataset[0]
            date = dataset[1]
            dir  = ''.join([dataset[2], object_name, '/', date, '/'])
            sat  = dataset[3]
            def_sat  = dataset[3]
            imstring = dataset[4]
            units = dataset[5]
            gain  = dataset[6]
            ron = dataset[7]

        # Pull the temporal data and the NaN mask from the dictionary for each specfic dataset
        time_def = time_data[''.join([object_name, '_', date])]['time (minutes since midnight)']
        time = [(i - np.nanmin(time_def)) for i in time_def]
        nan_mask = time_data[''.join([object_name, '_', date])]['nan mask']

        # Wavelengths of the images
        wavelength = ['Line', 'Cont']

        # Confirm the units and saturation limit
        if (units == 'ADU'):
            u, sat_limit = units, 16_000

        # Set saturation limit in the case of electrons
        if (units == 'electrons'):
            u, gain, sat_limit = 'e', 1, 50_000

        # Define the lists of values that we'll append to
        line_flux = []
        line_flux_err = []
        line_fwhm = []
        line_peak = []
        line_peak_err = []
        line_r = []
        ratio_colors = []

        cont_flux = []
        cont_flux_err = []
        cont_fwhm = []
        cont_peak = []
        cont_peak_err = []
        cont_r = []

        # Define the area to be sliced out of the image for the central star and ghost
        star_ycen,  star_xcen,  star_box_dim  = 225, 225, 300
        ghost_ycen, ghost_xcen, ghost_box_dim = 219, 383, 160

        # Manually set the positions of the star and ghost (this is for the error image, not the photometry)
        star_xpos,  star_ypos  = np.array([151]), np.array([151])
        ghost_xpos, ghost_ypos = np.array([81]),  np.array([80])

        # Pair up the positions as a tupple and stores them as a list
        star_pos  = list(zip(star_xpos,  star_ypos))
        ghost_pos = list(zip(ghost_xpos, ghost_ypos))

        # Print the name of the dataset being ran through
        print(''.join(['Extracting Photometry for ', object_name, ' ', date, ': \n']))
        
        # Iterate over each wavelength
        for wave in wavelength:
            
            # State ghost scale and diffraction limit for both wavelengths
            if (wave == 'Line'):
                ghost_scale, un_gs = 224.80, 0.98
                fwhm_diffraction_limit = 3.17
            if (wave == 'Cont'):
                ghost_scale, un_gs = 237.98, 0.84
                fwhm_diffraction_limit = 3.11

            # Define maximum FWHM
            max_fwhm = 10

            # Load in the image cube
            image = fits.getdata(''.join([dir, 'preprocessed/', wave, imstring, '.fits'])) 
            n = len(image)

            # Estimate the background error image and slice it around the source
            star_error_image  = bg_error_estimate(image[0], star_pos,  star_box_dim,  gain)
            ghost_error_image = bg_error_estimate(image[0], ghost_pos, ghost_box_dim, gain)

            # Compute MAD of the background
            star_bkg_mad  = DataFrame(np.hstack(star_error_image)).mad(skipna = True)[0]
            ghost_bkg_mad = DataFrame(np.hstack(ghost_error_image)).mad(skipna = True)[0]
                        
            # Extract the files that contain the integer locations of the accepted images
            rejection_integers = fits.getdata(''.join([dir, 'calibration/', wave, 'cosmics.fits']))

            # Iterate over all of the images
            for i in range(0, n):

                # True if you want to print the values
                if (print_values == True):
                    print(''.join([wave, ' Image: ', f'{i:,.0f}', '/', f'{n:,.0f}', '\n']))

                # True if the image is unsaturated
                if (sat == 'N'):
                    xcen, ycen, pos, box_dim, error_image, bkg_mad = star_xcen,  star_ycen,  star_pos,  star_box_dim,  star_error_image,  star_bkg_mad
                if (sat == 'Y'):
                    xcen, ycen, pos, box_dim, error_image, bkg_mad = ghost_xcen, ghost_ycen, ghost_pos, ghost_box_dim, ghost_error_image, ghost_bkg_mad

                # Slice the image around the central source or ghost and model the psf to extract a FWHM and peak
                psf = psf_model(image[i], sat, xcen, ycen, box_dim, i, print_psfs)
                sliced_image, model_fwhm, model_peak, peak_err = psf[0], psf[1], psf[2], psf[3]
                
                # True if the peak pixel exceeds the saturation limit
                if (model_peak > sat_limit):

                    model_peak, model_fwhm = np.nan, np.nan

                # Checks if the FWHM is nonphysical
                if (np.isnan(model_fwhm) == True) or (model_fwhm > max_fwhm) or (model_fwhm < fwhm_diffraction_limit):

                    # Set FWHM and lux to NaN
                    model_fwhm, flux, noise = np.nan, np.nan, np.nan
                    
                    # True if you want to print the values
                    if (print_values == True):
                        print('FWHM = NaN \n')
                        print(''.join(['Flux = NaN +/- NaN  \n']))
                        print('- - ' * 25)
                        print('\n')
                else:
                                
                    # True if you want to print the values
                    if (print_values == True):
                        print(f'FWHM = {model_fwhm:.2f} pix \n')

                    # Extract photometry
                    phot = photometry_extractor(sliced_image, error_image, sat, pos, model_fwhm, u, gain, ron, ghost_scale, un_gs, print_values)
                    flux, noise = phot[0], phot[1]
                    
                    # True if you want to print the values
                    if (print_values == True):
                        print('- - ' * 25)
                        print('\n')

                # True if the wavelength is Line
                if (wave == 'Line'):
                    line_flux.append(flux)
                    line_flux_err.append(noise)
                    line_r.append(model_fwhm)
                    line_peak.append(model_peak)
                    line_peak_err.append(peak_err)
                
                # True if the wavelength is Cont
                else:
                    cont_flux.append(flux)
                    cont_flux_err.append(noise)
                    cont_r.append(model_fwhm)
                    cont_peak.append(model_peak)
                    cont_peak_err.append(peak_err)

            # Loop through the integers of the timestamps
            for k in range(0, len(time)):

                # True if indices in rejection_integers are missing
                if k not in rejection_integers:

                    # True if working with Line images
                    if (wave == 'Line'):

                        # Place NaN in location of missing integers
                        line_flux.insert(k, np.nan)
                        line_flux_err.insert(k, np.nan)
                        line_fwhm.insert(k, np.nan)
                        line_r.insert(k, np.nan)
                        line_peak.insert(k, np.nan)
                        line_peak_err.insert(k, np.nan)

                    # True if working with Cont images
                    else:

                        # Place NaN in location of missing integers
                        cont_flux.insert(k, np.nan)
                        cont_flux_err.insert(k, np.nan)
                        cont_fwhm.insert(k, np.nan)
                        cont_r.insert(k, np.nan)
                        cont_peak.insert(k, np.nan)
                        cont_peak_err.insert(k, np.nan)

        # Convert fluxes to numpy array's
        line_flux = np.array(line_flux)
        cont_flux = np.array(cont_flux)

        # Make adjusted NaN mask from the fluxes
        new_mask = (line_flux/cont_flux)/(line_flux/cont_flux)

        line_flux_err = np.array(line_flux_err)*new_mask
        cont_flux_err = np.array(cont_flux_err)*new_mask

        # Take the ratio of the fluxes
        ratio = line_flux/cont_flux
        ratio_err = ratio*((cont_flux_err/cont_flux) + (line_flux_err/line_flux))

        # Pull the specific info from the dictionary
        line_peak = np.array(line_peak)*new_mask
        line_peak_err = np.array(line_peak_err)*new_mask
        line_r = np.array(line_r)*new_mask

        cont_peak = np.array(cont_peak)*new_mask
        cont_peak_err = np.array(cont_peak_err)*new_mask
        cont_r = np.array(cont_r)*new_mask

        # Ratio of the peak pixels
        peak_ratio = line_peak/cont_peak
        peak_ratio_err = peak_ratio*((cont_peak_err/cont_peak) + (line_peak_err/line_peak))

        # Write FWHM and peak lists to new fits files
        if (sat == 'N'):
            fits.writeto(dir + 'dq_cuts_J/Linefwhmlist_J.fits',  line_r, overwrite = True)
            fits.writeto(dir + 'dq_cuts_J/Linestarpeaks_J.fits', line_peak, overwrite = True)
            fits.writeto(dir + 'dq_cuts_J/Contfwhmlist_J.fits',  cont_r, overwrite = True)
            fits.writeto(dir + 'dq_cuts_J/Contstarpeaks_J.fits', cont_peak, overwrite = True)
        else:
            fits.writeto(dir + 'dq_cuts_J/Lineghostfwhmlist_J.fits', line_r, overwrite = True)
            fits.writeto(dir + 'dq_cuts_J/Lineghostpeaks_J.fits', line_peak, overwrite = True)
            fits.writeto(dir + 'dq_cuts_J/Contghostfwhmlist_J.fits', cont_r, overwrite = True)
            fits.writeto(dir + 'dq_cuts_J/Contghostpeaks_J.fits', cont_peak, overwrite = True)

        ###############################################################
        # BEGINNING OF SECTION TO REMOVING OUTLIERS IN THE FINAL DATA #
        ###############################################################

        # Compile a list of all the data to be plotted and stored
        dummy_data = [time*new_mask, line_flux*new_mask, line_flux_err*new_mask, cont_flux*new_mask, cont_flux_err*new_mask, 
                      line_r, cont_r, line_peak, line_peak_err, cont_peak, cont_peak_err, peak_ratio, peak_ratio_err, ratio, ratio_err]

        # Remove datapoints that are above 2*median
        clean_ratio = [np.nan if i > 2*np.nanmedian(ratio) else i for i in ratio]

        # Create mask for the cleaned data
        mask = np.array(clean_ratio)/np.array(clean_ratio)

        # Empty list for cleaned data storage
        clean_data = []

        # Apply mask to all the new data
        for variable in dummy_data:
            clean_variable = variable*mask
            clean_data.append(clean_variable)
        
        ###############################################################
        #             END OF OUTLIER REMOVAL SECTION                  #
        ###############################################################

        # List of datasets to be plotted with new mask
        data = [line_r*mask, cont_r*mask, line_peak*mask, cont_peak*mask, line_flux*mask, cont_flux*mask, peak_ratio*mask, clean_ratio]

        # Calculate the mean, median, and std of the data
        median = np.nanmedian(clean_data[13])
        std = np.nanstd(clean_data[13])
        peak_median = np.nanmedian(clean_data[12])
        peak_std = np.nanstd(clean_data[12])

        # Write data to a dictionary
        inner_dict = {               'time (minutes)':  clean_data[0], 
                                       'H-alpha Flux':  clean_data[1],
                                 'H-alpha Flux Error':  clean_data[2],
                                         'Cont. Flux':  clean_data[3], 
                                   'Cont. Flux Error':  clean_data[4],
                              'H-alpha FWHM (pixels)':  clean_data[5], 
                                'Cont. FWHM (pixels)':  clean_data[6], 
                                  'H-alpha Starpeaks':  clean_data[7],
                            'H-alpha Starpeaks Error':  clean_data[8], 
                                    'Cont. Starpeaks':  clean_data[9],
                              'Cont. Starpeaks Error':  clean_data[10],
                                    'Starpeaks Ratio':  clean_data[11], 
                              'Starpeaks Ratio Error':  clean_data[12],
                                'H-alpha-Cont. Ratio':  clean_data[13], 
                          'H-alpha-Cont. Ratio Error':  clean_data[14]}

        # Define outer dictionary to store all of the indivual datasets
        data_dictionary = {''.join([object_name, '_', date]): inner_dict}

        # Define the colors and labels for plots
        colors = ['co', 'co', 'ro', 'ro', 'bo', 'bo', 'mo', 'mo']
        color_scatter = ['cyan', 'cyan', 'red', 'red', 'blue', 'blue', 'magenta', 'magenta']
 
        ylabels = [ r'Line FWHM (pixels)', 
                    r'Cont. FWHM (pixels)',  
                    r'H$\alpha$ Peak Pixel (' + u + ')', 
                    r'Cont. Peak Pixel (' + u + ')', 
                    r'H$\alpha$ (' + u + ')', 
                    r'Cont. (' + u + ')',
                    r'Peak H$\alpha$/Cont.', 
                    r'H$\alpha$/Cont.']

        # List of strings for plotting labels
        plot_label = [ r'Moffat FWHM for H$\alpha$ Images', 
                        'Moffat FWHM for Cont. Images',
                       r'Peak Pixel Values in H$\alpha$ Images',
                        'Peak Pixel Values in Cont. Images',
                       r'Where H$\alpha$ is measured at 656 nm',
                       r'Where Cont. is measured at 643 nm',
                       r'Peak H$\alpha$/Cont. Ratio',
                       r'H$\alpha$/Cont. Ratio']

        # True if MAG1 Seeing data exists
        if 'MAG Seeing' in list(time_data[''.join([object_name, '_', date])].keys()):

            # True if the MAG1 and DIMM seeing data exists
            if 'DIMM Seeing' in list(time_data[''.join([object_name, '_', date])].keys()):

                # Store all the data in a list and replace any nonphysical seeing values with NaN
                mag  = time_data[''.join([object_name, '_', date])]['MAG Seeing']  * new_mask
                mag  = [np.nan if i == -1 else i for i in mag]

                dimm = time_data[''.join([object_name, '_', date])]['DIMM Seeing'] * new_mask
                dimm = [np.nan if i == -1 else i for i in dimm]

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in dimm) != True):
                    data.insert(0, dimm)
                    colors.insert(0, 'go')
                    color_scatter.insert(0, 'green')
                    ylabels.insert(0, 'DIMM Seeing (arcseconds)')
                    plot_label.insert(0, 'DIMM Instrument')
                    inner_dict['DIMM Instrument'] = dimm

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in mag) != True):
                    data.insert(0, mag)
                    colors.insert(0, 'go')
                    color_scatter.insert(0, 'green')
                    ylabels.insert(0, 'MAG1 Seeing (arcseconds)')
                    plot_label.insert(0, 'MAG1 Instrument')
                    inner_dict['MAG1 Seeing (arcseconds)'] = mag

            # True if the MAG1 seeing data ecists and the DIMM seeing data does NOT exist
            else:

                # Store all the data in a list
                mag = time_data[''.join([object_name, '_', date])]['MAG Seeing'] * new_mask
                mag = [np.nan if i == -1 else i for i in mag]

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in mag) != True):
                    data.insert(0, mag)
                    colors.insert(0, 'go')
                    color_scatter.insert(0, 'green')
                    ylabels.insert(0, 'MAG1 Seeing (arcseconds)')
                    plot_label.insert(0, 'MAG1 Instrument')
                    inner_dict['MAG1 Seeing (arcseconds)'] = mag
                
        # True if the MAG1 seeing data does NOT exist
        else:

            # True if the MAG1 seeing does NOT exist but the DIMM seeing data does exist
            if 'DIMM Seeing' in list(time_data[''.join([object_name, '_', date])].keys()):

                # Store all the data in a list
                dimm = time_data[''.join([object_name, '_', date])]['DIMM Seeing']*new_mask
                dimm = [np.nan if i == -1 else i for i in dimm]

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in dimm) != True):
                    data.insert(0, dimm)
                    colors.insert(0, 'go')
                    color_scatter.insert(0, 'green')
                    ylabels.insert(0, 'DIMM Seeing (arcseconds)')
                    plot_label.insert(0, 'DIMM Instrument')
                    inner_dict['DIMM Instrument'] = dimm

        # Set the title of the plot for saturated and nonsaturated cases
        if def_sat == 'Y':
            plot_title = ''.join([object_name, ' ', date, ' Ghost'])
            scl = 0.8
        else:
            plot_title = ''.join([object_name, ' ', date])
            scl = 0.03

        # Set plot font size
        my_size = 22
        plt.rc('font', size = my_size)

        # Plot the Photometry Ratio
        plt.figure(figsize = (40, 10))
        plt.errorbar(clean_data[0], clean_data[13], clean_data[14], fmt = 'mo', markersize = 3, capsize = 5, label = r'Where H$\alpha$ and the Cont. are measured at 656 nm and 643 nm respectively')
        plt.hlines(median, xmin = np.nanmin(clean_data[0]), xmax = np.nanmax(clean_data[0]), colors = 'blue', linestyles = '--', linewidth = 2, label = f'Median Ratio = {median:.2f} +/- {std:.2f}')

        # Set the scale of the figure based on saturation status
        if (sat == 'Y'):
            plt.ylim(0, 2*median)
        else:
            plt.ylim(0.8*median, 1.2*median)      

        plt.xlabel('time (minutes)')
        plt.ylabel('H-alpha-Cont. Ratio')
        plt.title(plot_title)
        plt.legend(loc = 'best')
        plt.tight_layout()

        # Plot the peak ratio
        plt.figure(figsize = (40, 10))
        plt.errorbar(clean_data[0], clean_data[11], clean_data[12], fmt = 'mo', markersize = 3, capsize = 5, label = r'Where H$\alpha$ and the Cont. are measured at 656 nm and 643 nm respectively')
        plt.hlines(peak_median, xmin = np.nanmin(clean_data[0]), xmax = np.nanmax(clean_data[0]), colors = 'blue', linestyles = '--', linewidth = 2, label = f'Median Ratio = {peak_median:.2f} +/- {peak_std:.2f}')

        # Set the scale of the figure based on saturation status
        if (sat == 'Y'):
            plt.ylim(0, 2*peak_median)
        else:
            plt.ylim(0.8*peak_median, 1.2*peak_median)
        
        plt.xlabel('time (minutes)')
        plt.ylabel('H-alpha-Cont. Peak Ratio')
        plt.title(plot_title)
        plt.legend(loc = 'best')
        plt.tight_layout()

        # Define the figure for all of the values to be plotted on
        fig, axs = plt.subplots(len(data), sharex = True, sharey = False, figsize = (55, 35))
        fig.suptitle(plot_title)
        fig.subplots_adjust(hspace = 0)

        # Loop through all the data we have for this specific dataset
        for i in range(0, len(data)):

            # Make the plots    
            axs[i].plot(np.nan, np.nan, colors[i], label = plot_label[i])
            axs[i].plot(time, data[i], colors[i], markersize = 4)
            axs[i].set_ylabel(ylabels[i])
            axs[i].legend(loc = 'best')

        # Finish plot formating and save 
        fig.supxlabel('time (minutes)')
        fig.tight_layout()
        fig.savefig(''.join([dataframedir, plot_title, '_stacked.jpg']), dpi = 400, bbox_inches = 'tight', rasterize = False)   
        fig.show()
        filesorter(''.join([plot_title, '_stacked.jpg']), dataframedir, ''.join([object_name, '/', date]))
        print('\n')

        # Update the dictionary defined at the beginning of the code
        dictionary.update(data_dictionary)
        print('_' * 100, '\n')

        # True if user has input a specific dataset
        if (one_dataset == True):

            # Stack the info
            stacked_info = np.column_stack((median, std, peak_median, peak_std))

            # Place all the info into a dataframe and append to empty previously existing dataframe
            info = DataFrame(stacked_info, columns = ['Ratio Median', 'Ratio Standard Deviation.', 'Peak Ratio Median', 'Peak Ratio Standard Deviation'])
            stat_dataframe = concat([dataframe, info], axis = 1)

            # Write the statsistics to a dataframe
            stat_dataframe.to_csv(''.join([dataframedir, object_name, '/', date, '/', object_name, '_', date, '_Photometric_Statistics.csv']), index = False)

            # Save the data and break the loop after we're done with the single dataset
            print('Done Extracting Photometry! \n')
            print(''.join(['The dictionary with your data lives in ', dataframedir, '\n']))
            print(''.join(['The Dataframe with your statistics lives in ', dataframedir, '\n']))
            print('_' * 100, '\n')

            # Save the photometric data for that specific dataset
            return np.save(''.join([dataframedir, object_name, '/', date, '/', object_name, '_', date, '_Photometric_Data.npy']), dictionary)

    # Stack the info
    stacked_info = np.column_stack((median, std, peak_median, peak_std))

    # Place all the info into a dataframe and append to empty previously existing dataframe
    info = DataFrame(stacked_info, columns = ['Ratio Median', 'Ratio Standard Deviation.', 'Peak Ratio Median', 'Peak Ratio Standard Deviation'])
    stat_dataframe = concat([dataframe, info], axis = 1)

    # Save the dataframe as a .csv
    stat_dataframe.to_csv(''.join([dataframedir, 'GAPlanetS_Photometric_Statistics.csv']), index = False)

    print('Done Extracting Photometry for all of the Datasets! \n')
    print(''.join(['The dictionary with your data lives in ', dataframedir, '\n']))
    print(''.join(['The Dataframe with your statistics lives in ', dataframedir, '\n']))
    print('_' * 100, '\n')
    
    return np.save(''.join([dataframedir, 'GAPlanetS_Survey_Photometric_Data.npy']), dictionary)