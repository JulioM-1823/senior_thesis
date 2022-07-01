# Import all required packages
!pip install --upgrade matplotlib
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from astropy.io import fits
from scipy.stats import norm
from scipy.stats import skew
from google.colab import drive
drive.mount('/content/drive')

def filesorter(filename, dataframedir, foldername):

    '''
    PURPOSE: 
            Checks if the directory (dir + foldername + filename) exists, then creates the directory if it doesn't.

    INPUTS:  
            [filename; string]:  Name of the file
                 [dir; string]:  Directory where you want things to be saved
          [foldername; string]:  Name of the folder you want to check/generate

    RETURNS:  
            Creates the specified folders

    AUTHOR:  
            Julio M. Morales (adapted from Connor E. Robinson), November 04, 2021
    '''

    # The statement checks to see if the file exists.
    if os.path.exists(dataframedir + filename):
        pass
    else:
        print(dataframedir + filename + " does not exist or has already been moved.")
        return
    
    # If the foldername (input) doesn't exist, then it creates a new directory with the name foldername.
    if os.path.exists(dataframedir + foldername):
        pass
    else:
        print("Making new directory: " + dataframedir + foldername)
        os.mkdir(dataframedir + foldername)

    # Move files to new folder
    print('Moving ' + filename + ' to:  ' + dataframedir + foldername + '/' + filename)
    os.rename(dataframedir + filename, dataframedir + foldername + '/' + filename)

    return

def correlate(dataframename, dir, data_name_1=None, data_name_2=None, correlation_plots=False, residual_plots=False, object_name=None, date=None):

    '''
    PURPOSE: 
            Calculate the correlation coefficents for a given set of values for all of 
            the datasets in the GAPlanetS Database at once, and plot them against each other.  We also save the indivdual plots
            and compile a grid of all the plots together.

    INPUTS:  
                  [dataframename; string]:  File name of dataframe which contains all of the relevant strings needed to pull our data from the dictionary
                            [dir; string]:  Directory that contains dictionary which contains all of the data.
            [correlation_plots;  boolean]:  True if you want the individual correlation plots (False is the default).
               [residual_plots;  boolean]:  True if you want the residual plots with histograms (False is the default).

    OPT.
    INPUTS:
            [data_name_1; string]:  Name of the first variable you wish to correlate in the dictionary (Default is None).
            [data_name_1; string]:  Name of the second variable you wish to correlate in the dictionary (Default is None).
            [object_name; string]:  Name of the object in the database  (default is None)
                   [date; string]:  Date of the dataset in the database (default is None)    

    RETURNS:  
            (.jpg) Produces and saves the plots of each input parameter plotted against each other with their correlation
            coefficents as their label.  

    AUTHOR:  
            Julio M. Morales, November 05, 2021
    '''

    # Load in dictionary that contains the data
    data = np.load(dir + 'GAPlanetS_Survey_Photometric_Data.npy', allow_pickle = 'TRUE').item()

    # Load in the dataframe that contains the filenames
    dataframe = pd.read_csv(dir + dataframename)

    # Compile list of object names and dates
    object_list = dataframe['Object']
    date_list = dataframe['Date']
    sat_list  = dataframe['Saturated?']

    # Store these names and dates as a tupple
    tupple = list(zip(object_list, date_list, sat_list))
    
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
            one_dataset =  True
            sat = dataframe['Saturated?'][i]

    # True if the user has not specified a dataset
    if (object_name == None) and (date == None):

        one_dataset = False

    for dataset in tupple:

        # True if no object or date was specified
        if (one_dataset == False):
            
            # Pull the all the relevant info from the tupple
            object_name = dataset[0]
            date = dataset[1]
            sat  = dataset[2]

        # Pull the data and the NaN mask from the dictionary for each specfic dataset
        wfe = data[''.join([object_name, '_', date])]['Avg. WFE']
        line_r = np.array(data[''.join([object_name, '_', date])]['H-alpha FWHM (pixels)'])
        cont_r = np.array(data[''.join([object_name, '_', date])]['Cont. FWHM (pixels)'])
        time   = np.array(data[''.join([object_name, '_', date])]['time (minutes)'])
        ratio  = np.array(data[''.join([object_name, '_', date])]['H-alpha-Cont. Ratio'])
        peak_ratio = np.array(data[''.join([object_name, '_', date])]['Starpeaks Ratio'])
        ratio_err  = np.array(data[''.join([object_name, '_', date])]['H-alpha-Cont. Ratio Error'])

        # Combine the variables into one single array and convert to a pandas dataframe
        stacked_ratio = np.column_stack((line_r, ratio))
        ratio_df = pd.DataFrame(stacked_ratio)

        stacked_peak = np.column_stack((line_r, peak_ratio))
        peak_df = pd.DataFrame(stacked_peak)

        # Calculate the pearson correlation coefficents for each pair of data
        ratio_coeff = ratio_df.corr(method = 'pearson')
        peak_coeff  = peak_df.corr(method  = 'pearson')

        # Define variables to be plotted, axes names, and plot colors
        ratio_label = ratio_coeff[0][1]
        peak_label  = peak_coeff[0][1]

        # Calculate the median ratio
        median_ratio = np.nanmedian(ratio)
        median_peak_ratio = np.nanmedian(peak_ratio)

        # Cacluclate standard deviations
        ratio_std = np.nanstd(ratio)
        peak_ratio_std = np.nanstd(peak_ratio)

        # Define variables to plotted
        variables = [data[''.join([object_name, '_', date])]['Avg. WFE'], 
                     data[''.join([object_name, '_', date])]['H-alpha Flux'], 
                     data[''.join([object_name, '_', date])]['Cont. Flux'], 
                     data[''.join([object_name, '_', date])]['H-alpha-Cont. Ratio'], 
                     data[''.join([object_name, '_', date])]['Starpeaks Ratio'], 
                     data[''.join([object_name, '_', date])]['H-alpha FWHM (pixels)'], 
                     data[''.join([object_name, '_', date])]['Cont. Starpeaks'], 
                     data[''.join([object_name, '_', date])]['H-alpha Starpeaks'], 
                     data[''.join([object_name, '_', date])]['Cont. FWHM (pixels)']]

        # Axes labels for variables to be plotted
        variable_names = [r'Avg. WFE',
                          r'H-alpha Flux (ADU)', 
                          r'Cont. Flux (ADU)',  
                          r'H-alpha-Cont Ratio.', 
                          r'Peak H-alpha-Cont. Ratio', 
                          r'H-alpha FWHM (pixels)', 
                          r'Cont. Peak (ADU)',
                          r'H-alpha Peak (ADU)', 
                          r'Cont. FWHM (pixels)']

        # Colors of plots
        color_list = ['ko', 'bo', 'bo', 'mo', 'mo', 'co', 'co', 'ro', 'ro']
        
        # True if MAG1 Seeing data exists
        if 'MAG1 Seeing (arcseconds)' in list(data[''.join([object_name, '_', date])].keys()):
            
            # Store all the data in a list
            mag  = data[''.join([object_name, '_', date])]['MAG1 Seeing (arcseconds)']
            variables.insert(0, mag)
            color_list.insert(0, 'go')
            variable_names.insert(0, 'MAG1 Seeing (arcseconds)')

        # True if the MAG1 and DIMM seeing data exists
        if 'DIMM Instrument' in list(data[''.join([object_name, '_', date])].keys()):

            # Store all the data in a list
            dimm = data[''.join([object_name, '_', date])]['DIMM Instrument']
            variables.insert(0, dimm)
            color_list.insert(0, 'go')
            variable_names.insert(0, 'DIMM Seeing (arcseconds)')

        # Set the title of the plot for saturated and nonsaturated cases
        if sat == 'Y':
            plot_title = ''.join([object_name, ' ', date, ' Ghost'])
        else:
            plot_title = ''.join([object_name, ' ', date])

        # True if you want to print the correlation plots
        if (correlation_plots == True):

            # True if only want to correlate 2 variables
            if (data_name_1 != None) and (data_name_2 != None):

                # Pull the data
                variable_1 = data[''.join([object_name, '_', date])][data_name_1]
                variable_2 = data[''.join([object_name, '_', date])][data_name_2]

                # Combine the variables into one single array and convert to a pandas dataframe
                stacked_variables = np.column_stack((variable_1, variable_2))
                variables_df = pd.DataFrame(stacked_variables)

                # Calculate the pearson correlation coefficents for each pair of data
                coeff = variables_df.corr(method = 'pearson')

                # Define plot color and label
                color = 'mo'
                label = coeff[0][1]

                # Make the plots with their respective correlation coeffcients as their labels
                plt.figure(figsize = (10, 10))
                plt.rc('font', size = 22)
                plt.plot(variable_2, variable_1, color, label = f'Correlation:  {label:.2f}')
                plt.xlabel(data_name_2)
                plt.ylabel(data_name_1)
                plt.ylim(0.5*np.nanmedian(variable_1), 1.5*np.nanmedian(variable_1))
                plt.xlim(0.5*np.nanmedian(variable_2), 1.5*np.nanmedian(variable_2))
                plt.title(plot_title)
                plt.legend(loc = 'best')
                plt.tight_layout()
                plt.savefig(dir + data_name_1 + ' vs. ' + data_name_2 + ' Correlation Plot for ' + object_name + ' ' + date + '.jpg')
                filesorter(data_name_1 + ' vs. ' + data_name_2 + ' Correlation Plot for ' + object_name + ' ' + date + '.jpg', dir, object_name + '/' + date  + '/' + 'correlation plots')

                # True if only want one dataset, else continue on to next dataset
                if one_dataset == True:
                    return
                else:
                    continue
            
            else:

                # Loop over all variables pairs, calculate their correlation coefficents, and plot them against each other
                # while not duplicating plots or plotting the same variables against each other
                for i in range(0, len(variables) - 1):

                    for j in range(i + 1, len(variables)):

                        # Combine the variables into one single array and convert to a pandas dataframe
                        stacked_variables = np.column_stack((variables[i], variables[j]))
                        variables_df = pd.DataFrame(stacked_variables)

                        # Calculate the pearson correlation coefficents for each pair of data
                        coeff = variables_df.corr(method = 'pearson')

                        # Define variables to be plotted, axes names, and plot colors
                        plotting_pairs = [variables[i], variables[j]]
                        axis_names = [variable_names[i], variable_names[j]]
                        color = color_list[i]
                        label = coeff[0][1]

                        # Make the plots with their respective correlation coeffcients as their labels
                        plt.figure(figsize = (10, 10))
                        plt.rc('font', size = 22)
                        plt.plot(plotting_pairs[1], plotting_pairs[0], color, label = f'Correlation:  {label:.2f}')
                        plt.xlabel(axis_names[1])
                        plt.ylabel(axis_names[0])
                        plt.ylim(0.9*np.nanmin(plotting_pairs[0]), 1.1*np.nanmax(plotting_pairs[0]))
                        plt.xlim(0.9*np.nanmin(plotting_pairs[1]), 1.1*np.nanmax(plotting_pairs[1]))
                        plt.title(plot_title)
                        plt.legend(loc = 'best')
                        plt.tight_layout()
                        plt.savefig(dir + axis_names[0] + ' vs. ' + axis_names[1] + ' Correlation Plot for ' + object_name + ' ' + date + '.jpg')
                        filesorter(axis_names[0] + ' vs. ' + axis_names[1] + ' Correlation Plot for ' + object_name + ' ' + date + '.jpg', dir, object_name + '/' + date  + '/' + 'correlation plots')

        # True if you want to print the residuals and histograms
        if (residual_plots == True):

            # Define the figure for all of the values to be plotted on
            plt.rc('font', size = 25)
            fig, axs = plt.subplots(2, 2, figsize = (20, 10), sharey = True, gridspec_kw = {'height_ratios': [0.5, 1], 'wspace': 0})

            # axs[0, 0].errorbar(time, ratio, yerr = ratio_err, fmt = 'mo', alpha = 0.5, markersize = 3)
            # axs[0, 0].hlines(median_ratio, xmin = np.nanmin(time), xmax = np.nanmax(time), colors = 'blue', linestyles = '--', linewidth = 2, label = f'Median = {median_ratio:.2f} +/- {ratio_std:.2f}')
            # axs[0, 0].set_xlabel('time (minutes)')
            # axs[0, 0].set_ylabel(r'H$\alpha$/Cont')
            # axs[0, 0].set_title('Photometry')
            # axs[0, 0].legend(loc = 'best')

            # axs[0, 1].plot(time, peak_ratio, 'mo', alpha = 0.5, markersize = 3)
            # axs[0, 1].hlines(median_peak_ratio, xmin = np.nanmin(time), xmax = np.nanmax(time), colors = 'blue', linestyles = '--', linewidth = 2, label = f'Median = {median_peak_ratio:.2f} +/- {peak_ratio_std:.2f}')
            # axs[0, 1].set_xlabel('time (minutes)')
            # axs[0, 1].set_title('Peaks')
            # axs[0, 1].legend(loc = 'best')

            # axs[1, 0].plot(cont_r, ratio, 'ro', label = f'Correlation:  {ratio_label:.2f}', alpha = 0.5, markersize = 3)
            # axs[1, 0].set_xlabel('Continuum FWHM (pixels)')
            # axs[1, 0].set_ylabel(r'H$\alpha$/Cont')
            # axs[1, 0].legend(loc = 'best')

            # axs[1, 1].plot(cont_r, peak_ratio, 'ro', label = f'Correlation:  {peak_label:.2f}', alpha = 0.5, markersize = 3)
            # axs[1, 1].set_xlabel('Continuum FWHM (pixels)')
            # axs[1, 1].legend(loc = 'best')

            axs[0, 0].errorbar(time, ratio, yerr = ratio_err, fmt = 'mo', alpha = 0.5, markersize = 3)
            axs[0, 0].hlines(median_ratio, xmin = np.nanmin(time), xmax = np.nanmax(time), colors = 'blue', linestyles = '--', linewidth = 2, label = f'Median = {median_ratio:.2f} +/- {ratio_std:.2f}')
            axs[0, 0].set_xlabel('time (minutes)')
            axs[0, 0].set_ylabel(r'H$\alpha$/Cont')
            axs[0, 0].set_title('Photometry')
            axs[0, 0].legend(loc = 'best')

            axs[0, 1].plot(time, peak_ratio, 'mo', alpha = 0.5, markersize = 3)
            axs[0, 1].hlines(median_peak_ratio, xmin = np.nanmin(time), xmax = np.nanmax(time), colors = 'blue', linestyles = '--', linewidth = 2, label = f'Median = {median_peak_ratio:.2f} +/- {peak_ratio_std:.2f}')
            axs[0, 1].set_xlabel('time (minutes)')
            axs[0, 1].set_title('Peaks')
            axs[0, 1].legend(loc = 'best')

            axs[1, 0].plot(wfe, ratio, 'ro', label = f'Correlation:  {ratio_label:.2f}', alpha = 0.5, markersize = 3)
            axs[1, 0].set_xlabel('Avg. WFE (nm)')
            axs[1, 0].set_ylabel(r'H$\alpha$/Cont')
            axs[1, 0].legend(loc = 'best')

            axs[1, 1].plot(wfe, peak_ratio, 'ro', label = f'Correlation:  {peak_label:.2f}', alpha = 0.5, markersize = 3)
            axs[1, 1].set_xlabel('Avg. WFE (nm)')
            axs[1, 1].legend(loc = 'best')

            fig.suptitle(plot_title)
            fig.subplots_adjust(wspace = 0)
            fig.tight_layout()
            fig.show()

            # Define resiudals
            residuals = ratio - median_ratio
            peak_residuals = peak_ratio - median_peak_ratio
            
            # Remove NaN's from the list
            residuals = residuals[~np.isnan(residuals)]
            peak_residuals = peak_residuals[~np.isnan(peak_residuals)]
    
            # Define parameters for Gaussian fit
            mu, std = norm.fit(residuals)
            peak_mu, peak_std = norm.fit(peak_residuals)
            mean = np.mean(residuals)
            peak_mean = np.mean(peak_residuals)
            variance  = np.var(residuals)
            peak_variance = np.var(peak_residuals)
            sigma = np.sqrt(variance)
            peak_sigma = np.sqrt(peak_variance)
            x = np.linspace(min(residuals), max(residuals), 100)
            peak_x = np.linspace(min(peak_residuals), max(peak_residuals), 100)

            # Calculate the skew
            ratio_skew = skew(residuals)
            peak_ratio_skew = skew(peak_residuals)

            # Messy code to set axes of plots on the same scale
            if (np.nanmax(residuals) >= np.nanmax(peak_residuals)):
                max_scale = np.nanmax(residuals)
            else:
                max_scale = np.nanmax(peak_residuals)

            if (np.nanmin(residuals) <= np.nanmin(peak_residuals)):
                min_scale = np.nanmin(residuals)
            else:
                min_scale = np.nanmin(peak_residuals)

            # Messy code to set axes of plots on the same scale
            if (np.nanmax(residuals) >= np.nanmax(peak_residuals)):
                max_scale = np.nanmax(residuals)
            else:
                max_scale = np.nanmax(peak_residuals)

            if (np.nanmin(residuals) <= np.nanmin(peak_residuals)):
                min_scale = np.nanmin(residuals)
            else:
                min_scale = np.nanmin(peak_residuals)

            # Define figure for residuals and histograms
            fig = plt.figure(figsize = (20, 8))
            gs  = GridSpec(4, 4)

            # Add first plot for the residuals
            ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            plt.plot(line_r, ratio - median_ratio, 'mo', markersize = 3, label = 'Photometry')
            plt.plot(line_r, np.zeros(len(ratio)), 'k-')
            plt.ylim(min_scale, max_scale)
            plt.xlim(np.nanmin(line_r), np.nanmax(line_r))
            plt.locator_params(axis = 'y', nbins = 6)
            plt.xticks([])
            ax_scatter.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.ylabel('Residuals')
            plt.title(plot_title)
            plt.legend(loc = 'best')

            # Tighten up the formatting
            plt.subplots_adjust(hspace = 0)
            plt.subplots_adjust(wspace = 0)

            ### DUMMY PEAK HISTOGRAM TO CALCULATE X LIMITS ON PLOTS.
            ### I DID THIS SO THE HISTOGRAMS X-LIMS LINE UP
            ### ALPHA IS TURNED TO 0 SO THESE DONT OVERPLOT
            peak_histogram = plt.hist(peak_residuals, bins = 'auto', alpha = 0, color = 'w', orientation = 'horizontal')
            peak_dx = peak_histogram[1][1] - peak_histogram[1][0]
            peak_scale = len(peak_residuals)*peak_dx

            # Add the histogram to the side of the plot
            ax_hist_x = fig.add_subplot(gs[1:4, 3], sharey = ax_scatter)
            histogram = plt.hist(residuals, bins = 'auto', alpha = 0.6, color = 'r', orientation = 'horizontal')
            dx = histogram[1][1] - histogram[1][0]
            scale = len(residuals)*dx
            max_hist_1 = max(norm.pdf(x, mean, sigma)*scale)
            max_hist_2 = max(norm.pdf(peak_x, peak_mean, peak_sigma)*peak_scale)

            # Messy code to set axes of plots on the same scale
            if (max_hist_1 >= max_hist_2):
                max_hist = max_hist_1
            else:
                max_hist = max_hist_2

            plt.plot(norm.pdf(x, mean, sigma)*scale, x, 'k-')
            plt.plot(np.nan, np.nan, 'w', markersize = 0, label = f'Skew = {ratio_skew:.2f}')
            plt.legend(loc = 'best')
            plt.xlim(0, max_hist)

            # Get rid of the tick labels on the histogram plot
            ax_hist_x.get_yaxis().set_visible(False)
            ax_hist_x.get_xaxis().set_visible(False)
            fig.tight_layout()
            fig.savefig(dir + 'slope_fitted_light_curves.jpg', dpi = 400, bbox_inches = 'tight', rasterize = False)
            fig.show()

            #####################
            # NOW FOR THE PEAKS #
            #####################

            # Define figure for residuals and histograms
            fig2 = plt.figure(figsize = (20, 8))
            gs2  = GridSpec(4, 4)

            peak_ax_scatter = fig2.add_subplot(gs2[1:4, 0:3])
            plt.plot(line_r, peak_ratio - median_peak_ratio, 'mo', markersize = 3, label = 'Peaks')
            plt.plot(line_r, np.zeros(len(peak_ratio)), 'k-')
            plt.ylim(min_scale, max_scale)
            plt.xlim(np.nanmin(line_r), np.nanmax(line_r))
            plt.locator_params(axis = 'y', nbins = 6)
            peak_ax_scatter.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.xlabel(r'H$\alpha$ FWHM (pixels)')
            plt.ylabel('Residuals')
            plt.legend(loc = 'best')

            # Tighten up the formatting
            plt.subplots_adjust(hspace = 0)
            plt.subplots_adjust(wspace = 0)

            peak_ax_hist_x = fig2.add_subplot(gs2[1:4, 3], sharey = peak_ax_scatter)
            peak_histogram = plt.hist(peak_residuals, bins = 'auto', alpha = 0.6, color = 'r', orientation = 'horizontal')
            peak_dx = peak_histogram[1][1] - peak_histogram[1][0]
            peak_scale = len(peak_residuals)*peak_dx
            plt.plot(norm.pdf(peak_x, peak_mean, peak_sigma)*peak_scale, peak_x, 'k-')
            plt.plot(np.nan, np.nan, 'w', markersize = 0, label = f'Skew = {peak_ratio_skew:.2f}')
            plt.legend(loc = 'best')
            plt.xlim(0, max_hist)

            # Get rid of the tick labels on the histogram plot
            peak_ax_hist_x.get_yaxis().set_visible(False)
            fig2.tight_layout()
            fig2.savefig(dir + 'slope_fitted_light_curves.jpg', dpi = 400, bbox_inches = 'tight', rasterize = False)
            fig2.show()

        # True if user has input a specific dataset
        if (one_dataset == True):

            return 

    return 