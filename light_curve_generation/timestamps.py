# Import all the toys
import numpy as np
import glob
from astropy.io import fits
from datetime import datetime
from pandas import DataFrame
from pandas import read_csv
from pandas import to_datetime

# Mount the google drive (all the data is stored in the 'Follette-Lab-AWS' and 'Follette-Lab-AWS-2' drives)
from google.colab import drive
drive.mount('/content/drive')

def timestamps(dataframedir, dataframename, object_name = None, date = None):

    '''
    PURPOSE:
            Extract timestamps from the headers of .fits files, convert them into minutes, and place NaN's in the locations where
            images where rejected in the preprocessing due to cosmic rays.

    INPUTS:
           [dataframedir; string]:  Directory where the dataframe that contains all of the naming conventions lay for the various datasets
          [dataframename; string]:  Name of the dataframe.
  
    OPT.
    INPUTS:
            [object_name; string]:  Name of the object in the database  (default is None)
                   [date; string]:  Date of the dataset in the database (default is None)
  
    RETURNS:
            [time_dictionary; {dictionary}]: ---> keywords()
                                                     |
                                                     |
                                                     v
                                        [object_date; string]:  Object and date of interest
             [time (minutes since midnight); np.array, float]:  Timestamps in minutes since midnight
                                                [dim; string]:  Name of the DIM seeing values in the dictionary
                                                [mag; string]:  Name of the MAG1 seeing values in the dictionary
                                [nan_mask; np.array, integer]:  Array of 1's and NaNs

    AUTHOR:
          Julio M. Morales, November 22, 2021
    '''

    # Define empty dictionary to be added to
    time_dictionary = {}

    # Load in dataframe
    dataframe = read_csv(dataframedir + dataframename)

    # Compile list of object names and dates
    object_list   = dataframe['Object']
    date_list     = dataframe['Date']
    dir_list      = dataframe['Directory']
    sat_list      = dataframe['Saturated?']
    imstring_list = dataframe['Preprocessed File Name']

    # Store these names and dates as a tupple
    tupple = list(zip(object_list, date_list, dir_list, sat_list, imstring_list))
    
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
            dir      = dataframe['Directory'][i] + object_name + '/' + date + '/'
            sat      = dataframe['Saturated?'][i]
            imstring = dataframe['Preprocessed File Name'][i]
            one_dataset =  True

    # True if the user has not specified a dataset
    if (object_name == None) and (date == None):

        one_dataset = False

    for dataset in tupple:

        # True if no object or date was specified
        if (one_dataset == False):
            
            # Pull the all the relevant info from the tupple
            object_name = dataset[0]
            date        = dataset[1]
            dir         = dataset[2] + object_name + '/' + date + '/'
            sat         = dataset[3]
            imstring    = dataset[4]

        # Collect all of the raw files, and sort them by their names (their names are their timestamps
        # so this is essentially sorting them in order of time)
        raw_files = glob.glob(dir + 'raw/' + '*.fits')
        raw_files.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        # Define empty list for data
        data = []                                                                      

        print('Extracting Timestamps for ' + object_name + ' ' + date + '...\n')
        print('- - ' * 25)
        print('\n')

        # Loop through all the files
        for i in np.arange(len(raw_files)):
            print('Image ' + str(i) + '/' + str(len(raw_files)) + '\n')
        
            # Extract the specified header info
            header =      fits.getheader(raw_files[i])
            header_keys = list(header.keys())

            # The following 'if' statements check if the raw file contain seeing values and addes them to the dataframe if they do
            info = [header['Object'], header['DATE-OBS'], header['VIMTYPE']]
            cols = ['Object Name','Date', 'Image Type']

            # True if MAG1 Seeing data exists
            if 'MAG1FWHM' in header_keys:

                # True if the MAG1 and DIMM seeing data exists
                if 'DIMMFWHM' in header_keys:
                    
                    # Append info
                    info.append(header['DIMMFWHM'])
                    info.append(header['MAG1FWHM'])
                    cols.append('MAG Seeing')
                    cols.append('DIMM Seeing')

                # True if the MAG1 seeing data ecists and the DIMM seeing data does NOT exist
                else:
                    
                    # Append info
                    info.append(header['MAG1FWHM'])
                    cols.append('MAG Seeing')

            # True if the MAG1 seeing data does NOT exist
            else:

                # True if the MAG1 seeing does NOT exist but the DIMM seeing data does exist
                if 'DIMMFWHM' in header_keys:
   
                    # Append info
                    info.append(header['DIMMFWHM'])
                    cols.append('DIMM Seeing')

            # This condition ensures that the images are SCIENCE images, and that the AO system was working
            if (header['VIMTYPE'] == 'SCIENCE') and (header['AOLOOPST'] == 'CLOSED'): 

                # Append info to the empty list
                data.append(info)

        print('- - ' * 25)
        print('\n')

        # Create pandas dataframe from the data
        df = DataFrame(data, columns = cols)
        timestamps_none_rejected = to_datetime(df['Date'])

        # Print timestamps of the given object
        print('Timestamps of ' + object_name + ' ' + date + ':', '\n')
        print(timestamps_none_rejected, '\n')
        print('- - ' * 25, '\n')

        # Wavelength of image
        wavelength = ['Line', 'Cont']
        
        # Lists of the timestamps for both wavelengths that we will modify
        line_timestamps = timestamps_none_rejected
        cont_timestamps = timestamps_none_rejected

        # Empty list of timestamps in seconds to be added to
        line_seconds = []
        cont_seconds = []

        # We need the timestamps in a form that we can operate on so we can convert the timestamps into seconds since midnight
        # Define midnight using the first timestamps of the day
        date1 = timestamps_none_rejected[0].date()   
        year  = date1.year  
        month = date1.month                           
        day  = date1.day                                            
        hour = 23
        minute  = 59
        seconds = 59
        miliseconds = 999999
        
        # Iterate over the possible wavelengths
        for w in wavelength:
    
            # Extract the data from the image
            image = fits.getdata(dir + 'preprocessed/' + w + imstring + '.fits')
            print('Files in ' + w + ' Preproccessed Data:   ',  image.shape[0])

            # Extract the files that contain the integer locations of the accepted images
            rejection_integers = fits.getdata(dir + 'calibration/' + w + 'cosmics.fits')
            print('Files in ' + w + 'cosmics:               ', len(rejection_integers), '(this should be the same as the number printed in the line above) \n')

            # Convert all of the timestamps to an operable form (floats)
            for j in range(0, len(timestamps_none_rejected)):

                # If True, deletes from the line timestamps
                if (w == 'Line'):

                    # True when the timestamps are before midnight of the first night
                    if (line_timestamps[j] <= datetime(year, month, day, hour, minute, seconds, miliseconds)):

                        # Pulls the time info from the timestamps
                        t = datetime.time(line_timestamps[j])

                        # Convert the line timestamps to a float time and replace each stamp in the list with the float
                        line_seconds.append((t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1000000.0))

                    # True when the timestamps are after midnight
                    else:

                        # Pull the time info from the timestamp
                        t = datetime.time(line_timestamps[j])

                        # Convert the line timestamps to a float time and add each stamp to the list with the float NOTE THAT WE ADD 86400s to acccount
                        # for the fact that these timestamps are from the next day, so we add a days worth of seconds.
                        line_seconds.append((t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1000000.0) + 86400)

                # True when the wavelength is Cont
                else:
                
                    # True when the timestamps are before midnight of the first night
                    if (cont_timestamps[j] <= datetime(year, month, day, hour, minute, seconds, miliseconds)):

                        # Pulls the time info from the timestamps
                        t = datetime.time(cont_timestamps[j])

                        # Convert the line timestamps to a float time and replace each stamp in the list with the float
                        cont_seconds.append((t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1000000.0))
                
                    # True when the timestamps are after midnight
                    else:

                        # Pull the time info from the timestamp
                        t = datetime.time(cont_timestamps[j])

                        # Convert the line timestamps to a float time and add each stamp to the list with the float NOTE THAT WE ADD 86400s to acccount
                        # for the fact that these timestamps are from the next day, so we add a days worth of seconds.
                        cont_seconds.append((t.hour * 3600) + (t.minute * 60) + t.second + (t.microsecond / 1000000.0) + 86400)

            # Comb through range of integer values
            for k in range(0, len(timestamps_none_rejected)):

                # True if indices in rejection_integers are missing
                if k not in rejection_integers:

                    # True if working with Line images
                    if (w == 'Line'):

                        # Place NaN in location of missing integers
                        line_seconds[k] = np.nan

                    # True if working with Cont images
                    else:

                        # Place NaN in location of missing integers
                        cont_seconds[k] = np.nan

        # Creates an array of 1's and 'NaN's.
        nan_mask = np.array(line_seconds) / np.array(cont_seconds)

        # Convert general timestamps to a numpy array and convert to minutes
        time = (np.array(line_seconds) * nan_mask) / 60

        # Dictionary to hold time and seeing data
        inner_dict = {'time (minutes since midnight)':  time, 
                                           'nan mask':  nan_mask}

        # Add the timestamps and the NaN mask to a dictionary
        dictionary = {object_name +  '_' + date: inner_dict}

        # True if MAG1 Seeing data exists
        if 'MAG Seeing' in df:

            # True if the MAG1 and DIMM seeing data exists
            if 'DIMM Seeing' in df:

                # Store all the data in a list and replace any nonphysical seeing values with NaN
                dimm = df['DIMM Seeing'] * nan_mask
                dimm = [np.nan if i == -1 else i for i in dimm]

                mag = df['MAG Seeing']  * nan_mask
                mag = [np.nan if i == -1 else i for i in mag]

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in dimm) != True):
                    inner_dict['DIMM Instrument'] = dimm

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in mag) != True):
                    inner_dict['MAG1'] = mag

            # True if the MAG1 seeing data ecists and the DIMM seeing data does NOT exist
            else:

                # Store all the data in a list
                mag = df['MAG Seeing']  * nan_mask
                mag = [np.nan if i == -1 else i for i in mag]

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in mag) != True):
                    inner_dict['MAG1'] = mag
                
        # True if the MAG1 seeing data does NOT exist
        else:

            # True if the MAG1 seeing does NOT exist but the DIMM seeing data does exist
            if 'DIMM Seeing' in df:

                # Store all the data in a list and replace any nonphysical seeing values with NaN
                dimm = df['DIMM Seeing'] * nan_mask
                dimm = [np.nan if i == -1 else i for i in dimm]

                # True if all the seeing values were NaN
                if (all(np.isnan(i) for i in dimm) != True):
                    inner_dict['DIMM Instrument'] = dimm

        # Add data to the dictionary defined earlier
        time_dictionary.update(dictionary)
        print('- - ' * 25, '\n')
        print('Timestamps for ' + object_name + '_' + date + ' Extracted Successfully! \nExtracting Timestamps for the Next Dataset...\n')
        print('_' * 100, '\n')
        
        # True if user has input a specific dataset
        if (one_dataset == True):

            # Save the data and break the loop after we're done with the single dataset
            np.save(dataframedir + object_name + '/' + date + '/' + object_name + '_' + date + '_Timestamps.npy', dictionary)
            print('\n')
            print('The dictionary with your data lives in ' + dataframedir + '\n')
            print('_' * 100, '\n')

            return 
  
    # Save the dictionary to a .npy file
    np.save(dataframedir + 'GAPlanetS_Survey_Timestamps.npy', time_dictionary)
    print('All Timestamps Extracted! \n')
    print('The dictionary with your data lives in ' + dataframedir + '\n')
    print('_' * 100, '\n')

    return time_dictionary