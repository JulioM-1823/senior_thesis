import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from IPython.display import display, Latex
from scipy.interpolate import interp1d
import glob
from google.colab import drive
from __future__ import print_function
!pip install PyAstronomy
from PyAstronomy.pyasl import planck
!pip install synphot

# Import and define plotting style
import seaborn as sb
plt.rcParams['font.family']    = 'monospace'
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
sb.set_context("talk")
drive.mount('/content/drive')
dir = '/content/drive/My Drive/Julios Thesis Codes/Data/Ratio Plots/'
N = 400

# Define wavelength in meters and angstroms
lam  = np.arange(0, 30000 * 10**(-10), 20e-10)
wave = np.linspace(0, 30000, 1500)

# Get the Planck spectrum in [W/(m**2 m)] for a temperature of 7000 K
planck_m =  planck(6632, lam = lam)

# Convert into erg/(cm**2 * A * s)
planck_cm = planck_m * 10**(-14)

plt.figure(figsize = (15, 10))
plt.rc('font', size = 20)
plt.title(r'Planck Function ($T=6632 K)$')
plt.plot(wave, planck_cm, 'k-', linewidth = 2.5, label = r'$B_{\lambda}(T, \lambda)$')
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.xlim(0, 30000)
plt.ylim(0, 2.3)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'planck_function_HD.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

plt.figure(figsize = (15, 10))
plt.rc('font', size = 20)
plt.title(r'Planck Function ($T=6632 K)$')
plt.plot(wave, planck_cm, 'k-', linewidth = 2.5, label = r'$B_{\lambda}(6632\, K, \lambda)$')
plt.vlines(6560, ymin = np.nanmin(planck_cm), ymax = np.nanmax(planck_cm), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    ($6560\, \AA$)')
plt.vlines(6430, ymin = np.nanmin(planck_cm), ymax = np.nanmax(planck_cm), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. ($6430\, \AA$)')
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.xlim(0, 30000)
plt.ylim(0, 2.3)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'planck_function_lines_HD.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

plt.figure(figsize = (10, 10))
plt.rc('font', size = 20)
plt.title(r'Planck Function ($T=6632 K)$')
plt.plot(wave, planck_cm, 'k-', linewidth = 4, label = r'$B_{\lambda}(6632\, K, \lambda)$')
plt.vlines(6560, ymin = np.nanmin(planck_cm), ymax = np.nanmax(planck_cm), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    ($6560\, \AA$)')
plt.vlines(6430, ymin = np.nanmin(planck_cm), ymax = np.nanmax(planck_cm), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. ($6430\, \AA$)')
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.xlim(6400, 6600)
plt.ylim(0.8, 1.4)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'planck_function_lines_zoomed_HD.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

plt.figure(figsize = (10, 10))
plt.rc('font', size = 20)
plt.title(r'Planck Function ($T=6632 K)$')
plt.plot(wave, planck_cm, 'k-', linewidth = 4, label = r'$B_{\lambda}(6632 \, K, \lambda)$')
plt.hlines(planck(6632, lam = 6560*10**(-10)) * 10**(-14) , xmin = np.nanmin(wave), xmax = np.nanmax(wave), colors = 'blue', linestyles = '--', linewidth = 3, label = r'$B_{\lambda}(6560 \, \AA, 6750 \, K)$')
plt.vlines(6560, ymin = np.nanmin(planck_cm), ymax = np.nanmax(planck_cm), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    ($6560\, \AA$)')
plt.vlines(6430, ymin = np.nanmin(planck_cm), ymax = np.nanmax(planck_cm), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. ($6430\, \AA$)')
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.xlim(6400, 6600)
plt.ylim(0.8, 1.4)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'planck_function_lines_zoomed_interpolation_HD.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

ratio_value = planck(6632, lam = 6560*10**(-10)) / planck(6632, lam = 6430*10**(-10))
print('H-alpha/Cont Flux Ratio Using the Planck Function = ' + str(f'{float(f"{ratio_value:.2g}"):g}'))

# Define the range of temperatures to be that of main-sequence stars
T =     np.linspace(2000, 32000, 1500)
ratio = planck(T, lam = 6560*10**(-10)) / planck(T, lam = 6430*10**(-10))

tick_locations = [30, 25, 10, 7.5, 6.0, 5.2, 3.7]
tick_labels =    ['O', 'B', 'A', 'F', 'G', 'K', 'M']

fig = plt.figure(figsize = (25, 15))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(T/1000, ratio, 'r-', linewidth = 3)
ax2.set_xbound(ax1.get_xbound())
ax1.set_xlabel(r'Effective Temperature $[10^3 \,K]$')
ax1.set_ylabel(r'Quiescent H$\alpha$/Cont. Ratio')
ax1.set_xlim(ax1.get_xlim()[::-1])
ax2.set_xlim(ax2.get_xlim()[::-1])
ax1.set_xticks(tick_locations)
ax2.set_xticks(tick_locations)
ax2.set_xticklabels(tick_labels)
ax2.set_xlabel('Spectral Class')
plt.tight_layout()
plt.savefig(dir + 'planck_ratio.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

# Load in the model data
data, header = fits.getdata('/content/drive/MyDrive/Julios Thesis Codes/Data/Stellar Atmospheres/ckp02_6500.fits', header = True)

# Pull the wavelength and flux data from the header
wavelength = data['WAVELENGTH']
flux =       data['g30']/10**7

# Make the plot
plt.figure(figsize = (15, 10))
plt.rc('font', size = 20)
plt.title(r'Model SED [$T=6750 \, K$, $log_{10}(g)=+3.0$, $[M/H]=+0.2$]')
plt.plot(wavelength, flux, 'ko-', markersize = 1, label = 'Model')
plt.xlim(0, 30000)
plt.ylim(0, 2.3)
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'model_sed.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

plt.figure(figsize = (15, 10))
plt.rc('font', size = 20)
plt.title(r'Model SED [$T=6750 \, K$, $log_{10}(g)=+3.0$, $[M/H]=+0.2$]')
plt.plot(wavelength, flux, 'ko-', markersize = 1, label = 'Model')
plt.vlines(6560, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    ($6560\, \AA$)')
plt.vlines(6430, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. ($6430\, \AA$)')
plt.xlim(0, 30000)
plt.ylim(0, 2.3)
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'model_sed_lines.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

# Interpolate the data
f = interp1d(wavelength, flux)

plt.figure(figsize = (15, 10))
plt.rc('font', size = 20)
plt.title(r'Model SED [$T=6750 \, K$, $log_{10}(g)=+3.0$, $[M/H]=+0.2$]')
plt.plot(wavelength, flux, 'ko-', label = 'Model')
plt.plot(wavelength, f(wavelength), 'b-', label = 'Interpolation')
plt.vlines(6560, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    ($6560\, \AA$)')
plt.vlines(6430, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. ($6430\, \AA$)')
plt.xlim(0, 30000)
plt.ylim(0, 2.3)
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'model_sed_lines_interpolation.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

plt.figure(figsize = (10, 10))
plt.title(r'Model SED [$T=6750 \, K$, $log_{10}(g)=+3.0$, $[M/H]=+0.2$]')
plt.plot(wavelength, flux, 'ko-', label = 'Model')
plt.plot(wavelength, f(wavelength), 'b-', label = 'Interpolation')
plt.vlines(6560, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    ($6560\, \AA$)')
plt.vlines(6430, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. ($6430\, \AA$)')
plt.xlim(6400, 6600)
plt.ylim(0.8, 1.4)
plt.xlabel("Wavelength [$\AA$]")
plt.ylabel(r"Flux $[10^7 \,erg/cm^2/A/s$]")
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'model_sed_lines_interpolation_zoomed.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

# Load in the filter response fnctions
line_filter = np.loadtxt('/content/drive/MyDrive/Julios Thesis Codes/Data/Filter Response Functions/VisAO_Line_Filter_Response.txt')
cont_filter = np.loadtxt('/content/drive/MyDrive/Julios Thesis Codes/Data/Filter Response Functions/VisAO_Cont_Filter_Response.txt')

line_wave = line_filter[:,0]
line_norm = line_filter[:,1]

cont_wave = cont_filter[:,0]
cont_norm = cont_filter[:,1]

# Interpolate filter response function
line_int = interp1d(line_wave, line_norm)
cont_int = interp1d(cont_wave, cont_norm)

plt.figure(figsize = (17, 10))
plt.rc('font', size = 20)
plt.title(r'VisAO H$\alpha$ and Continuum Filter Response Functions')
plt.plot(line_wave, line_norm, 'tab:red',    linewidth = 4, label = r'Centered on $6560 \, \AA$')
plt.plot(cont_wave, cont_norm, 'tab:orange', linewidth = 4, label = r'Centered on $6430\, \AA$')
plt.xlabel("Wavelength [$\AA$]")
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(dir + 'line_filter_response.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()

# Define the range of wavelengths the filters are sensitive to
line_wavespace = np.linspace(min(line_wave), max(line_wave), 10000)
cont_wavespace = np.linspace(min(cont_wave), max(cont_wave), 10000)

# Calculate the total fluxes using the interpolated filter response functions
line_flux = sum(line_int(line_wavespace) * f(line_wavespace))
cont_flux = sum(cont_int(cont_wavespace) * f(cont_wavespace))

# Calculate the ratio
ratio = line_flux / cont_flux
print('Quiescent H-alpha Line Flux (6,560 A) = ' + str(f'{float(f"{line_flux:.2g}"):g}') + ' erg/s/A/cm^2 \n')
print('Quiescent Continumn Flux    (6,430 A) = ' + str(f'{float(f"{cont_flux:.2g}"):g}') + ' erg/s/A/cm^2 \n')
print('Quiescent H-alpha/Cont Flux Ratio for F5 V Star = ' + str(f'{float(f"{ratio:.2g}"):g}'))

# Retrieve all the models
models = glob.glob('/content/drive/MyDrive/Julios Thesis Codes/Data/Stellar Atmospheres/*.fits')
models.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

model_ratios = []
temperature =  []

# Extract header from each
for path in models:

    # Pull the data and header
    data =   fits.getdata(path)
    header = fits.getheader(path)

    # Pull the wavelength and flux data from the data and header
    wavelength = data['WAVELENGTH']
    flux =       data['g30']
    t_eff =      header['TEFF']

    # Interpolate the data
    f = interp1d(wavelength, flux)
    
    # Calculate the total fluxes using the interpolated filter response functions
    line_flux = sum(line_int(line_wavespace) * f(line_wavespace))
    cont_flux = sum(cont_int(cont_wavespace) * f(cont_wavespace))
    ratio = line_flux/cont_flux

    model_ratios.append(ratio)
    temperature.append(t_eff)

    # Plot the SED
    # plt.figure(figsize = (25, 10))
    # plt.rc('font', size = 20)
    # plt.title(r'Model SED for $Teff = ' + str(t_eff) + '\, K$')
    # plt.plot(wavelength, flux, 'ko-', markersize = 1)
    # plt.xlim(0, 30000)
    # plt.xlabel("Wavelength [$\AA$]")
    # plt.ylabel("Flux [erg/cm$^2$/A/s]")
    # plt.legend(loc = 'best')
    # plt.show()

    # # Plot the SED zoomed in on the two emission lines
    # plt.figure(figsize = (25, 10))
    # plt.rc('font', size = 20)
    # plt.title(r'Model SED for $Teff = ' + str(t_eff) + '\, K$')
    # plt.plot(wavelength, flux, 'ko-', markersize = 3)
    # plt.vlines(6560, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'red',    linestyles = '--', linewidth = 3, label = r'H$\alpha$    (656nm)')
    # plt.vlines(6430, ymin = np.nanmin(flux), ymax = np.nanmax(flux), colors = 'orange', linestyles = '--', linewidth = 3, label = 'Cont. (643nm)')
    # plt.xlim(6000, 7000)
    # plt.xlabel("Wavelength [$\AA$]")
    # plt.ylabel("Flux [erg/cm$^2$/A/s]")
    # plt.legend(loc = 'best')
    # plt.show()

    # Break when no flux data is available
    if (t_eff == 31000):
        break

# Convert to arrays
model_ratios = np.array(model_ratios)
temperature =  np.array(temperature)
r = interp1d(temperature, model_ratios)

tick_locations = [30000/1000, 25000/1000, 10000/1000, 7500/1000, 6000/1000, 5200/1000, 3700/1000]
tick_labels =    ['O', 'B', 'A', 'F', 'G', 'K', 'M']

fig = plt.figure(figsize = (25, 15))
plt.rc('font', size = 60)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(temperature/1000, model_ratios, 'r-', linewidth = 3)
ax2.set_xbound(ax1.get_xbound())
ax1.set_xlabel(r'Effective Temperature $[10^3 \,K]$')
ax1.set_ylabel(r'Quiescent H$\alpha$/Cont. Ratio')
ax1.set_xlim(ax1.get_xlim()[::-1])
ax2.set_xlim(ax2.get_xlim()[::-1])
ax1.set_xticks(tick_locations)
ax2.set_xticks(tick_locations)
ax2.set_xticklabels(tick_labels)
ax2.set_xlabel(r'Spectral Class [$log_{10}(g)=+3.0$, $[M/H]=+0.2$]')
plt.tight_layout()
plt.savefig(dir + 'ratio_vs_temp.jpg', dpi = N, bbox_inches = 'tight', rasterize = False)
plt.show()