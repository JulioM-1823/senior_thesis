# Import all the toys
!pip install photutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import and define plotting style
import seaborn as sb
plt.rcParams['font.family']    = 'monospace'
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
sb.set_context("talk")

from cmath import sqrt
from pandas import DataFrame, read_csv
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
from google.colab import drive
from warnings import simplefilter, filterwarnings
from os import path, mkdir, rename
drive.mount('/content/drive')
simplefilter('ignore', category = AstropyWarning)
filterwarnings('ignore')
%matplotlib inline

from astropy.modeling.models import Moffat1D
s1 = Moffat1D()
r = np.linspace(-10,10,1000)

f = plt.figure(figsize=(25,10))
ax = f.add_subplot(121)

ax.plot(r, s1(r), color = 'blue')
ax.hlines(0.8 , xmin = -10, xmax = 10, colors = 'red', linestyles = '--', linewidth = 3)
ax.set_ylim(0, 1)

ax.set_ylabel('Flux (counts)')
ax.set_xlabel('Radius (pixels)')

f.canvas.draw()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = '10'
labels[2] = '7.5'
labels[3] = '5'
labels[4] = '2.5'

ax.set_xticklabels(labels)

labels_y = [item.get_text() for item in ax.get_yticklabels()]
labels_y[1] = ''
labels_y[2] = ''
labels_y[3] = ''
labels_y[4] = 'Sat. Limit (16,000)'
labels_y[5] = 'True Max'

ax.set_yticklabels(labels_y)

ax2 = f.add_subplot(122)

ax2.plot(r, s1(r), color = 'blue')
ax2.set_ylim(0, 0.8)


ax2.set_xlabel('Radius (pixels)')

f.canvas.draw()

labels = [item.get_text() for item in ax2.get_xticklabels()]
labels[1] = '10'
labels[2] = '7.5'
labels[3] = '5'
labels[4] = '2.5'

ax2.set_xticklabels(labels)

labels_y = [item.get_text() for item in ax2.get_yticklabels()]
labels_y[0] = '0'
labels_y[1] = ''
labels_y[2] = ''
labels_y[3] = ''
labels_y[4] = ''
labels_y[5] = ''
labels_y[6] = ''
labels_y[7] = ''
labels_y[8] = 'Sat. Limit (16,000)'

ax2.set_yticklabels(labels_y)


f.tight_layout()
plt.savefig('/content/drive/My Drive/Julios Thesis Codes/Data/saturated_radial_profile.jpg', dpi = 400, bbox_inches = 'tight', rasterize = False)
plt.show()

from astropy.modeling.models import Moffat1D
s1 = Moffat1D()
r = np.linspace(-10,10,1000)

f = plt.figure(figsize=(25, 5))
ax = f.add_subplot(121)


# ax.grid()
ax.plot(r, s1(0.7*r), color = 'blue', label = 'Stellar PSF (high seeing)')
ax.plot(r, s1(4.5*r), color = 'green', linestyle = 'dashed', label = 'Stellar PSF (low seeing)')
ax.plot(r+6, 0.2*s1(4.5*r), color = 'red', label = 'Ghost PSF')
ax.set_ylim(0, 1)
ax.set_xlim(-8, 10)

ax.set_ylabel('Flux (counts)')
ax.set_xlabel('Radius from Ghost Centroid (pixels)')

f.canvas.draw()

labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = '350'
labels[1] = '300'
labels[2] = '250'
labels[3] = '200'
labels[4] = '150'
labels[5] = '100'
labels[6] = '50'
labels[7] = '0'
labels[8] = '50'
labels[9] = '100'

ax.set_xticklabels(labels)

labels_y = [item.get_text() for item in ax.get_yticklabels()]
labels_y[1] = '3,200'
labels_y[2] = '6,400'
labels_y[3] = '9,600'
labels_y[4] = '12,800'
labels_y[5] = '16,000'

ax.set_yticklabels(labels_y)
ax.legend(loc = 'best')
f.tight_layout()
plt.savefig('/content/drive/My Drive/Julios Thesis Codes/Data/ghost_psf_with_central_wings.jpg', dpi = 400, bbox_inches = 'tight', rasterize = False)
plt.show()