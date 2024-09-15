import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.datasets import load_star_image
from photutils.detection import DAOStarFinder
from photutils.photometry import aperture_photometry
from photutils.aperture import CircularAperture
import matplotlib.pyplot as plt

# Load the star image
hdu = load_star_image()
data = hdu.data.astype(float)

# Estimate background and noise
bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(data, sigma=3.0)

# Subtract background
data -= bkg_median

# Detect stars
daofind = DAOStarFinder(fwhm=4.0, threshold=3.0 * bkg_std)
sources = daofind(data)

# Define circular apertures
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)

# Perform photometry
phot_table = aperture_photometry(data, apertures)

# Calculate FWHM
fwhm = sources['fwhm']

# Convert FWHM to diameter (assuming 1 arcsecond per pixel)
diameter_arcsec = fwhm * 1.0  # 1 arcsecond per pixel

print("Star diameters:")
for i, dia in enumerate(diameter_arcsec):
    print(f"Star {i+1}: {dia:.2f} arcseconds")

# Plot the results
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='gray', origin='lower')
apertures.plot(color='red', lw=1.5, alpha=0.5)
plt.title('Stars with Calculated Diameters')
plt.colorbar(label='Intensity')
plt.show()