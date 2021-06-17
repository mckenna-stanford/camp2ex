import glob
from PIL import Image

files = glob.glob('flight_aerosol_*v3.png')
files = sorted(files)

pdf1_filename = '/home/mwstanfo/camp2ex/figures/flight_aerosol_advanced_cloud_flag_combined.pdf'

im1 = Image.open(files[0])
rgb = Image.new('RGB',im1.size, (255,255,255))
rgb.paste(im1,mask=im1.split()[3])

im_list = []
for file in files:
	tmpim = Image.open(file)
	tmprgb = Image.new('RGB',tmpim.size,(255,255,255))
	tmprgb.paste(tmpim,mask=tmpim.split()[3])
	im_list.append(tmprgb)

rgb.save(pdf1_filename, "PDF" ,resolution=100.0, save_all=True, append_images=im_list)

