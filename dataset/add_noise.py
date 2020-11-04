import numpy as np
import h5py
import math
import sys
import argparse
from PIL import Image
import os

def get_parser():
	parser = argparse.ArgumentParser(
		description='Positron signal + noise.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('--batch_size', action='store', type=int, default=5000,
		help='Number of event for each batch.')
	parser.add_argument('--signal', action='store', type=str, default="pos/",
		help='Positron signal.')
	parser.add_argument('--noise', action='store', type=str, default="noi/u238/",
		help='noise signal.')
	parser.add_argument('--output', action='store', type=str, 
		default="pos+noi/u238/positron+u238_skim_dataset.h5",
		help='noise signal.')
	return parser

if __name__ == "__main__":

	parser = get_parser()
	parse_args = parser.parse_args()

	batch_size = parse_args.batch_size
	signal = parse_args.signal
	noise = parse_args.noise
	output = parse_args.output

	noise = "noi/Acrylic_Th232/"
	output = "pos+noi/Acrylic_Th232/positron+th232_dataset.h5"

	sig_list = [signal + x for x in os.listdir(signal) if ".h5" in x]
	noi_list = [noise + x for x in os.listdir(noise) if ".h5" in x]

	print(len(sig_list),len(noi_list))

	for i in range(min(len(sig_list), len(noi_list))):
		sig_arr = h5py.File(sig_list[i], 'r')['data'][:]
		noi_arr = h5py.File(noi_list[i], 'r')['data'][:]
		sig_and_noi_arr = sig_arr + noi_arr

		out_name = output.replace('.h5','_batch%d_N%d.h5'%(i, batch_size))
		hf = h5py.File(out_name, 'w')
		hf.create_dataset('data', data=sig_and_noi_arr)
		hf.close()
		print("saved "+ out_name)
