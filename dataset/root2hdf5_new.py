import ROOT as rt
import numpy as np
import h5py
import math
import sys
import argparse
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce training samples for JUNO background remove. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch_size', action='store', type=int, default=5000,
                        help='Number of event for each batch.')
    parser.add_argument('--input', action='store', type=str, 
    					default="/cefs/higgs/wxfang/JUNO/noise_remove/positron_skim/*.root",
                        help='input root file.')
    parser.add_argument('--output', action='store', type=str, 
    					default='pos/positron_skim_dataset.h5',
                        help='output hdf5 file.')
    parser.add_argument('--r_scale', action='store', type=float, default=17700,
                        help='r normalization')
    parser.add_argument('--theta_scale', action='store', type=float, default=180,
                        help='theta scale.')
    parser.add_argument('--projection_mode', action='store', type=int, default=1,
    					help='1: theta-phi; 2:vertexRec')	

    return parser

def getProjectionDict(mode):
	if mode==1:
		file="/junofs/users/yuansc/dataset/id2pos_interval.txt"
	else:
		file="/junofs/users/yuansc/dataset/id2pos_theta-phi_256-256.txt"

	projection_dict={}
	with open(file, "r") as f:
		for line in f.readlines():
			line=[int(x) for x in line.split(',')]
			projection_dict[line[0]]=[line[1], line[2]]

	shape=[]
	shape.append(max([v[0] for v in projection_dict.values()]))
	shape.append(max([v[1] for v in projection_dict.values()]))

	return projection_dict, shape

def root2hdf5(batch_size, tree, start_event, out_name, projection_dict, projection_dict_shape, 
				npe_cut=0):
	hf = h5py.File(out_name, 'w')
	df = np.full( ( batch_size,projection_dict_shape[0], projection_dict_shape[1] ), 0, np.float32)
	ie = start_event
	n = 0
	global totalEntries
	while n < batch_size:
		if ie == totalEntries-1:
			return False
		tree.GetEntry(ie)
		ie += 1
		pmt_id=getattr(tree, "pmt_id")
		#cut npe
		if len(list(pmt_id)) <= npe_cut:
			continue
		for i in pmt_id:
			try:
				pos=projection_dict[i]
				df[n,projection_dict[i][0],projection_dict[i][1]]+=1
			except:
				pass
		n += 1
		#array2img(df[ie - start_event]).show()
		#print(df[ie - start_event].max())

	hf.create_dataset('data', data=df)
	hf.close()
	print('saved %s'%out_name)

	return ie

def array2img(array):
	array_max = array.max()
	array = array * 255 / array_max
	#print(array.max())
	img = Image.fromarray(array)
	return img


if __name__=="__main__":
	parser = get_parser()
	parse_args = parser.parse_args()

	batch_size = parse_args.batch_size
	filePath = parse_args.input
	outFileName= parse_args.output
	r_scale = parse_args.r_scale
	theta_scale = parse_args.theta_scale
	projection_mode = parse_args.projection_mode

	filePath = '/cefs/higgs/wxfang/JUNO/noise_remove/Acrylic_Th232/*.root'
	outFileName = 'noi/Acrylic_Th232/Acrylic_Th232_dataset.h5'

	projection_dict, projection_dict_shape=getProjectionDict(projection_mode)

	treeName='evt'
	chain =rt.TChain(treeName)
	chain.Add(filePath)
	tree = chain
	totalEntries=tree.GetEntries()
	batch = int(float(totalEntries)/batch_size)
	print ('total events=%d, batch_size=%d, batchs=%d, last=%d'%(totalEntries, batch_size, batch, totalEntries%batch_size))

	start=1
	npe_cut = 100
	
	for i in range(batch):
	#for i in range(1):
		if start:
			out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i, batch_size))
			print("creating dataset "+out_name)
			start = root2hdf5 (batch_size, tree, start, out_name, 
				projection_dict, projection_dict_shape, npe_cut=npe_cut)
			print('start=', start)
		else:
			break
	print('done')




