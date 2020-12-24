import ROOT as rt
import numpy as np
import h5py
import math
import sys
import argparse
# npe will be save one dataframe for each r,theta
def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce training samples for JUNO study. ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch_size', action='store', type=int, default=5000,
                        help='Number of event for each batch.')
    parser.add_argument('--input', action='store', type=str, default='',
                        help='input root file.')
    parser.add_argument('--output', action='store', type=str, default='',
                        help='output hdf5 file.')
    parser.add_argument('--r_scale', action='store', type=float, default=17700,
                        help='r normalization')
    parser.add_argument('--theta_scale', action='store', type=float, default=180,
                        help='theta scale.')

    return parser

def root2hdf5 (batch_size, tree, start_event, out_name, id_dict):

    hf = h5py.File(out_name, 'w')
    df = np.full( ( batch_size,len(id_dict) ), 0, np.float32)
    for ie in range(start_event, start_event+batch_size):
        tree.GetEntry(ie)
        tmp_dict = {}
        init_x    = getattr(tree, "init_x")
        init_y    = getattr(tree, "init_y")
        init_z    = getattr(tree, "init_z")
        pmtID     = getattr(tree, "pmt_id")
        hittime   = getattr(tree, "pmt_hit_time")
        init_r    = math.sqrt(init_x*init_x + init_y*init_y + init_z*init_z)
        for i in range(0, len(pmtID)):
            ID     = pmtID[i]
            if ID not in id_dict:continue
            if ID not in tmp_dict:
                tmp_dict[ID] = 1
            else:
                tmp_dict[ID] = tmp_dict[ID] + 1
        for i in range(len(id_dict)):
            df[ie-start_event, i] = tmp_dict[i] if i in tmp_dict else 0
         
    hf.create_dataset('data', data=df)
    hf.close()
    print('saved %s'%out_name)




def get_pmt_theta_phi(file_pos, sep, i_id, i_theta, i_phi):
    id_dict = {}
    theta_list = []
    phi_list = []
    f = open(file_pos,'r')
    lines = f.readlines()
    for line in lines:
        items = line.split()
        ID    = float(items[i_id])
        ID    = int(ID)
        theta = float(items[i_theta])
        phi   = float(items[i_phi])
        #phi   = int(phi) ## otherwise it will be too much
        if theta not in theta_list:
            theta_list.append(theta)
        if phi not in phi_list:
            phi_list.append(phi)
        if ID not in id_dict:
            id_dict[ID]=[theta, phi]
    return (id_dict, theta_list, phi_list)



if __name__ == '__main__':

    large_PMT_pos = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/PMT_pos/J20v1r0-Pre2/PMTPos_Acrylic_with_chimney.csv' 
    (L_id_dict, L_theta_list, L_phi_list) = get_pmt_theta_phi(large_PMT_pos, '', 0, 4, 5)
    print('L theta size=',len(L_theta_list),',L phi size=',len(L_phi_list))
    small_PMT_pos = '/junofs/users/wxfang/FastSim/GAN/JUNO/StepSim/PMT_pos/3inch_pos.csv' 
    (S_id_dict, S_theta_list, S_phi_list) = get_pmt_theta_phi(small_PMT_pos, '', 0, 1, 2)
    print('S theta size=',len(S_theta_list),',S phi size=',len(S_phi_list))
    ###########################################################
    parser = get_parser()
    parse_args = parser.parse_args()

    batch_size = parse_args.batch_size
    filePath = parse_args.input
    outFileName= parse_args.output
    r_scale = parse_args.r_scale
    theta_scale = parse_args.theta_scale

    treeName='evt'
    chain =rt.TChain(treeName)
    chain.Add(filePath)
    tree = chain
    totalEntries=tree.GetEntries()
    batch = int(float(totalEntries)/batch_size)
    print ('total events=%d, batch_size=%d, batchs=%d, last=%d'%(totalEntries, batch_size, batch, totalEntries%batch_size))
    start = 0
    for i in range(batch):
        out_name = outFileName.replace('.h5','_batch%d_N%d.h5'%(i, batch_size))
        root2hdf5 (batch_size, tree, start, out_name, L_id_dict)
        start = start + batch_size
    print('done')  
