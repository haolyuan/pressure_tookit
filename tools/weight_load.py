import glob
import os.path as osp
import pickle
import numpy as np
import argparse

BASE_DIR = 'D:/dataset/tebu_contact'

def loadPressureData(sub_ids,seq_name):

    insole_sync = np.load(osp.join(BASE_DIR,'Sync-list-total.npy'), allow_pickle=True).item()

    '''load insole data'''

    insole_name = insole_sync[sub_ids][seq_name]
    insole_path = osp.join(BASE_DIR, 'insole_pkl', insole_name + '.pkl')
    with open(insole_path, "rb") as f:
        insole_data = pickle.load(f)
    indice_name = osp.join(BASE_DIR, 'Synced_indice', insole_name + '*')
    synced_indice = np.loadtxt(glob.glob(indice_name)[0]).astype(np.int32)
    insole_data = insole_data[synced_indice]
    return insole_data

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_ids',
                        default='S1')
    parser.add_argument('--seq_name',
                        default='MoCap_20230422_092324')
    parser.add_argument('--frame_idx',
                        default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    
    
    insole_data = loadPressureData(args.sub_ids, args.seq_name)
    press_data = insole_data[args.frame_idx]
    print(press_data[0].shape, np.sum(press_data[0]))
    