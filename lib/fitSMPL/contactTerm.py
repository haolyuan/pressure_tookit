import torch.nn as nn
import numpy as np
import torch

class ContactTerm(nn.Module):
    def __init__(self,
                 dtype=np.float32,
                 device='cpu') -> None:
        super(ContactTerm, self).__init__()
        self.dtype = dtype
        self.device = device
        
        self.RegionInsole2SMPLL = np.load('essentials/pressure/RegionInsole2SMPLL.npy', allow_pickle=True).item()
        self.RegionInsole2SMPLR = np.load('essentials/pressure/RegionInsole2SMPLR.npy', allow_pickle=True).item()
        # _enhanced
        
    def contact2smpl(self, contact_data):
        # contact_data np.array. shape is [2, 9]
        cont_smpl_idL = self.region2verts_id(contact_data[0,...],
                                            self.RegionInsole2SMPLL['smpl'])
        cont_smpl_idR = self.region2verts_id(contact_data[1,...],
                                            self.RegionInsole2SMPLR['smpl'])
        cont_smpl_id = cont_smpl_idL + cont_smpl_idR
        
        return torch.tensor(cont_smpl_id, device=self.device).long() ,\
                torch.tensor(cont_smpl_idL, device=self.device).long() ,\
                torch.tensor(cont_smpl_idR, device=self.device).long()
        
    def region2verts_id(self,contact_label, insoleRegion):
        contact_label = np.insert(contact_label,[2],contact_label[1])
        region_name = list(insoleRegion.keys())
        contact_ids = []
        for ci in range(contact_label.shape[0]):
            region_label = contact_label[ci]
            if region_label == 1:
                contact_ids += insoleRegion[region_name[ci]].tolist()
        return contact_ids    