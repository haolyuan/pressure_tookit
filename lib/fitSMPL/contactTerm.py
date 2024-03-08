import torch.nn as nn
import numpy as np
import torch

from lib.Utils.refineSMPL_utils import compute_normal_batch

class ContactTerm(nn.Module):
    def __init__(self,
                 essential_root=None,
                 dtype=torch.float32,
                 device='cpu') -> None:
        super(ContactTerm, self).__init__()
        self.dtype = dtype
        self.device = device
        
        self.RegionInsole2SMPLL = np.load(f'{essential_root}/pressure/RegionInsole2SMPLL_enhanced.npy', allow_pickle=True).item()
        self.RegionInsole2SMPLR = np.load(f'{essential_root}/pressure/RegionInsole2SMPLR_enhanced.npy', allow_pickle=True).item()
        
        self.foot_region9_l, self.foot_region9_r = self.foot_region_load()
        
        # TODO: add aver norm direction in future version
        
        # set pre plane position to fix foot
        self.pre_plane = None
        self.pre_plane_ids_back_l, self.pre_plane_ids_front_l, self.pre_plane_ids_back_r, self.pre_plane_ids_front_r =\
            None, None, None, None
        
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

    def update_foot_plane(self, foot_plane,
                        contact_data,
                        foot_plane_ids_smplL,
                        foot_plane_ids_smplR):
        """_summary_

        Args:
            foot_plane (torch.tensor([bs, 30+42+30+42, 3])): foot plane
        """        
        self.pre_plane = foot_plane.detach()
        
        pre_plane_ids = self.get_plane_idx(contact_data=contact_data,
                                       foot_plane_ids_smplL=foot_plane_ids_smplL,
                                       foot_plane_ids_smplR=foot_plane_ids_smplR)
        self.pre_plane_ids_back_l, self.pre_plane_ids_front_l,\
            self.pre_plane_ids_back_r, self.pre_plane_ids_front_r = \
                pre_plane_ids[0], pre_plane_ids[1], pre_plane_ids[2], pre_plane_ids[3]

    def calcPenetrateLoss(self, live_verts, contact_ids):
        """_summary_

        Args:
            live_verts (torch.tensor([bs, 6890, 3])): _description_
            contact_ids (torch.tensor([num_contact])): foot contact idx in smpl 

        Returns:
            _type_: _description_
        """        
        return torch.mean(torch.abs(live_verts[0, contact_ids, 1]))
    
    def calcTempLoss(self,
                     live_plane,
                     contact_data,
                     foot_plane_ids_smplL,
                     foot_plane_ids_smplR):
        """temp loss to prevent skating foot

        Args:
            foot_plane (torch.tensor([bs, 30+42+30+42, 3])): pose foot plane
            contact_data (list, (2,9)): contact label
        """

        curr_plane_back_l, curr_plane_front_l,\
            curr_plane_back_r, curr_plane_front_r =\
                live_plane[:, :30, :].detach(),\
                live_plane[:, 30:30+42, :].detach(), \
                live_plane[:, 30+42:30+42+30, :].detach(), \
                live_plane[:, 30+42+30:, :].detach()

        curr_plane_ids = self.get_plane_idx(contact_data=contact_data,
                                       foot_plane_ids_smplL=foot_plane_ids_smplL,
                                       foot_plane_ids_smplR=foot_plane_ids_smplR)
        curr_plane_ids_back_l, curr_plane_ids_front_l,\
            curr_plane_ids_back_r, curr_plane_ids_front_r = \
                curr_plane_ids[0], curr_plane_ids[1], curr_plane_ids[2], curr_plane_ids[3]    
        # four foot parts loss
        transl_loss_dict = {}

        if len(curr_plane_ids_back_l) != 0:
            inter_ids = list(set(curr_plane_ids_back_l).\
                intersection(set(self.pre_plane_ids_back_l)))
            
            if len(inter_ids) != 0:
                temp_offset = live_plane[:, inter_ids, :] -\
                        self.pre_plane[:, inter_ids, :]
                transl_loss_dict['back_l'] =\
                    torch.mean(torch.linalg.norm(temp_offset, dim=2))
            else:
                transl_loss_dict['back_l'] = 0
        else:
            transl_loss_dict['back_l'] = 0

        if len(curr_plane_ids_front_l) != 0:
            inter_ids = list(set(curr_plane_ids_front_l).\
                intersection(set(self.pre_plane_ids_front_l)))
            if len(inter_ids) != 0:
                temp_offset = live_plane[:, inter_ids, :] -\
                        self.pre_plane[:, inter_ids, :]
                transl_loss_dict['front_l'] =\
                    torch.mean(torch.linalg.norm(temp_offset, dim=2))
            else:
                transl_loss_dict['front_l'] = 0
        else:
            transl_loss_dict['front_l'] = 0

        if len(curr_plane_ids_back_r) != 0:
            inter_ids = list(set(curr_plane_ids_back_r).\
                intersection(set(self.pre_plane_ids_back_r)))
            if len(inter_ids) != 0:
                temp_offset = live_plane[:, inter_ids, :] -\
                        self.pre_plane[:, inter_ids, :]
                transl_loss_dict['back_r'] =\
                    torch.mean(torch.linalg.norm(temp_offset, dim=2))
            else:
                transl_loss_dict['back_r'] = 0
        else:
            transl_loss_dict['back_r'] = 0

        if len(curr_plane_ids_front_r) != 0:
            inter_ids = list(set(curr_plane_ids_front_r).\
                intersection(set(self.pre_plane_ids_front_r)))
            if len(inter_ids) != 0:
                temp_offset = live_plane[:, inter_ids, :] -\
                        self.pre_plane[:, inter_ids, :]
                transl_loss_dict['front_r'] =\
                    torch.mean(torch.linalg.norm(temp_offset, dim=2))
            else:
                transl_loss_dict['front_r'] = 0
        else:
            transl_loss_dict['front_r'] = 0
                    
        foot_temp_loss = transl_loss_dict['back_l'] + transl_loss_dict['front_l'] +\
                transl_loss_dict['back_r'] + transl_loss_dict['front_r']    
        return foot_temp_loss

    def get_plane_idx(self,contact_data,
                     foot_plane_ids_smplL,
                     foot_plane_ids_smplR):

        contact_label_l, contact_label_r =\
                    contact_data[0], contact_data[1]

        # init foot plane ids, seperate front and back
        foot_ids_back_l, foot_ids_front_l = foot_plane_ids_smplL[0], foot_plane_ids_smplL[1]
        foot_ids_back_r, foot_ids_front_r = foot_plane_ids_smplR[0], foot_plane_ids_smplR[1]
        
        plane_ids_back_l, plane_ids_front_l, plane_ids_back_r, plane_ids_front_r = \
                    [], [], [], []
        if not all(label == 0 for label in contact_label_l):
            contact_region_l = [i for i, x in enumerate(contact_label_l) if x==1]
            # get contact smpl idx from the default 9 foot region 
            contact_footl_ids = [self.foot_region9_l[i] for i in contact_region_l]
            contact_footl_ids = np.hstack(contact_footl_ids)           
            # get contact plane idx according to the corres between contact_smpl_idx and 4 default smpl plane idx 
            # because we must use contact plane idx to search contact points position
            plane_ids_back_l = [foot_ids_back_l.index(i) for i in foot_ids_back_l if i in contact_footl_ids]
            plane_ids_front_l = [foot_ids_front_l.index(i) + len(foot_ids_back_l)\
                for i in foot_ids_front_l if i in contact_footl_ids]       
                 
        if not all(label == 0 for label in contact_label_r):
            contact_region_r = [i for i, x in enumerate(contact_label_r) if x==1]
            # get contact smpl idx from the default 9 foot region 
            contact_footr_ids = [self.foot_region9_r[i] for i in contact_region_r]
            contact_footr_ids = np.hstack(contact_footr_ids)

            plane_ids_back_r = [foot_ids_back_r.index(i) + len(foot_ids_back_l) + len(foot_ids_front_l)\
                for i in foot_ids_back_r if i in contact_footr_ids]
            plane_ids_front_r = [foot_ids_front_r.index(i) + len(foot_ids_back_l) * 2 + len(foot_ids_front_l)\
                for i in foot_ids_front_r if i in contact_footr_ids]        

        return [plane_ids_back_l, plane_ids_front_l, plane_ids_back_r, plane_ids_front_r]
        
    def foot_region_load(self):
        foot_region_keys = list(self.RegionInsole2SMPLR['smpl'].keys())
        foot_region9_l, foot_region9_r = [], []
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[0]])
        foot_region9_l.append(np.concatenate(
            [self.RegionInsole2SMPLL['smpl'][foot_region_keys[1]], self.RegionInsole2SMPLL['smpl'][foot_region_keys[2]]]))
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[3]])
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[4]])
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[5]])
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[6]])
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[7]])
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[8]])
        foot_region9_l.append(self.RegionInsole2SMPLL['smpl'][foot_region_keys[9]])
        
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[0]])
        foot_region9_r.append(np.concatenate(
            [self.RegionInsole2SMPLR['smpl'][foot_region_keys[1]], self.RegionInsole2SMPLR['smpl'][foot_region_keys[2]]]))
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[3]])
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[4]])
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[5]])
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[6]])
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[7]])
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[8]])
        foot_region9_r.append(self.RegionInsole2SMPLR['smpl'][foot_region_keys[9]])
        
        return foot_region9_l, foot_region9_r        
    
    def calcNormLoss(self, source_verts, verts_faces_list, target_n_list):
        
        n_y_nega = torch.tensor(np.array([0, -1, 0]), dtype=self.dtype, device=self.device)

        source_n_back_l = compute_normal_batch(vertices=source_verts[:, :30], faces=verts_faces_list[0].to(self.device)) 
        source_n_front_l = compute_normal_batch(vertices=source_verts[:, 30:30+42], faces=verts_faces_list[2].to(self.device))
        source_n_back_r = compute_normal_batch(vertices=source_verts[:, 30+42:30*2+42], faces=verts_faces_list[1].to(self.device))
        source_n_front_r = compute_normal_batch(vertices=source_verts[:, 30*2+42:], faces=verts_faces_list[3].to(self.device)) 
        

        # all norm should at positive y axis
        for i in range(source_n_back_l.shape[1]):
            if torch.dot(source_n_back_l[:, i, :].squeeze(0), n_y_nega) > 0:
                source_n_back_l[:, i, :] *= -1    
            if torch.dot(source_n_front_l[:, i, :].squeeze(0), n_y_nega) > 0:
                source_n_front_l[:, i, :] *= -1      

        for i in range(source_n_back_r.shape[1]):
            if torch.dot(source_n_back_r[:, i, :].squeeze(0), n_y_nega) > 0:
                source_n_back_r[:, i, :] *= -1    
            if torch.dot(source_n_front_r[:, i, :].squeeze(0), n_y_nega) > 0:
                source_n_front_r[:, i, :] *= -1      
                          
        source_n_aver_back_l = torch.mean(source_n_back_l, dim=1)
        source_n_aver_back_r = torch.mean(source_n_back_r, dim=1)
        source_n_aver_front_l = torch.mean(source_n_front_l, dim=1)
        source_n_aver_front_r = torch.mean(source_n_front_r, dim=1)
                  
        source_n_aver_norm_back_l = source_n_aver_back_l / \
                    torch.linalg.norm(source_n_aver_back_l)  
        source_n_aver_norm_back_r = source_n_aver_back_r / \
                    torch.linalg.norm(source_n_aver_back_r)
        source_n_aver_norm_front_l = source_n_aver_front_l / \
                    torch.linalg.norm(source_n_aver_front_l)
        source_n_aver_norm_front_r = source_n_aver_front_r / \
                    torch.linalg.norm(source_n_aver_front_r) 
        
        loss_back_l = 1 - torch.dot(source_n_aver_norm_back_l[0], target_n_list[0].to(self.device).to(self.dtype))
        loss_back_r = 1 - torch.dot(source_n_aver_norm_back_r[0], target_n_list[1].to(self.device).to(self.dtype))
        loss_front_l = 1 - torch.dot(source_n_aver_norm_front_l[0], target_n_list[2].to(self.device).to(self.dtype))
        loss_front_r = 1 - torch.dot(source_n_aver_norm_front_r[0], target_n_list[3].to(self.device).to(self.dtype))

        return loss_back_l + loss_back_r + loss_front_l + loss_front_r
        
        