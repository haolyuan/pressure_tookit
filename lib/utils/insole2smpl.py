import sys
import cv2,pickle
import numpy as np
import trimesh,glob
import os.path as osp
from tqdm import tqdm,trange
from icecream import ic

sys.path.append('E:/projects/pressure_toolkit')
from lib.dataset.PressureDataset import PressureDataset
from color_utils import rgb_code
from lib.utils.fileio import read_json,save_json,saveImgSeqAsvideo
from lib.dataset.insole_sync import insole_sync

def mapInsole2Smpl():
    footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
    footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
    masksmplL = np.loadtxt('essentials/pressure/masksmplL.txt').astype(np.int32)
    masksmplR = np.loadtxt('essentials/pressure/masksmplR.txt').astype(np.int32)
    # a = np.arange(0,33*11,dtype=np.int32).reshape([33,11])

    lsR = {}
    for ids in footIdsR:
        tmp = np.stack(np.where(masksmplR == ids))
        lsR[str(ids)] = tmp
    np.save('debug/insole2smplR.npy', lsR)

    lsL = {}
    for ids in footIdsL:
        tmp = np.stack(np.where(masksmplL == ids))
        lsL[str(ids)] = tmp
    np.save('debug/insole2smplL.npy', lsL)
    exit()


def drawSMPLFoot(v_foot,footIds,faces,img_H=3300,img_W=1100,
                 vert_color=None,save_name=None,
                 point_size=40):
    tex_color = rgb_code['DarkCyan']
    line_color = rgb_code['Green']
    x_col = img_W-(v_foot[:,0]-np.min(v_foot[:,0]))/(np.max(v_foot[:,0])-np.min(v_foot[:,0]))*(img_W-1)-1
    x_row = img_H-(v_foot[:,2]-np.min(v_foot[:,2]))/(np.max(v_foot[:,2])-np.min(v_foot[:,2]))*(img_H-1)-1

    img = np.ones(((img_H+50),(img_W+100),3), dtype=np.uint8) * 255
    point = np.concatenate([x_row.reshape([-1,1]).astype(np.int32),x_col.reshape([-1,1])],axis=1)

    for j in range(faces.shape[0]):
        x,y,z = faces[j]
        if x in footIds and y in footIds:
            xi = np.where(footIds==x)[0]
            yi = np.where(footIds==y)[0]
            img = cv2.line(img, (int(point[xi, 1]), int(point[xi, 0])),
                           (int(point[yi, 1]), int(point[yi, 0])),
                           (line_color[2], line_color[1], line_color[0]), 2)
        if z in footIds and y in footIds:
            zi = np.where(footIds == z)[0]
            yi = np.where(footIds == y)[0]
            img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                           (int(point[yi, 1]), int(point[yi, 0])),
                           (line_color[2], line_color[1], line_color[0]), 2)
        if z in footIds and x in footIds:
            zi = np.where(footIds == z)[0]
            xi = np.where(footIds == x)[0]
            img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                           (int(point[xi, 1]), int(point[xi, 0])),
                           (line_color[2], line_color[1], line_color[0]), 2)

    for i in range(point.shape[0]):
        x,y = point[i, 0],point[i, 1]
        if vert_color is None:
            v_color = rgb_code['Black']
        else:
            v_color = vert_color[i,::-1]
        img = cv2.circle(img, (int(y), int(x)), point_size, (int(v_color[2]), int(v_color[1]), int(v_color[0])), -1)
        img = cv2.putText(img, f"{footIds[i]}", (int(y), int(x)+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (tex_color[2],tex_color[1],tex_color[0]))

    if save_name is not None:
        cv2.imwrite(osp.join(save_dir, save_name), img)
    else:
        return img

def getVertsColor(pressure_img,footIds,insole2smpl):
    verts_color = np.zeros([footIds.shape[0],3])
    for i in range(footIds.shape[0]):
        ids = footIds[i]
        if str(ids) in insole2smpl.keys():
            tmp = insole2smpl[str(ids)]
            _color = pressure_img[tmp[0],tmp[1],:]
            if _color.shape[0] != 0:
                verts_color[i] = np.sum(_color,axis=0)
    return verts_color.astype(np.int32)

def loadInsoleData(basdir='E:/dataset/PressureDataset/20230422',
        sub_ids='S10',seq_name='MoCap_20230422_171535'):
    insole_name = insole_sync[sub_ids][seq_name]

    insole_path = osp.join(basdir, 'insole_pkl', insole_name + '.pkl')
    with open(insole_path, "rb") as f:
        insole_data = pickle.load(f)
    indice_name = osp.join(basdir, 'Synced_indice', insole_name + '*')
    synced_indice = np.loadtxt(glob.glob(indice_name)[0]).astype(np.int32)
    insole_data = insole_data[synced_indice]
    return insole_data

def show_insole(idx,insole_data):
    data = insole_data[idx]
    press_dim,rows,cols = data.shape
    img = np.ones((rows, cols * 2), dtype=np.uint8)
    imgL = np.uint8(data[0] * 5)
    imgR = np.uint8(data[1] * 5)
    img[:, :imgL.shape[1]] = imgL
    img[:, img.shape[1] - imgR.shape[1]:] = imgR
    imgLarge = cv2.resize(img, (cols * 10 * 2, rows * 10))
    imgColor = cv2.applyColorMap(imgLarge, cv2.COLORMAP_HOT)
    # imgColor = cv2.applyColorMap(imgLarge, cv2.COLORMAP_JET)
    cv2.imshow("img", imgColor)
    cv2.waitKey(1)
    return img

def footRegionMapping():
    '''
    :return: (row_num,col_num)
    '''
    insoleRegionR={}
    insoleMaskR = np.loadtxt('E:\dataset\PressureDataset\insole_mask/insoleMaskR.txt').astype(np.int32)
    ic(insoleMaskR.shape)

    BigToe = np.array(np.where((insoleMaskR[:6,:5]>0.5)))
    insoleRegionR['BigToe'] = BigToe

    MidToe = np.array(np.where((insoleMaskR[:6, 5:8] > 0.5)))
    MidToe[1,:] = MidToe[1,:]+5
    insoleRegionR['MidToe'] = MidToe

    SmallToe = np.array(np.where((insoleMaskR[:6, 8:] > 0.5)))
    SmallToe[1,:] = SmallToe[1,:]+8
    insoleRegionR['SmallToe'] = SmallToe

    Metatarsus0 = np.array(np.where((insoleMaskR[6:15, :4] > 0.5)))
    Metatarsus0[0,:] = Metatarsus0[0,:] + 6
    insoleRegionR['Front0'] = Metatarsus0

    Metatarsus1 = np.array(np.where((insoleMaskR[6:15, 4:8] > 0.5)))
    Metatarsus1[0,:] = Metatarsus1[0,:] + 6
    Metatarsus1[1,:] = Metatarsus1[1,:] + 4
    insoleRegionR['Front1'] = Metatarsus1

    Metatarsus2 = np.array(np.where((insoleMaskR[6:15, 8:] > 0.5)))
    Metatarsus2[0,:] = Metatarsus2[0,:] + 6
    Metatarsus2[1,:] = Metatarsus2[1,:] + 8
    insoleRegionR['Front2'] = Metatarsus2

    Cuboid0 = np.array(np.where((insoleMaskR[15:25, :8] > 0.5)))
    Cuboid0[0,:] = Cuboid0[0,:] + 15
    insoleRegionR['Mid0'] = Cuboid0

    Cuboid1 = np.array(np.where((insoleMaskR[15:25, 8:] > 0.5)))
    Cuboid1[0,:] = Cuboid1[0,:] + 15
    Cuboid1[1,:] = Cuboid1[1,:] + 8
    insoleRegionR['Mid1'] = Cuboid1

    Calcaneus0 = np.array(np.where((insoleMaskR[25:, :8] > 0.5)))
    Calcaneus0[0,:] = Calcaneus0[0,:] + 25
    insoleRegionR['Back0'] = Calcaneus0

    Calcaneus1 = np.array(np.where((insoleMaskR[25:, 8:] > 0.5)))
    Calcaneus1[0, :] = Calcaneus1[0, :] + 25
    Calcaneus1[1, :] = Calcaneus1[1, :] + 8
    insoleRegionR['Back1'] = Calcaneus1

    smplRegionR={}
    smplRegionR['BigToe'] = np.array([6637,6640,6636,6639,6693,6695,6697,6696])
    smplRegionR['MidToe'] = np.array([6621,6622,6654,6651,6653,6652,6696,6701,6707,6661,6663,6664,6660,6706,6705])
    smplRegionR['SmallToe'] = np.array([6630,6629,6674,6677,6675,6678,6707,6711,6716,6687,6690,6686,6689,6626,6625])
    smplRegionR['Front0'] = np.array([6752,6753,6806,6837,6755,6754])
    smplRegionR['Front1'] = np.array([6758,6759,6838,6839,6761,6760])
    smplRegionR['Front2'] = np.array([6762,6757,6763,6756,6840,6819])
    smplRegionR['Mid0'] = np.array([6807,6844,6843,6808,6848,6847,6830,6849,6850])
    smplRegionR['Mid1'] = np.array([6842,6841,6820,6846,6845,6821,6851,6852,6822])
    smplRegionR['Back0'] = np.array([6831,6856,6855,6829,6862,6863,6828,6861,6864,6867,6860,6827])
    smplRegionR['Back1'] = np.array([6854,6853,6823,6865,6857,6824,6866,6859,6826,6868,6858,6825])

    RegionInsole2SMPLR={'insole':insoleRegionR,
                        'smpl':smplRegionR}
    np.save('debug/insole2smpl/RegionInsole2SMPLR.npy', RegionInsole2SMPLR)
    exit()

def footRegionMappingRight2Left():
    footIdsLR = np.loadtxt('essentials/footLR_ids.txt').astype(np.int32)

    RegionInsole2SMPL = np.load('debug/insole2smpl/RegionInsole2SMPLR.npy', allow_pickle=True).item()
    insoleRegion = RegionInsole2SMPL['insole']
    smplRegion = RegionInsole2SMPL['smpl']

    all_idsR = []
    for _key in smplRegion.keys():
        all_idsR.append(smplRegion[_key])
        for ids in smplRegion[_key]:
            if ids not in footIdsLR[:,1]:
                ic(_key,ids)
    all_idsR = np.concatenate(all_idsR)
    for idx in footIdsLR[:,1]:
        if idx not in all_idsR:
            ic(idx)

    for _key in smplRegion.keys():
        for i in range(smplRegion[_key].shape[0]):
            Rids = smplRegion[_key][i]
            a = np.where(footIdsLR[:,1]==Rids)[0]
            smplRegion[_key][i] = footIdsLR[a,0]
    for _key in insoleRegion.keys():
        insoleRegion[_key][1,:] = 10-insoleRegion[_key][1,:]

    RegionInsole2SMPLL = {
        'insole':insoleRegion,
        'smpl':smplRegion
    }
    ic(RegionInsole2SMPLL['smpl'])
    ic(RegionInsole2SMPLL['insole'])
    np.save('debug/insole2smpl/RegionInsole2SMPLL.npy', RegionInsole2SMPLL)
    exit()

def visFootRegionMapping():
    from lib.utils.vis_foot import visFootImage
    m_vis = visFootImage()
    RegionInsole2SMPLR = np.load('debug/insole2smpl/RegionInsole2SMPLR.npy', allow_pickle=True).item()
    insoleRegionR,smplRegionR = RegionInsole2SMPLR['insole'],RegionInsole2SMPLR['smpl']

    RegionInsole2SMPLL = np.load('debug/insole2smpl/RegionInsole2SMPLL.npy', allow_pickle=True).item()
    insoleRegionL,smplRegionL = RegionInsole2SMPLL['insole'],RegionInsole2SMPLL['smpl']

    '''visual'''
    insole_img = []
    for insoleRegion in [insoleRegionL,insoleRegionR]:
        color_map = np.ones([33,11,3])*255
        color_map[insoleRegion['BigToe'][0,:],insoleRegion['BigToe'][1,:],:] = rgb_code['Red']
        color_map[insoleRegion['MidToe'][0,:],insoleRegion['MidToe'][1,:],:] = rgb_code['Blue']
        color_map[insoleRegion['SmallToe'][0,:],insoleRegion['SmallToe'][1,:],:] = rgb_code['Green']
        color_map[insoleRegion['Front0'][0,:],insoleRegion['Front0'][1,:],:] = rgb_code['Purple']
        color_map[insoleRegion['Front1'][0,:],insoleRegion['Front1'][1,:],:] = rgb_code['LightSlateGray']
        color_map[insoleRegion['Front2'][0,:],insoleRegion['Front2'][1,:],:] = rgb_code['Cyan']
        color_map[insoleRegion['Mid0'][0,:],insoleRegion['Mid0'][1,:],:] = rgb_code['OliveDrab']
        color_map[insoleRegion['Mid1'][0,:],insoleRegion['Mid1'][1,:],:] = rgb_code['Orange']
        color_map[insoleRegion['Back0'][0,:],insoleRegion['Back0'][1,:],:] = rgb_code['FireBrick']
        color_map[insoleRegion['Back1'][0,:],insoleRegion['Back1'][1,:],:] = rgb_code['LightSeaGreen']
        insole_img.append(color_map)
    insole_img = np.concatenate(insole_img,axis=1)
    cv2.imwrite('debug/insole2smpl/insoleRegion.png',insole_img.astype(np.uint8))

    footIdsL, footIdsR = m_vis.footIdsL, m_vis.footIdsR
    v_footL, v_footR = m_vis.v_footL, m_vis.v_footR
    foot_regions = ['BigToe','MidToe','SmallToe','Front0','Front1','Front2','Mid0','Mid1','Back0','Back1']
    colors = [rgb_code['Red'],rgb_code['Blue'],rgb_code['Green'],rgb_code['Purple'],
              rgb_code['LightSlateGray'],rgb_code['Cyan'],rgb_code['OliveDrab'],
              rgb_code['Orange'],rgb_code['FireBrick'],rgb_code['LightSeaGreen']]

    smpl_img = []
    for footIds,v_foot,smplRegion in zip([footIdsL, footIdsR],[v_footL, v_footR],[smplRegionL,smplRegionR]):
        vert_color = np.ones([footIds.shape[0], 3])
        for region_name, color_name in zip(foot_regions,colors):
            ic(region_name, color_name)
            smpl_ids = smplRegion[region_name]
            for ids in smpl_ids:
                b = np.where(footIds==ids)[0]
                vert_color[b] = color_name

        imgL = m_vis.drawSMPLFoot(v_foot, footIds,vert_color=vert_color)
        smpl_img.append(imgL)
    smpl_img = np.concatenate(smpl_img,axis=1)
    cv2.imwrite('debug/insole2smpl/smpl_foot.png', smpl_img.astype(np.uint8))
    exit()


if __name__ == '__main__':
    # footRegionMapping()
    # footRegionMappingRight2Left()
    visFootRegionMapping()
    save_dir = 'debug/insole2smpl'
    insoleMaskR = np.loadtxt('E:/dataset/PressureDataset/insole_mask/insoleMaskR.txt').astype(np.int32)
    footIdsL = np.loadtxt('essentials/footL_ids.txt').astype(np.int32)
    footIdsR = np.loadtxt('essentials/footR_ids.txt').astype(np.int32)
    model_temp = trimesh.load(osp.join(save_dir, 'v_template.obj'))
    v_template = np.array(model_temp.vertices)
    v_footL,v_footR = v_template[footIdsL, :],v_template[footIdsR, :]
    # trimesh.Trimesh(vertices=v_template[footIdsR,:],process=False).export('debug/footR.obj')
    # trimesh.Trimesh(vertices=v_template[footIdsL,:],process=False).export('debug/footL.obj')
    foot_verts = np.concatenate([v_footL, v_footR], axis=0)
    verts_xz = foot_verts[:, [0, 2]]

    insole2smplR = np.load('essentials/pressure/insole2smplR_.npy', allow_pickle=True).item()
    insole2smplL = np.load('essentials/pressure/insole2smplL_.npy', allow_pickle=True).item()

    insole_data = loadInsoleData(sub_ids='S12',seq_name='MoCap_20230422_145422')
    for insole_ids in trange(insole_data.shape[0]):
        # insole_ids = 24
        pressure_img = show_insole(insole_ids,insole_data)
        pressure_img = cv2.applyColorMap(pressure_img, cv2.COLORMAP_HOT)
        # cv2.imwrite(osp.join(save_dir,'pressure%d.png'%insole_ids),pressure_img)
        vert_colorR = getVertsColor(pressure_img[:,11:,:], footIdsR, insole2smplR)
        imgR = drawSMPLFoot(v_footR,footIdsR,np.array(model_temp.faces),
                            vert_color=vert_colorR,save_name=None,
                            point_size=4)

        vert_colorL = getVertsColor(pressure_img[:,:11,:], footIdsL, insole2smplL)
        imgL = drawSMPLFoot(v_footL,footIdsL,np.array(model_temp.faces),
                            vert_color=vert_colorL,save_name=None,
                            point_size=4)
        img = np.concatenate([cv2.resize(pressure_img, (2200, 3350)),imgL,imgR],axis=1)
        cv2.imwrite('debug/insole2smpl/results/%03d.png'%insole_ids,img)

    saveImgSeqAsvideo('debug/insole2smpl/results', fps=10,ratio=10)




