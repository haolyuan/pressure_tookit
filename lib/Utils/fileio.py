from icecream import ic

def saveOBJ(filename,model):
    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], vc[0], vc[1], vc[2]))

def saveFloorAsOBJ(filename, floor_point,floor_normal):
    with open(filename, 'w') as fp:
        fp.write('v %f %f %f\n' % (floor_point[0],floor_point[1],floor_point[2]))
        fp.write('v %f %f %f\n' % (floor_point[0]+floor_normal[0],
                                    floor_point[1]+floor_normal[1],
                                    floor_point[2]+floor_normal[2]))
        fp.write('l 1 2\n')

def saveJointsAsOBJ(filename, joints,parents):
    with open(filename, 'w') as fp:
        for joint in joints:
            fp.write('v %f %f %f\n' % (joint[0],joint[1],joint[2]))
        for pi in range(1,parents.shape[0]):
            fp.write('l %d %d\n' % (pi+1,parents[pi]+1))

def saveNormalsAsOBJ(filename, verts, normals,ratio=0.2):
    with open(filename, 'w') as fp:
        for vi in range(verts.shape[0]):
            fp.write('v %f %f %f\n' % (verts[vi,0],verts[vi,1],verts[vi,2]))
            fp.write('v %f %f %f\n' % (verts[vi,0]+ratio*normals[vi,0],
                                       verts[vi,1]+ratio*normals[vi,1],
                                       verts[vi,2]+ratio*normals[vi,2]))
        for li in range(verts.shape[0]):
            fp.write('l %d %d\n' % (2*li+1,2*li+2))

def saveCorrsAsOBJ(filename, verts_src, tar_verts):
        with open(filename, 'w') as fp:
            for vi in range(verts_src.shape[0]):
                fp.write('v %f %f %f\n' % (verts_src[vi, 0], verts_src[vi, 1], verts_src[vi, 2]))
                fp.write('v %f %f %f\n' % (tar_verts[vi, 0], tar_verts[vi, 1], tar_verts[vi, 2]))
            for li in range(verts_src.shape[0]):
                fp.write('l %d %d\n' % (2*li + 1, 2*li+2))