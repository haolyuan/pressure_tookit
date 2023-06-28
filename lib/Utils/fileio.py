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
