import bpy
import os
from math import radians

if __name__ == '__main__':

    smpl_seq_root = '20230422/S01/MoCap_20230422_093211/'
    start_idx, end_idx = 10, 37
    output_path = 'debug/blender_output/MoCap_20230422_093211'
    os.makedirs(output_path, exist_ok=True)

    # delete all
    if (len(bpy.data.objects) != 0):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False, confirm=False)

    # create an empty file
    bpy.ops.wm.read_homefile(use_empty=True)

    # import a plane
    bpy.ops.mesh.primitive_plane_add(
        size=20,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
        scale=(2, 2, 1))
    ground = bpy.context.object
    ground.name = 'test_ground'
    ground.data.name = 'test_ground_'

    ground.location[0] = -0.63
    ground.location[1] = 2.12

    # add camera
    bpy.ops.object.camera_add(
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=(1, 1, 1))

    cam = bpy.context.object
    cam.name = 'test_cam'
    cam.data.name = 'test_cam_'

    cam.rotation_euler[0] = radians(70.0881)
    cam.rotation_euler[1] = radians(0)
    cam.rotation_euler[2] = radians(60.0)
    cam.location[0] = 4.03354
    cam.location[1] = -2.0418
    cam.location[2] = 2.70828

    # add light
    bpy.ops.object.light_add(
        type='SUN',
        radius=1,
        align='WORLD',
        location=(0, 0, 0),
        scale=(1, 1, 1))
    light = bpy.context.object
    light.name = 'test_light'
    light.data.name = 'test_light_'

    light.location[0] = 0.821
    light.location[1] = 0.744
    light.location[2] = 2.85

    light.rotation_euler[0] = -9.59045
    light.rotation_euler[1] = -16.6077
    light.rotation_euler[2] = 26.0277

    light.data.energy = 3.0

    mat_floor = bpy.data.materials.new(name='floor')
    ground.data.materials.append(mat_floor)
    mat_floor.use_nodes = True
    nodes = mat_floor.node_tree.nodes

    for node in nodes:
        nodes.remove(node)

    principled_node =\
        mat_floor.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

    texture_node = mat_floor.node_tree.nodes.new(type='ShaderNodeTexChecker')
    texture_node.inputs['Scale'].default_value = 20.0  # 调整棋盘格中每一个小方格的大小

    subsurface_color_node = nodes.new(type='ShaderNodeRGB')
    subsurface_color_node.outputs['Color'].default_value =\
        (0.1844, 0.0875, 0.355, 1)

    links = mat_floor.node_tree.links
    links.new(texture_node.outputs['Color'],
              principled_node.inputs['Base Color'])
    links.new(subsurface_color_node.outputs['Color'],
              principled_node.inputs['Subsurface Color'])

    principled_node.inputs['Subsurface'].default_value = 0.5

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    bpy.data.scenes['Scene'].cycles.samples = 1024
    bpy.data.scenes['Scene'].cycles.adaptive_threshold = 0.02

    for frame in range(start_idx, end_idx):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.obj(
            filepath=f'{smpl_seq_root}/smpl_{frame:03d}.obj')

        imported_object = bpy.context.object

        # render
        bpy.context.scene.camera =\
            bpy.data.objects['test_cam']  # link camera

        bpy.context.scene.render.resolution_x = 576
        bpy.context.scene.render.resolution_y = 640

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath =\
            f'{output_path}/test_{frame:03d}.png'
        bpy.ops.render.render(write_still=True)  # use_viewport = True

        # bpy.ops.wm.save_mainfile(filepath="debug/render_save.blend")

        obj = bpy.data.objects[f'smpl_{frame:03d}']
        bpy.data.objects.remove(obj)

        # import pdb;pdb.set_trace()
    # bpy.ops.wm.save_mainfile(filepath="D:/utils/blender_utils/render_save.blend")
    import pdb
    pdb.set_trace()
