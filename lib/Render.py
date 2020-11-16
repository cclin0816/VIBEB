import os
import math
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh
import pyrender
import cv2
from lib.models.smpl import get_smpl_faces


def Render(hp, vibe_result):
    renderer = pyrender.OffscreenRenderer(800, 800)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    nc = pyrender.Node(camera=camera, matrix=np.eye(4))
    faces = get_smpl_faces()
    nc.translation = (0, 0, 2.73)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, -1, 1]
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = [0, 1, 1]
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = [1, 1, 2]
    scene.add(light, pose=light_pose)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='BLEND',
        baseColorFactor=(0.4, 0.6, 0.7, 1.0)
    )

    for i in range(len(hp)):
        verts = vibe_result[(i + 1) % 2]['verts'][hp[i]]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        Rz = trimesh.transformations.rotation_matrix(math.radians(180), [0, 0, 1])
        mesh.apply_transform(Rz)
        Ty = trimesh.transformations.translation_matrix([0, -0.2, 0])
        mesh.apply_transform(Ty)
        pymesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = pyrender.Node(mesh=pymesh)
        out = cv2.VideoWriter(f'data/render/{i:02d}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 24, (800, 800))

        for j in range(180):
            
            Ry = trimesh.transformations.rotation_matrix(math.radians(2 * j), [0, 1, 0])
            
            mesh_node.matrix = Ry
            scene.add_node(mesh_node)
            scene.add_node(nc)
            
            rgb, _ = renderer.render(scene)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
            out.write(bgr)
            scene.remove_node(nc)
            scene.remove_node(mesh_node)

        out.release()



