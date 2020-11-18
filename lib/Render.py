from util.FrameReader import FR
from lib.models.smpl import get_smpl_faces
import cv2
import pyrender
import trimesh
import numpy as np
import os
import math

from torch import uint8
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def Render(hp, vibe_result):
    renderer = pyrender.OffscreenRenderer(800, 800)
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
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
        verts = vibe_result[hp[i][0]]['verts'][hp[i][1]]
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        Rz = trimesh.transformations.rotation_matrix(
            math.radians(180), [0, 0, 1])
        mesh.apply_transform(Rz)
        Ty = trimesh.transformations.translation_matrix([0, -0.2, 0])
        mesh.apply_transform(Ty)
        pymesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = pyrender.Node(mesh=pymesh)
        out = cv2.VideoWriter(
            f'data/render/{hp[i][1]}/body.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (800, 800))

        for j in range(180):

            Ry = trimesh.transformations.rotation_matrix(
                math.radians(2 * j), [0, 1, 0])

            mesh_node.matrix = Ry
            scene.add_node(mesh_node)
            scene.add_node(nc)

            rgb, _ = renderer.render(scene)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out.write(bgr)
            scene.remove_node(nc)
            scene.remove_node(mesh_node)

        out.release()


def normalize(norm, vec):
    v = np.zeros(vec.shape[0])
    for i in range(vec.shape[0]):
        v[i] = norm[0] * vec[i][0] + norm[1] * vec[i][1]
    return v


def draw_line(joint_n, ba, a, b, c):
    return cv2.line(ba, (joint_n[a][0], joint_n[a][1]), (joint_n[b][0], joint_n[b][1]), c, 2)


def skeleton(hp, vibe_result, vid_path):
    fr = FR(vid_path)
    for i in range(len(hp)):
        _, data = fr.read_frame(hp[i][1])
        bbox = vibe_result[hp[i][0]]['bboxes'][hp[i][1]]
        s = bbox[3] * 0.6
        data = data[int(bbox[1] - s):int(bbox[1] + s),
                    int(bbox[0] - s):int(bbox[0] + s), :]
        data = cv2.resize(data, (400, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f'data/render/{hp[i][1]}/crop.jpg', data)

    backgroud = np.zeros((400, 400, 3), dtype=np.uint8)

    for i in range(len(hp)):
        b = np.copy(backgroud)
        joint = vibe_result[hp[i][0]]['joints3d'][hp[i][1]][-24:-8]
        l = joint[3, [0, 2]]
        r = joint[2, [0, 2]]
        norm = l - r
        norm = norm / math.sqrt(np.sum(norm * norm))
        joint_n = np.zeros((joint.shape[0], 2))
        joint_n[:, 0] = normalize(norm, joint[:, [0, 2]])
        joint_n[:, 1] = joint[:, 1] + 0.2
        joint_n = ((joint_n + 1) * 185).astype(int)
        b = draw_line(joint_n, b, 0, 1, (0, 255, 0))
        b = draw_line(joint_n, b, 1, 2, (0, 255, 0))
        b = draw_line(joint_n, b, 2, 3, (255, 0, 0))
        b = draw_line(joint_n, b, 3, 4, (0, 0, 255))
        b = draw_line(joint_n, b, 4, 5, (0, 0, 255))
        b = draw_line(joint_n, b, 6, 7, (0, 255, 0))
        b = draw_line(joint_n, b, 7, 8, (0, 255, 0))
        b = draw_line(joint_n, b, 8, 9, (255, 0, 0))
        b = draw_line(joint_n, b, 9, 10, (0, 0, 255))
        b = draw_line(joint_n, b, 10, 11, (0, 0, 255))
        b = draw_line(joint_n, b, 12, 13, (255, 0, 0))
        b = draw_line(joint_n, b, 12, 15, (255, 0, 0))
        b = draw_line(joint_n, b, 14, 15, (255, 0, 0))
        cv2.imwrite(f'data/render/{hp[i][1]}/joint.jpg', b)
