import numpy as np
import trimesh
import taichi_three as t3
import taichi as ti
import matplotlib.pyplot as plt
import math
from pypoisson import poisson_reconstruction
from tqdm import tqdm
import argparse
import os


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--result', type=str)
    args = parser.parse_args()
    os.makedirs(args.result, exist_ok=True)
    for subject in tqdm(range(202, 1199)):
        ti.init(ti.cpu)
        scene = t3.Scene()
        light = t3.Light()
        scene.add_light(light)
        mesh0 = t3.readobj(os.path.join(args.dataroot, 'inference_eval_%d_0.obj' % subject))
        mesh0['vi'] = mesh0['vi'][:, :3]
        b_max = np.max(mesh0['vi'], axis=0).reshape((-1, 3))
        b_min = np.min(mesh0['vi'], axis=0).reshape((-1, 3))
        mesh0['vi'] -= (b_max + b_min) / 2
        model = t3.Model(obj=mesh0)
        scene.add_model(model)
        camera = t3.Camera((512, 512))
        scene.add_camera(camera)

        r = 2
        c_depth_list = {}
        c_pos_list = {}
        c_nms_list = {}
        c_depth_list[0] = []
        c_pos_list[0] = []
        c_nms_list[0] = []
        for angle in range(0, 360, 45):
            ang = angle / 180 * math.acos(-1)
            camera.set(pos=[r*math.cos(ang), 0, r*math.sin(ang)], target=[0, 0, 0])
            camera._init()
            scene.render()
            # plt.imshow(camera.normal.to_numpy()*0.5 + 0.5)
            # plt.show()
            normal = camera.normal.to_numpy()
            pos_map = camera.pos_map.to_numpy()
            depth = camera.zbuf.to_numpy()
            c_depth_list[0].append(depth)
            c_pos_list[0].append(pos_map)
            c_nms_list[0].append(normal)

        fuse_list = [-1, 1]
        for id in fuse_list:
            c_depth_list[id] = []
            c_pos_list[id] = []
            c_nms_list[id] = []
            ti.init(ti.cpu)
            scene = t3.Scene()
            light = t3.Light()
            scene.add_light(light)
            mesh = t3.readobj(os.path.join(args.dataroot, 'inference_eval_%d_%d.obj' % (subject, id)))
            mesh['vi'] = mesh['vi'][:, :3]
            mesh['vi'] -= (b_max + b_min) / 2
            model = t3.Model(obj=mesh)
            scene.add_model(model)
            camera = t3.Camera((512, 512))
            scene.add_camera(camera)
            r = 2
            for angle in range(0, 360, 45):
                ang = angle / 180 * math.acos(-1)
                camera.set(pos=[r * math.cos(ang), 0, r * math.sin(ang)], target=[0, 0, 0])
                camera._init()
                scene.render()
                normal = camera.normal.to_numpy()
                pos_map = camera.pos_map.to_numpy()
                depth = camera.zbuf.to_numpy()
                c_depth_list[id].append(depth)
                c_pos_list[id].append(pos_map)
                c_nms_list[id].append(normal)


        pts_list = []
        nms_list = []
        print(len(c_depth_list))
        for i in range(8):
            ind1 = np.logical_and(c_depth_list[0][i] > 0,
                        np.logical_or(np.abs(c_depth_list[0][i] - c_depth_list[-1][i]) < 0.01, np.abs(c_depth_list[0][i] - c_depth_list[1][i]) < 0.01))
            ind2 = np.logical_and(c_depth_list[-1][i] > 0,
                        np.logical_or(np.abs(c_depth_list[-1][i] - c_depth_list[0][i]) < 0.01, np.abs(c_depth_list[-1][i] - c_depth_list[1][i]) < 0.01))
            ind3 = np.logical_and(c_depth_list[1][i] > 0,
                        np.logical_or(np.abs(c_depth_list[1][i] - c_depth_list[0][i]) < 0.01, np.abs(c_depth_list[1][i] - c_depth_list[-1][i]) < 0.01))
            ind_cnt = (ind1.astype(int)+ind2.astype(int)+ind3.astype(int))
            ind = (ind1.astype(int)+ind2.astype(int)+ind3.astype(int)) >= 2
            print(np.sum(ind1), np.sum(ind2), np.sum(ind3))
            print(np.max(ind1+ind2+ind3))

            pos_map1 = c_pos_list[0][i]
            pos_map2 = c_pos_list[-1][i]
            pos_map3 = c_pos_list[1][i]

            normal1 = c_nms_list[0][i]
            normal2 = c_nms_list[-1][i]
            normal3 = c_nms_list[1][i]

            pos_map1[ind != ind1, :] = 0
            pos_map2[ind != ind2, :] = 0
            pos_map3[ind != ind3, :] = 0
            normal1[ind != ind1, :] = 0
            normal2[ind != ind2, :] = 0
            normal3[ind != ind3, :] = 0

            pts = (pos_map1[ind, :] + pos_map2[ind, :] + pos_map3[ind, :]) / ind_cnt[ind].reshape(-1, 1)
            nms = (normal1[ind, :] + normal2[ind, :] + normal3[ind, :]) / ind_cnt[ind].reshape(-1, 1)

            print(pts.shape)
            print(nms.shape)

            pts_list.append(pts)
            nms_list.append(nms)
        # ind = depth != 0
        # x, y, z = pos_map[:, :, 0][ind], pos_map[:, :, 1][ind], pos_map[:, :, 2][ind]
        # nx, ny, nz = normal[:, :, 0][ind], normal[:, :, 1][ind], normal[:, :, 2][ind]
        # pts = np.zeros( (x.shape[0], 3))
        # nms = np.zeros( (x.shape[0], 3))
        # pts[:, 0], pts[:, 1], pts[:, 2] = x, y, z
        # nms[:, 0], nms[:, 1], nms[:, 2] = nx, ny, nz
        # pts_list.append(pts)
        # nms_list.append(nms)
        points = np.concatenate(pts_list, 0)
        normals = np.concatenate(nms_list, 0)
        faces, vertices = poisson_reconstruction(points, normals, depth=9)
        vertices += (b_max + b_min) / 2
        save_obj_mesh(os.path.join(args.result, 'fused_%d.obj' % subject), vertices, faces)
        # mesh0 = trimesh.load('fusion/single_test/inference_eval_%d_0.obj' % subject)
        # mesh1 = trimesh.load('fusion/single_test/inference_eval_%d_1.obj' % subject)
        # mesh2 = trimesh.load('fusion/single_test/inference_eval_%d_2.obj' % subject)
        # def eval(points):
        #     points = points.T
        #     inside0 = mesh0.contains(points)
        #     inside1 = mesh1.contains(points)
        #     inside2 = mesh2.contains(points)
        #     inside = (inside0 + inside1 + inside2) > 2
        #     return inside.astype(float)
        # center = mesh0.bounding_box.centroid
        # bbx = np.max(mesh0.bounding_box.extents)
        # vertices = mesh0.vertices - center
        # vertices *= 2 / bbx

        # mesh0 = trimesh.Trimesh(vertices=vertices, faces=mesh0.faces)
        # voxels = mesh_to_voxels(mesh0, 128, 'sample')
        # print(mesh0.center_mass)
        # bmin = mesh0.center_mass - 1
        # bmax = mesh0.center_mass + 1
        # bbx = np.array([[bmin[0], bmin[1], bmin[2]], [bmax[0], bmax[1], bmax[2]]])
        # vox0 = mesh0.voxelized(pitch=1.0 / 128, method='binvox', bounds=bbx, exact=True, binvox_path='D:/2020工作/binvox.exe')
        # vox0.fill()
        # vox0 = vox0.matrix
        # coord, mat = create_grid(256, 256, 256, bmin, bmax)
        # sdf = eval_grid_octree(coord, eval, 64, 0.01, 512*512)
        # verts, faces, _, _ = measure.marching_cubes_lewiner(voxels, level=0)
        # verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        # verts = verts.T
        # exit(0)
