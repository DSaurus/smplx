import numpy as np
import trimesh

def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    # print('start creating grid')
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    # print('creating_grid_done')
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf.reshape(resolution)


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()

if __name__ == '__main__':
    from skimage import measure
    from mesh_to_sdf import mesh_to_voxels
    import taichi_three as t3
    import taichi as ti
    import matplotlib.pyplot as plt
    import math
    from pypoisson import poisson_reconstruction
    from tqdm import tqdm
    for subject in tqdm(range(202, 260-2)):
        ti.init(ti.cpu)
        scene = t3.Scene()
        light = t3.Light()
        scene.add_light(light)
        mesh0 = t3.readobj('fusion/single_test/inference_eval_%d_0.obj' % subject)
        mesh0['vi'] = mesh0['vi'][:, :3]
        b_max = np.max(mesh0['vi'], axis=0).reshape((-1, 3))
        b_min = np.min(mesh0['vi'], axis=0).reshape((-1, 3))
        mesh0['vi'] -= (b_max + b_min) / 2
        model = t3.Model(obj=mesh0)
        scene.add_model(model)
        camera = t3.Camera((512, 512))
        scene.add_camera(camera)

        r = 2
        c_depth_list = []
        c_pos_list = []
        c_nms_list = []
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
            c_depth_list.append(depth)
            c_pos_list.append(pos_map)
            c_nms_list.append(normal)

        ti.init(ti.cpu)
        scene = t3.Scene()
        light = t3.Light()
        scene.add_light(light)
        mesh1 = t3.readobj('fusion/single_test/inference_eval_%d_1.obj' % subject)
        mesh1['vi'] = mesh1['vi'][:, :3]
        mesh1['vi'] -= (b_max + b_min) / 2
        model = t3.Model(obj=mesh1)
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
            c_depth_list.append(depth)
            c_pos_list.append(pos_map)
            c_nms_list.append(normal)

        ti.init(ti.cpu)
        scene = t3.Scene()
        light = t3.Light()
        scene.add_light(light)
        mesh2 = t3.readobj('fusion/single_test/inference_eval_%d_2.obj' % subject)
        mesh2['vi'] = mesh2['vi'][:, :3]
        mesh2['vi'] -= (b_max + b_min) / 2
        model = t3.Model(obj=mesh2)
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
            c_depth_list.append(depth)
            c_pos_list.append(pos_map)
            c_nms_list.append(normal)

        pts_list = []
        nms_list = []
        print(len(c_depth_list))
        for i in range(8):
            ind1 = np.logical_and(c_depth_list[i] > 0,
                        np.logical_or(np.abs(c_depth_list[i] - c_depth_list[i+8]) < 0.01, np.abs(c_depth_list[i] - c_depth_list[i+16]) < 0.01))
            ind2 = np.logical_and(c_depth_list[i+8] > 0,
                        np.logical_or(np.abs(c_depth_list[i] - c_depth_list[i+8]) < 0.01, np.abs(c_depth_list[i+8] - c_depth_list[i+16]) < 0.01))
            ind3 = np.logical_and(c_depth_list[i+16] > 0,
                        np.logical_or(np.abs(c_depth_list[i+16] - c_depth_list[i+8]) < 0.01, np.abs(c_depth_list[i] - c_depth_list[i+16]) < 0.01))
            ind_cnt = (ind1.astype(int)+ind2.astype(int)+ind3.astype(int))
            ind = (ind1.astype(int)+ind2.astype(int)+ind3.astype(int)) >= 2
            print(np.sum(ind1), np.sum(ind2), np.sum(ind3))
            print(np.max(ind1+ind2+ind3))

            pos_map1 = c_pos_list[i]
            pos_map2 = c_pos_list[i+8]
            pos_map3 = c_pos_list[i+16]

            normal1 = c_nms_list[i]
            normal2 = c_nms_list[i+8]
            normal3 = c_nms_list[i+16]

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
        faces, vertices = poisson_reconstruction(points, normals, depth=8)
        vertices += (b_max + b_min) / 2
        save_obj_mesh('fusion/single_test/fused_%d.obj' % subject, vertices, faces)
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
