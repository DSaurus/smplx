# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse

import numpy as np
import torch

import smplx

import taichi_three as t3
import taichi as ti
import cv2
import torch.nn.functional as F
from pyflann import *
from tqdm import tqdm

def skinning(lbs, A, verts):
    # 5. Do skinning:
    # W is N x V x (J + 1)
    lbs = lbs.detach().cpu()
    A = A.detach().cpu()
    verts = verts.detach().cpu()
    W = lbs.unsqueeze(dim=0).expand([1, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.reshape(1, 55, 16)) \
        .view(1, -1, 4, 4)

    homogen_coord = torch.ones([1, verts.shape[1], 1])
    m_verts_homo = torch.cat([verts, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(m_verts_homo, dim=-1))

    return v_homo[:, :, :3, 0]

def main(model_folder,
         model_type, dataroot, result_dir, T, start, end,
         ext='npz',
         gender='neutral',
         num_betas=10,
         num_expression_coeffs=10,
         use_face_contour=False):
    import os
    import sys
    device = torch.device('cuda:0')
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    model = model.to(device)
    betas = torch.randn([1, model.num_betas], device=device, dtype=torch.float32, requires_grad=True)
    expression = torch.randn([1, model.num_expression_coeffs], device=device, dtype=torch.float32, requires_grad=True)
    trans = torch.randn([1, 3], device=device, requires_grad=True)
    pose = torch.zeros((1, model.NUM_BODY_JOINTS * 3), device=device)
    pose.requires_grad = True
    g_o = torch.zeros((1, 3), device=device)
    g_o.requires_grad = True

    is_init = False

    ti.init(ti.cpu)
    obj = t3.readobj(os.path.join(dataroot, 'smplx', str(args.start), 'smplx.obj'))
    obj['vi'][:, 0] = -obj['vi'][:, 0]
    obj['vi'][:, 2] = -obj['vi'][:, 2]
    # taichi show
    scene = t3.Scene()
    camera = t3.Camera(pos=[0, 0, -5])
    scene.add_camera(camera)
    light = t3.Light()
    scene.add_light(light)
    light2 = t3.Light([0, 0, -1])
    scene.add_light(light2)
    # intrinsic = np.load(os.path.join('dataset/parameter', '650', '%d_intrinsic.npy' % 0))
    # extrinsic = np.load(os.path.join('dataset/parameter', '650', '%d_extrinsic.npy' % 0))
    # camera.set_intrinsic(intrinsic[0, 0], -intrinsic[1, 1], intrinsic[0, 2], 512 - intrinsic[1, 2])
    # pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    # tr = extrinsic[:3, :3].T
    # camera.pos_py = [pos[i] for i in range(3)]
    # camera.trans_py = [[tr[i, j] for j in range(3)] for i in range(3)]

    t3_m = t3.Model(obj=obj)
    scene.add_model(t3_m)
    camera._init()
    gui = ti.GUI('smpl')
    cv2.namedWindow('view_0')
    cv2.imshow('view_0', np.zeros((512, 512, 3)))

    for subject in tqdm(range(args.start, args.end)):
        obj = t3.readobj(os.path.join(dataroot, 'smplx', '%d' % (subject+T), 'smplx.obj'))
        obj['vi'][:, 0] = -obj['vi'][:, 0]
        obj['vi'][:, 2] = -obj['vi'][:, 2]
        while 1:
            if not is_init:
                optim = torch.optim.Adam([trans, g_o], lr=1e-1)
                gt_v = torch.FloatTensor(obj['vi']).unsqueeze(0).to(device)
                for i in range(300):
                    output = model(betas=betas, expression=expression, body_pose=pose, transl=trans, global_orient=g_o,
                                   return_verts=True)
                    verts = output.vertices
                    loss = torch.nn.functional.mse_loss(verts, gt_v)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    print(loss.item())
                    if i % 100 == 0:
                        vs = verts.detach().cpu().numpy().squeeze()
                        vs[:, 0] = -vs[:, 0]
                        vs[:, 2] = -vs[:, 2]
                        t3_m.vi.from_numpy(vs)
                        scene.render()  # render the model(s) into image
                        gui.set_image(camera.img)  # display the result image
                        gui.show()
                optim = torch.optim.Adam([trans, g_o, pose], lr=1e-2)
                for i in range(300):
                    output = model(betas=betas, expression=expression, body_pose=pose, transl=trans, global_orient=g_o,
                                   return_verts=True)
                    verts = output.vertices
                    loss = torch.nn.functional.mse_loss(verts, gt_v)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    if i % 500 == 0:
                        print(i, loss.item())
                    if i % 30 == 0:
                        vs = verts.detach().cpu().numpy().squeeze()
                        vs[:, 0] = -vs[:, 0]
                        vs[:, 2] = -vs[:, 2]
                        t3_m.vi.from_numpy(vs)
                        scene.render()  # render the model(s) into image
                        gui.set_image(camera.img)  # display the result image
                        gui.show()
                optim = torch.optim.Adam([trans, g_o, pose, betas, expression], lr=1e-2)
                for i in range(1000):
                    output = model(betas=betas, expression=expression, body_pose=pose, transl=trans, global_orient=g_o,
                                   return_verts=True)
                    verts = output.vertices
                    loss = torch.nn.functional.mse_loss(verts, gt_v)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    if i % 30 == 0:
                        vs = verts.detach().cpu().numpy().squeeze()
                        vs[:, 0] = -vs[:, 0]
                        vs[:, 2] = -vs[:, 2]
                        t3_m.vi.from_numpy(vs)
                        scene.render()  # render the model(s) into image
                        gui.set_image(camera.img)  # display the result image
                        gui.show()
                        print(loss.item())
                is_init = True
            else:
                gt_v = torch.FloatTensor(obj['vi']).unsqueeze(0).to(device)
                optim = torch.optim.Adam([trans, g_o, pose, betas, expression], lr=1e-2)
                for i in range(200):
                    output = model(betas=betas, expression=expression, body_pose=pose, transl=trans, global_orient=g_o,
                                   return_verts=True)
                    verts = output.vertices
                    loss = torch.nn.functional.mse_loss(verts, gt_v)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    if i % 30 == 0:
                        vs = verts.detach().cpu().numpy().squeeze()
                        vs[:, 0] = -vs[:, 0]
                        vs[:, 2] = -vs[:, 2]
                        t3_m.vi.from_numpy(vs)
                        scene.render()  # render the model(s) into image
                        gui.set_image(camera.img)  # display the result image
                        gui.show()
                        print(loss.item())
            break
            # print('optim finished, press c to next step or other key to continue optimization.')
            # if cv2.waitKey() == ord('c'):
            #     break


        betas0 = betas.clone()
        expression0 = expression.clone()
        pose0 = pose.clone()
        trans0 = trans.clone()
        g_o0 = g_o.clone()

        obj2 = t3.readobj(os.path.join(dataroot, 'smplx', '%d' % (subject), 'smplx.obj'))
        print(subject)
        obj2['vi'][:, 0] *= -1
        obj2['vi'][:, 2] *= -1
        optim = torch.optim.Adam([trans, g_o, pose, betas, expression], lr=1e-2)

        gt_v = torch.FloatTensor(obj2['vi']).unsqueeze(0).to(device)
        for i in range(200):
            output = model(betas=betas, expression=expression, body_pose=pose, transl=trans, global_orient=g_o,
                           return_verts=True)
            verts = output.vertices
            # print(verts)
            loss = torch.nn.functional.mse_loss(verts, gt_v)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 30 == 0:
                vs = verts.detach().cpu().numpy().squeeze()
                vs[:, 0] = -vs[:, 0]
                vs[:, 2] = -vs[:, 2]
                t3_m.vi.from_numpy(vs)
                scene.render()  # render the model(s) into image
                gui.set_image(camera.img)  # display the result image
                gui.show()
                print(loss.item())

        output = model(betas=betas0, expressiotan=expression0, body_pose=pose0, transl=trans0, global_orient=g_o0,
                       return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        obj = t3.readobj(os.path.join(result_dir, 'inference_eval_%d_0.obj' % (subject+T)))
        # obj = t3.readobj(os.path.join(smpl_dir, subjects[0], 'smplx_new.obj'))
        m_verts = obj['vi'][:, :3]
        m_verts[:, 0] *= -1
        m_verts[:, 2] *= -1
        origin_verts = m_verts.copy()
        flann = FLANN()
        K = 4
        result, dists = flann.nn(vertices, m_verts, K)
        result = torch.LongTensor(result)
        obj_lbs_weights = torch.cat([model.lbs_weights[result[:, i], :].unsqueeze(0) for i in range(K)], 0)
        obj_lbs_weights = obj_lbs_weights.permute(1, 2, 0).detach().cpu()
        dists = torch.FloatTensor(dists)
        sigma = 0.1
        dists = torch.exp(-dists / sigma**2)
        obj_lbs_weights = obj_lbs_weights * dists.unsqueeze(1) / torch.sum(dists, dim=1).reshape(-1, 1, 1)
        obj_lbs_weights = torch.sum(obj_lbs_weights, dim=2)
        # print(result)
        lbs_weights = model.lbs_weights
        A = model(betas=betas0, expression=expression0, body_pose=pose0, transl=trans0, global_orient=g_o0, return_weight=True).vertices
        A_inv = torch.inverse(A.reshape(-1, 4, 4))
        A2 = model(betas=betas, expression=expression, body_pose=pose, transl=trans, global_orient=g_o,
                   return_weight=True).vertices
        A2 = A2.reshape(-1, 4, 4)
        B = torch.zeros_like(A2)
        for i in range(A2.shape[0]):
            B[i] = A2[i] @ A_inv[i]
        m_verts = torch.FloatTensor(m_verts).unsqueeze(0)
        m_verts -= trans0.reshape(1, 1, 3).detach().cpu()
        verts = skinning(obj_lbs_weights, B, m_verts)
        verts += trans.reshape(1, 1, 3).detach().cpu()
        # print(verts)
        # print(verts.shape)
        verts = verts.detach().cpu().numpy().squeeze()
        distance = np.sqrt(np.sum((verts - origin_verts)**2, axis=1))
        # print(distance)
        # print(np.sum(distance > 0.03))
        distance[distance > 0.12] = 1e3
        f = open(os.path.join(result_dir, 'inference_eval_%d_%d.obj' % (subject, T)), 'w')
        for i in range(verts.shape[0]):
            f.write('v %f %f %f\n' % (-verts[i, 0], verts[i, 1], -verts[i, 2]))
        for i in range(obj['f'].shape[0]):
            if distance[obj['f'][i, 0, 0]] > 1 or distance[obj['f'][i, 1, 0]] > 1 or distance[obj['f'][i, 2, 0]] > 1:
                continue
            f.write('f %d %d %d\n' % (obj['f'][i, 0, 0] + 1, obj['f'][i, 1, 0] + 1, obj['f'][i, 2, 0] + 1))
        # exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--result', type=str)
    parser.add_argument('--T', type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx', 'mano', 'flame'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='male',
                        help='The gender of the model')
    parser.add_argument('--num-betas', default=10, type=int,
                        dest='num_betas',
                        help='Number of shape coefficients.')
    parser.add_argument('--num-expression-coeffs', default=10, type=int,
                        dest='num_expression_coeffs',
                        help='Number of expression coefficients.')
    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    gender = args.gender
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs

    main(model_folder, model_type, args.dataroot, args.result, args.T, args.start, args.end,
         gender=gender,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs)
