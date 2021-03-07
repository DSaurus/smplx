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

def skinning(lbs, A, verts):
    # 5. Do skinning:
    # W is N x V x (J + 1)
    print(A.shape)
    W = lbs.unsqueeze(dim=0).expand([1, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.reshape(1, 55, 16)) \
        .view(1, -1, 4, 4)

    homogen_coord = torch.ones([1, verts.shape[1], 1])
    m_verts_homo = torch.cat([verts, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(m_verts_homo, dim=-1))

    return v_homo[:, :, :3, 0]

def main(model_folder,
         model_type, dataroot,
         ext='npz',
         gender='neutral',
         num_betas=10,
         num_expression_coeffs=10,
         use_face_contour=False):
    import os
    import sys

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)

    betas = torch.randn([1, model.num_betas], dtype=torch.float32, requires_grad=True)
    expression = torch.randn([1, model.num_expression_coeffs], dtype=torch.float32, requires_grad=True)
    trans = torch.randn([1, 3], requires_grad=True)
    pose = torch.zeros((1, model.NUM_BODY_JOINTS * 3))
    pose.requires_grad = True

    smpl_dir = os.path.join(dataroot, 'smplx')
    img_dir = os.path.join(dataroot, 'img')
    par_dir = os.path.join(dataroot, 'parameter')
    subjects = os.listdir(os.path.join(smpl_dir))
    is_init = False

    ti.init(ti.cpu)
    obj = t3.readobj(os.path.join(smpl_dir, '200', 'smplx.obj'))
    obj['vi'][:, 0] = -obj['vi'][:, 0]
    obj['vi'][:, 2] = -obj['vi'][:, 2]
    # taichi show
    scene = t3.Scene()
    camera = t3.Camera()
    scene.add_camera(camera)
    light = t3.Light()
    scene.add_light(light)
    light2 = t3.Light([0, 0, -1])
    scene.add_light(light2)
    intrinsic = np.load(os.path.join(par_dir, '650', '%d_intrinsic.npy' % 0))
    extrinsic = np.load(os.path.join(par_dir, '650', '%d_extrinsic.npy' % 0))
    camera.set_intrinsic(intrinsic[0, 0], -intrinsic[1, 1], intrinsic[0, 2], 512 - intrinsic[1, 2])
    pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    tr = extrinsic[:3, :3].T
    camera.pos_py = [pos[i] for i in range(3)]
    camera.trans_py = [[tr[i, j] for j in range(3)] for i in range(3)]

    t3_m = t3.Model(obj=obj)
    scene.add_model(t3_m)
    camera._init()
    gui = ti.GUI('smpl')
    cv2.namedWindow('view_0')
    cv2.imshow('view_0', np.zeros((512, 512, 3)))

    while 1:
        if not is_init:
            optim = torch.optim.Adam([trans], lr=1e-1)
            for i in range(300):
                output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                               return_verts=True)
                verts = output.vertices
                gt_v = torch.FloatTensor(obj['vi']).unsqueeze(0)
                loss = torch.nn.functional.mse_loss(verts, gt_v)
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(loss.item())
                if i % 100 == 0:
                    vs = verts.detach().numpy().squeeze()
                    vs[:, 0] = -vs[:, 0]
                    vs[:, 2] = -vs[:, 2]
                    t3_m.vi.from_numpy(vs)
                    scene.render()  # render the model(s) into image
                    gui.set_image(camera.img)  # display the result image
                    gui.show()
            optim = torch.optim.Adam([trans, pose], lr=1e-2)
            for i in range(300):
                output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                               return_verts=True)
                verts = output.vertices
                gt_v = torch.FloatTensor(obj['vi']).unsqueeze(0)
                loss = torch.nn.functional.mse_loss(verts, gt_v)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if i % 500 == 0:
                    print(i, loss.item())
                if i % 30 == 0:
                    vs = verts.detach().numpy().squeeze()
                    vs[:, 0] = -vs[:, 0]
                    vs[:, 2] = -vs[:, 2]
                    t3_m.vi.from_numpy(vs)
                    scene.render()  # render the model(s) into image
                    gui.set_image(camera.img)  # display the result image
                    gui.show()
            optim = torch.optim.Adam([trans, pose, betas, expression], lr=1e-2)
            for i in range(1000):
                output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                               return_verts=True)
                verts = output.vertices
                gt_v = torch.FloatTensor(obj['vi']).unsqueeze(0)
                loss = torch.nn.functional.mse_loss(verts, gt_v)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if i % 30 == 0:
                    vs = verts.detach().numpy().squeeze()
                    vs[:, 0] = -vs[:, 0]
                    vs[:, 2] = -vs[:, 2]
                    t3_m.vi.from_numpy(vs)
                    scene.render()  # render the model(s) into image
                    gui.set_image(camera.img)  # display the result image
                    gui.show()
                    print(loss.item())
            is_init = True
        else:
            optim = torch.optim.Adam([trans, pose, betas, expression], lr=1e-2)
            for i in range(200):
                output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                               return_verts=True)
                verts = output.vertices
                gt_v = torch.FloatTensor(obj['vi']).unsqueeze(0)
                loss = torch.nn.functional.mse_loss(verts, gt_v)
                optim.zero_grad()
                loss.backward()
                optim.step()
                if i % 30 == 0:
                    vs = verts.detach().numpy().squeeze()
                    vs[:, 0] = -vs[:, 0]
                    vs[:, 2] = -vs[:, 2]
                    t3_m.vi.from_numpy(vs)
                    scene.render()  # render the model(s) into image
                    gui.set_image(camera.img)  # display the result image
                    gui.show()
                    print(loss.item())
        print('optim finished, press c to next step or other key to continue optimization.')
        if cv2.waitKey() == ord('c'):
            break


    betas0 = betas.clone()
    expression0 = expression.clone()
    pose0 = pose.clone()
    trans0 = trans.clone()

    obj2 = t3.readobj(os.path.join(smpl_dir, '205', 'smplx.obj'))
    obj2['vi'][:, 0] *= -1
    obj2['vi'][:, 2] *= -1
    optim = torch.optim.Adam([trans, pose, betas, expression], lr=1e-2)
    for i in range(200):
        output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                       return_verts=True)
        verts = output.vertices
        gt_v = torch.FloatTensor(obj2['vi']).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(verts, gt_v)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 30 == 0:
            vs = verts.detach().numpy().squeeze()
            vs[:, 0] = -vs[:, 0]
            vs[:, 2] = -vs[:, 2]
            t3_m.vi.from_numpy(vs)
            scene.render()  # render the model(s) into image
            gui.set_image(camera.img)  # display the result image
            gui.show()
            print(loss.item())

    output = model(betas=betas0, expression=expression0, body_pose=pose0, transl=trans0,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    obj = t3.readobj('dataset/result_200.obj')
    # obj = t3.readobj(os.path.join(smpl_dir, subjects[0], 'smplx_new.obj'))
    m_verts = obj['vi'][:, :3]
    m_verts[:, 0] *= -1
    m_verts[:, 2] *= -1
    flann = FLANN()
    result, dists = flann.nn(vertices, m_verts, 1)
    result = torch.LongTensor(result)
    print(result)
    lbs_weights = model.lbs_weights
    A = model(betas=betas0, expression=expression0, body_pose=pose0, transl=trans0, return_weight=True).vertices
    A_inv = torch.inverse(A.reshape(-1, 4, 4))
    obj_lbs_weights = lbs_weights[result[:], :]
    m_verts = torch.FloatTensor(m_verts).unsqueeze(0)
    m_verts -= trans0.reshape(1, 1, 3)
    verts = skinning(obj_lbs_weights, A_inv, m_verts)

    A2 = model(betas=betas, expression=expression, body_pose=pose, transl=trans, return_weight=True).vertices
    verts = skinning(obj_lbs_weights, A2, verts)
    verts += trans.reshape(1, 1, 3)
    print(verts)
    print(verts.shape)
    verts = verts.detach().cpu().numpy().squeeze()
    f = open('test.obj', 'w')
    for i in range(verts.shape[0]):
        f.write('v %f %f %f\n' % (-verts[i, 0], verts[i, 1], -verts[i, 2]))
    for i in range(obj['f'].shape[0]):
        f.write('f %d %d %d\n' % (obj['f'][i, 0, 0] + 1, obj['f'][i, 1, 0] + 1, obj['f'][i, 2, 0] + 1))

    exit(0)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')
    parser.add_argument('--dataroot', type=str)
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

    main(model_folder, model_type, args.dataroot,
         gender=gender,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs)
