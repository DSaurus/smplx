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


class View:
    def __init__(self, img, vid, joints_2d):
        self.img = img
        self.vid = vid
        self.joints_2d = joints_2d
        self.joints_3d = None
        self.erase_id = -1
        self.mask_id = np.ones(joints_2d.shape[0])
        cv2.namedWindow('%d_view' % self.vid)
        cv2.moveWindow('%d_view' % self.vid, self.vid % 3 * 512, self.vid // 3 * 512)
    def init(self):
        def call_back(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                min_dis = 1e9
                min_i = -1
                for i in range(self.joints_2d.shape[0]):
                    if (x - self.joints_2d[i, 0]) ** 2 + (y - self.joints_2d[i, 1]) ** 2 < min_dis:
                        min_dis = (x - self.joints_2d[i, 0]) ** 2 + (y - self.joints_2d[i, 1]) ** 2
                        min_i = i
                self.mask_id[min_i] = 0
                self.joints_2d[min_i, :] = 0
                self.draw()
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.erase_id != -1:
                    self.joints_2d[self.erase_id, 0] = x
                    self.joints_2d[self.erase_id, 1] = y
                    self.erase_id = -1
                    self.draw()
            if event == cv2.EVENT_RBUTTONDOWN:
                if self.erase_id == -1:
                    min_dis = 1e9
                    min_i = -1
                    for i in range(self.joints_2d.shape[0]):
                        if (x-self.joints_2d[i, 0])**2 + (y-self.joints_2d[i, 1])**2 < min_dis:
                            min_dis = (x-self.joints_2d[i, 0])**2 + (y-self.joints_2d[i, 1])**2
                            min_i = i
                    self.erase_id = min_i
                    self.draw()
        cv2.setMouseCallback('%d_view' % self.vid, call_back)

    def draw(self):
        img = self.img.copy()
        for i in range(self.joints_2d.shape[0]):
            if i != self.erase_id:
                cv2.circle(img, (int(self.joints_2d[i, 0]), int(self.joints_2d[i, 1])), 2, (0, 0, 255), 2)
        edge = [[0, 1], [0, 2], [1, 4], [2, 5], [5, 8], [4, 7], [8, 11], [7, 10], [0, 3], [3, 6], [6, 9],
                [9, 12], [12, 15], [9, 14],
                [14, 17], [17, 19], [19, 21], [9, 13], [13, 16], [16, 18], [18, 20]]
        if self.joints_3d is not None:
            for i in range(self.joints_3d.shape[0]):
                cv2.circle(img, (int(self.joints_3d[i, 0]), int(self.joints_3d[i, 1])), 2, (255, 0, 0), 2)
            for e in edge:
                cv2.line(img, (int(self.joints_3d[e[0], 0]), int(self.joints_3d[e[0], 1])),
                         (int(self.joints_3d[e[1], 0]), int(self.joints_3d[e[1], 1])),
                         (255, 0, 0), 2)

        for e in edge:
            if self.mask_id[e[0]] == 0 or self.mask_id[e[1]] == 0:
                continue
            cv2.line(img, (int(self.joints_2d[e[0], 0]), int(self.joints_2d[e[0], 1])), (int(self.joints_2d[e[1], 0]), int(self.joints_2d[e[1], 1])),
                     (0, 255, 0), 2)
        cv2.imshow('%d_view' % self.vid, img)



def main(model_folder,
         model_type='smplx',
         ext='npz',
         gender='neutral',
         plot_joints=False,
         num_betas=10,
         sample_shape=True,
         sample_expression=True,
         num_expression_coeffs=10,
         plotting_module='pyrender',
         use_face_contour=False):
    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         num_betas=num_betas,
                         num_expression_coeffs=num_expression_coeffs,
                         ext=ext)
    print(model)
    ti.init(ti.cpu)
    obj = t3.readobj('dataset/650/smplx.obj')
    obj['vi'][:, 0] = -obj['vi'][:, 0]
    obj['vi'][:, 2] = -obj['vi'][:, 2]
    scene = t3.Scene()
    camera = t3.Camera()
    scene.add_camera(camera)
    light = t3.Light()
    scene.add_light(light)
    light2 = t3.Light([0, 0, -1])
    scene.add_light(light2)
    intrinsic = np.load('dataset/650/0_intrinsic.npy')
    extrinsic = np.load('dataset/650/0_extrinsic.npy')
    camera.set_intrinsic(intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    trans = extrinsic[:3, :3].T
    camera.pos_py = [pos[i] for i in range(3)]
    camera.trans_py = [[trans[i, j] for j in range(3)] for i in range(3)]

    t3_m = t3.Model(obj=obj)
    scene.add_model(t3_m)
    camera._init()

    gui = ti.GUI('smpl')

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32, requires_grad=True)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32, requires_grad=True)
    trans = torch.randn([1, 3], requires_grad=True)
    global_o = torch.zeros(3).unsqueeze(0)
    pose = torch.zeros((1, model.NUM_BODY_JOINTS*3))
    global_o.require_grad = True
    pose.requires_grad = True
    print(global_o.shape, pose.shape)

    optim = torch.optim.Adam([trans, global_o], lr=1e-1)
    for i in range(300):
        output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                       return_verts=True, global_orient=global_o)
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
    optim = torch.optim.Adam([trans, global_o, pose], lr=1e-2)
    for i in range(300):
        output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                       return_verts=True, global_orient=global_o)
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
    optim = torch.optim.Adam([trans, pose, global_o, betas, expression], lr=1e-2)
    for i in range(1000):
        output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                       return_verts=True, global_orient=global_o)
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

    output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                   return_verts=True, global_orient=global_o)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    f = open('smplx.obj', 'w')
    for i in range(vertices.shape[0]):
        f.write('v %f %f %f\n' % (vertices[i, 0], vertices[i, 1], vertices[i, 2]))

    for i in range(obj['f'].shape[0]):
        f.write('f %d %d %d\n' % (obj['f'][i, 0, 0]+1, obj['f'][i, 1, 0]+1, obj['f'][i, 2, 0]+1))

    print(joints.shape)
    joints[:, 0] *= -1
    joints[:, 2] *= -1
    view_list = []
    ex_list = []
    in_list = []
    for vid in range(6):
        intrinsic = np.load('dataset/650/%d_intrinsic.npy' % vid)
        extrinsic = np.load('dataset/650/%d_extrinsic.npy' % vid)
        pts = extrinsic[:3, :3] @ joints.T
        pts += extrinsic[:3, 3:]
        pts = intrinsic @ pts
        pts[:2, :] /= pts[2:, :]
        pts = pts.T
        view = View(cv2.imread('dataset/650/%d.jpg' % vid), vid, pts[:22, :2])
        view.init()
        view.draw()
        view_list.append(view)
        ex_list.append(torch.from_numpy(extrinsic).float())
        in_list.append(torch.from_numpy(intrinsic).float())

    while 1:
        if cv2.waitKey(20) & 0xFF == 27:
            break

    joints_3d = torch.from_numpy(joints[:22]).float()
    joints_3d.requires_grad = True
    optim = torch.optim.Adam([joints_3d], lr=1e-3)
    for _ in range(1000):
        loss = 0
        for vid in range(6):
            pts = ex_list[vid][:3, :3] @ joints_3d.T
            pts += ex_list[vid][:3, 3:]
            pts = in_list[vid] @ pts
            pts2 = pts[:2, :] /pts[2:, :]
            pts2 = pts2.T
            mask = torch.FloatTensor(view_list[vid].mask_id).unsqueeze(1)
            gt = torch.from_numpy(view_list[vid].joints_2d).float()
            loss += F.mse_loss(pts2*mask / 512, gt*mask / 512)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if _ % 10 == 0:
            for vid in range(6):
                pts = ex_list[vid][:3, :3] @ joints_3d.T
                pts += ex_list[vid][:3, 3:]
                pts = in_list[vid] @ pts
                pts[:2, :] /= pts[2:, :]
                pts = pts.T
                view_list[vid].joints_3d = pts.detach().cpu().numpy()
                view_list[vid].draw()
            if cv2.waitKey() & 0xFF == 27:
                break

    joints_3d.requires_grad = False
    joints_3d[:, 0] = -joints_3d[:, 0]
    joints_3d[:, 2] = -joints_3d[:, 2]
    optim = torch.optim.Adam([trans, pose, global_o, betas, expression], lr=1e-2)
    for i in range(1000):
        output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                       return_verts=True, global_orient=global_o)
        verts = output.vertices
        joints = output.joints
        loss = torch.nn.functional.mse_loss(joints[:, :22], joints_3d.unsqueeze(0))
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

    output = model(betas=betas, expression=expression, body_pose=pose, transl=trans,
                   return_verts=True, global_orient=global_o)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    f = open('smplx.obj', 'w')
    for i in range(vertices.shape[0]):
        f.write('v %f %f %f\n' % (-vertices[i, 0], vertices[i, 1], -vertices[i, 2]))
    for i in range(obj['f'].shape[0]):
        f.write('f %d %d %d\n' % (obj['f'][i, 0, 0] + 1, obj['f'][i, 1, 0] + 1, obj['f'][i, 2, 0] + 1))
    #
    # print('Vertices shape =', vertices.shape)
    # print('Joints shape =', joints.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

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
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--sample-shape', default=True,
                        dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True,
                        dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    num_betas = args.num_betas
    num_expression_coeffs = args.num_expression_coeffs
    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         num_betas=num_betas,
         num_expression_coeffs=num_expression_coeffs,
         sample_shape=sample_shape,
         sample_expression=sample_expression,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour)
