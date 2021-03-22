import taichi_three as t3
from tqdm import tqdm
import numpy as np
import queue
import sys
import os

single_name = 'C:\\code\Multiview_Pair-master\\results\\cmu_1500\\cmu6.obj'
dataroot = 'C:\\code\\Multiview_Pair-master\\results\\0309three2\\fusion'
for subject in tqdm(range(431, 700)):
    obj = t3.readobj(os.path.join(dataroot, 'fused_%d.obj' % subject))
    # obj = t3.readobj('C:\\code\\Multiview_Pair-master\\results\\mh_single/our.obj')
    # obj = t3.readobj(single_name)
    edges = {}
    visit = np.zeros(obj['vi'].shape[0])
    color = np.zeros(obj['vi'].shape[0])
    for i in range(obj['vi'].shape[0]):
        edges[i] = []
    for face in obj['f']:
        edges[face[0, 0]].append(face[1, 0])
        edges[face[0, 0]].append(face[2, 0])
        edges[face[1, 0]].append(face[0, 0])
        edges[face[1, 0]].append(face[2, 0])
        edges[face[2, 0]].append(face[0, 0])
        edges[face[2, 0]].append(face[1, 0])
    ind = 1
    for i in range(obj['vi'].shape[0]):
        if not visit[i]:
            Q = queue.Queue()
            Q.put(i)
            visit[i] = 1
            color[i] = ind
            while not Q.empty():
                x = Q.get()
                for to in edges[x]:
                    if visit[to]:
                        continue
                    visit[to] = 1
                    color[to] = ind
                    Q.put(to)
            ind += 1

    max_sum = 0
    max_color = 1
    for t in range(1, 10):
        sum = np.sum(color == t)
        if max_sum < sum:
            max_sum = sum
            max_color = t
    file = open(os.path.join(dataroot, 'clean_%d.obj' % subject), 'w')
    # file = open('C:\\code\\Multiview_Pair-master\\results\\mh_single/our.obj', 'w')
    # file = open(single_name, 'w')
    for v in obj['vi']:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in obj['f']:
        if color[f[0, 0]] != max_color or color[f[1, 0]] != max_color or color[f[2, 0]] != max_color:
            continue
        f = f[:, 0]
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
    # exit(0)