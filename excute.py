
import utils.SupportFuntcion as sf
import open3d as o3d
import numpy as np

from os import listdir
from os.path import isfile,join

class V_AT:
    def __init__(self, id, pos, att) -> None:
        self.id = id
        self.pos = pos
        self.att =att
        self.neighbor = []
    def add(self, n):
        self.neighbor.append(n)

    def weight(self):
        n = np.array(self.neighbor)
        n = np.unique(n)
        tot_weight = 0
        for id in n:
            tot_weight += self.att[id]
        return tot_weight/n.shape[0]

view = sf.CreateDisplayWindow()
op = view.get_render_option()
op.light_on = True

op.mesh_show_wireframe = False
# s = o3d.visualization.rendering.Scene()

mesh,v,f = sf.ReadObj('./data/obj/239L.obj')
center = mesh.get_center()
mesh.compute_vertex_normals()

mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
mesh_ls.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for i in range(len(mesh_ls.lines))])

shift = np.loadtxt('./data/save/L_shift.txt')  
j = np.loadtxt('./data/save/L_sort_joints.txt')  
adj = np.loadtxt('./data/save/L_adj.txt')

# joint_mark_path = './wristNetTrain/human_txt/239_j_mark.txt'
# joint_mark = np.loadtxt(joint_mark_path)




# Draw attention
att = np.loadtxt('./data/hand_txt/239L_attn.txt')
V_AT_list = []
for i in range(v.shape[0]):
    V_AT_list.append(V_AT(i, v[i], att))
for face in f:
    v0 = V_AT_list[face[0]]
    v1 = V_AT_list[face[1]]
    v2 = V_AT_list[face[2]]

    v0.add(face[1])
    v0.add(face[2])

    v1.add(face[0])
    v1.add(face[2])

    v2.add(face[0])
    v2.add(face[1])

# file = open('./data/save/att.obj', 'w')
# for i in range(len(v)):
#     w = V_AT_list[i].weight()
#     if w >0.6:
#         color=[1,(1-w),0]
#     else:
#         color=[0.7,0.7,0.7]
#     file.write('v ' + str(v[i,0]) + ' ' + str(v[i,1]) + ' ' + str(v[i,2])+' '+str(color[0]) + ' '+str(color[1])+' '+ str(color[2]) +'\n')

# for i in range(len(f)):
#     file.write('f ' + str(f[i,0]+1) + ' ' + str(f[i,1]+1) + ' ' + str(f[i,2]+1)+'\n')

# file.close()

# view.add_geometry(sf.drawSphere(j[0], 0.007, color=[1,0,0]))

# sf.DrawSkeleton(view, j, adj, radius=0.015, color=[0,0,1])
sf.DrawVertices(view, shift, radius=0.01, color=[1,0,0])
# view.add_geometry(mesh)
view.add_geometry(mesh_ls)



ctr = view.get_view_control()
ctr.set_lookat(center)
ctr.set_front([0.0001,1,0])

view.run()