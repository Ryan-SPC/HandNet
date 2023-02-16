import utils.SupportFuntcion as sf
import open3d as o3d
import numpy as np

from os import listdir
from os.path import isfile,join

view = sf.CreateDisplayWindow()

mesh,v,f= sf.ReadObj('./data/239.obj')
center = mesh.get_center()
mesh.compute_vertex_normals()

mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
mesh_ls.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for i in range(len(mesh_ls.lines))])

shift = np.loadtxt('./data/body_save/shift.txt')  
cluster = np.loadtxt('./data/body_save/cluster.txt')  
refine = np.loadtxt('./data/body_save/refine.txt')  
att = np.loadtxt('./data/body_save/att.txt')
# print(np.max(att))

# for i in range(len(att)):

#     view.add_geometry(sf.drawSphere(shift[i], 0.004, color=[1,1-att[i],1-att[i]]))


# view.add_geometry(sf.drawSphere(j[0], 0.007, color=[1,0,0]))

# sf.DrawSkeleton(view, j, adj, radius=0.015, color=[0,0,1])
sf.DrawVertices(view, v, radius=0.004, color=[1,0,0])
# sf.DrawVertices(view, refine, radius=0.004, color=[1,0,0])
# view.add_geometry(mesh)
view.add_geometry(mesh_ls)



ctr = view.get_view_control()
ctr.set_lookat(center)
ctr.set_front([0.0001,1,0])

view.run()
