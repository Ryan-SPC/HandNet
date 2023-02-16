import utils.SupportFuntcion as sf
import open3d as o3d
import numpy as np



handL_j = np.loadtxt('./data/save/sort_joints.txt')  
handR_j = np.loadtxt('./data/save/R_sort_joints.txt')  
handL_adj = np.loadtxt('./data/save/adj.txt')
handR_adj = np.loadtxt('./data/save/R_adj.txt')

model_j = np.loadtxt('./data/8290_j.txt')  
model_adj = np.loadtxt('./data/8290_adj.txt')

# pivotL = [0.398305, 0.663273, 0.0464565]#239
pivotL = [0.32403851, 0.52980298, 0.0531995]
# scaleL = 10.502436565283142
scaleL = 9.69001589472517

pivotR=[-0.39831300000000003, 0.663455, 0.0466865]
# [-0.32403751,  0.52980298,  0.0532    ]
scaleR = 10.504201680672269
9.68982841017416

handL_j = handL_j/scaleL
handL_j += pivotL

handR_j = handR_j/scaleR
handR_j += pivotR
# np.savetxt('./data/o239_j.txt', handR_j)

mesh,_,_ = sf.ReadObj('./data/8290.obj')
mesh.compute_vertex_normals()
center = mesh.get_center()
mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
mesh_ls.colors = o3d.utility.Vector3dVector([[0.75, 0.75, 0.75] for i in range(len(mesh_ls.lines))])

model_j[17] = handL_j[0]
model_j[16] = handR_j[0]

view = sf.CreateDisplayWindow()
# view.add_geometry(mesh)
view.add_geometry(mesh_ls)

sf.DrawVertices(view, handL_j, radius=0.002, color=[0,1,0])
sf.DrawSkeleton(view, handL_j, handL_adj, radius=0.002, color=[0,0,1])

# sf.DrawVertices(view, handR_j, radius=0.002, color=[1,0,0])
# sf.DrawSkeleton(view, handR_j, handR_adj, radius=0.002)

sf.DrawVertices(view, model_j, radius=0.002, color=[0,1,0])
sf.DrawSkeleton(view, model_j, model_adj, radius=0.002, color=[0,0,1])

# view.add_geometry(sf.drawSphere(model_j[24], 0.005, color=[1,0,0]))

ctr = view.get_view_control()
# ctr.set_front([0.00000001,1,0])
# ctr.set_front([-0.5,0.5,1])
ctr.set_lookat(center)



view.run()
