import numpy as np
import open3d as o3d
import os
import utils.SupportFuntcion as sf

mesh,v,f = sf.ReadObj('./data/obj/239L.obj')
distance = np.loadtxt('./data/distance.txt')
print(distance[0])
dis_per_frame = distance/60
print(dis_per_frame[0])

view = sf.CreateDisplayWindow()
# view.add_geometry(v+distance)
for i in range(60):
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.85, 0.85, 0.85] for i in range(len(mesh_ls.lines))])
    view.add_geometry(mesh_ls)
    v = v+dis_per_frame
    sf.DrawVertices(view, v)

    ctr = view.get_view_control()
    ctr.set_front([0.0001,1,0])
    view.run()
    view.capture_screen_image('./shift_gif/'+str(i)+'.png', True)
    view.clear_geometries()


