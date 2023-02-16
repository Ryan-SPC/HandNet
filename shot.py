import torch
import numpy as np
import open3d as o3d
import os
from os.path import isfile, join
from sys import platform
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data


from models.GCN import JointPredNet
import utils.mesh_utils as Mesh
from utils.mesh_utils import Vector3d, Vector3i
from utils import binvox_rw

import utils.SupportFuntcion as sf








def get_id_list(folder):
    files = [f for f in os.listdir(folder) if isfile(join(folder, f))]
    id_list = []
    for file in files:
        name = file.split('.')[0]
        id = name.split('_')[0]
        if id not in id_list:
            id_list.append(id)
    return id_list



if __name__ == '__main__':
    id_list = get_id_list('./data/obj')


    view = sf.CreateDisplayWindow()
    for id in id_list:
        id = '3349L'
        mesh,v,f = sf.ReadObj('./data/obj/'+id+'.obj')
        j = np.loadtxt('./data/'+id+'_pred.txt')  

    # show 
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_ls.colors = o3d.utility.Vector3dVector([[0.85, 0.85, 0.85] for i in range(len(mesh_ls.lines))])
        view.add_geometry(mesh_ls)
        sf.DrawVertices(view, j)

        ctr = view.get_view_control()
        ctr.set_front([0.0001,1,0])
        view.run()
        view.capture_screen_image('./shot/'+id+'.png', True)
        view.clear_geometries()
   
    
    # sf.WriteVerticeInObj('./data/out.obj', y_pred)


