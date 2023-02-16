import torch
import numpy as np
import open3d as o3d
import os
from os.path import isfile, join
from sys import platform
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import itertools as it


from models.GCN import JointPredNet
from models.PairCls_GCN import PairCls as BONENET
import utils.mesh_utils as Mesh
from utils.mesh_utils import Vector3d, Vector3i
from utils import binvox_rw

import utils.SupportFuntcion as sf
from utils.mst_utils import sample_on_bone, inside_check, primMST_symmetry, increase_cost_for_outside_bone, loadSkel_recur, primMST
from utils.tree_utils import TreeNode

from HandNet import HandNet
from RigNet import RigNet

def CutWrist_plane(mesh, joint_list, adj, wrist_pos):
    wrist_joint = None
    wrist_id = 0
    for i in range(len(joint_list)):
        joint = joint_list[i]
        if np.array_equal(joint, wrist_pos):
            wrist_joint = joint
            wrist_id = i
    # if wrist_joint == None:
    #     print("Can't find wrist!")
    #     return
    
    vertices = np.asarray(mesh.vertices)
    v_count = vertices.shape[0]

    parent_id = 0
    for i in range(len(adj[wrist_id])):
        if adj[wrist_id,i] == 1:
            parent_id = i
            break

    # Use wrist joint to find plane normal vector
    plane_normal = joint_list[wrist_id] - joint_list[parent_id]


    # Cut Mesh and save index map
    # Use plane normal vector to cut wrist
    # Vector dot plane normal > 0 means Hand side
    map_old_to_new = {}
    hand_vertex_count = 0
    hand_vertice = []
    for i in range(v_count):
        v = vertices[i]
        vec = v - wrist_joint
        if np.dot(vec, plane_normal) >0:
            map_old_to_new[i] = hand_vertex_count
            hand_vertice.append(v)
            hand_vertex_count += 1

    hand_indexs = map_old_to_new.keys()
    faces = np.array(mesh.triangles)
    isolate_indexs = []
    for index in hand_indexs:
        neighbors = FindNeighbor(index, faces)
        contain_count = 0
        for n in neighbors:
            if n in hand_indexs:
                contain_count += 1
        if contain_count == 0:
            isolate_indexs.append(index)
    isolate_indexs.reverse()
    
    if len(isolate_indexs) >0:
        for index in isolate_indexs:
            new_index = map_old_to_new[index]
            del map_old_to_new[index]
            del hand_vertice[new_index]



        count = 0
        for key in map_old_to_new.keys():
            map_old_to_new[key] = count
            count+=1
        
    

    hand_vertice = np.array(hand_vertice)
    hand_mesh = SealMesh(mesh, hand_vertice, map_old_to_new, wrist_pos)
    return hand_mesh

def SealMesh(origin_mesh, new_vertice, index_map_old_to_new, seal_point):
    faces = np.array(origin_mesh.triangles)

    fTemp = []
    vertice_old_index = index_map_old_to_new.keys()
    last_index = new_vertice.shape[0]
    index_map_old_to_new[-1] = last_index
    for face_np in faces:
        face = face_np.tolist()
        # if face contain more than 1 index that not in wrist, then it should be ignore
        not_in_wrist_count  = 0
        for i in range(len(face)):
            index = face[i]
            if index not in vertice_old_index:
                # Change not in wrist index to -1
                # Later we can map -1 to Last Index
                face[i] = -1
                not_in_wrist_count += 1
        if not_in_wrist_count <= 1:
            fTemp.append(face)

    for face in fTemp:
        for i in range(3):
            old_index = face[i]
            face[i] = index_map_old_to_new[old_index]

    new_faces = np.array(fTemp)
    new_vertice = np.append(new_vertice,[seal_point],axis=0)
    new_mesh = sf.Mesh(new_vertice, new_faces)
    return new_mesh

def FindNeighbor(v_id, faces):
    neighbor = []
    face_ids = np.argwhere(faces == v_id)[:, 0]

    for face_id in face_ids:
        for v in faces[face_id]:
            if v != v_id and v not in neighbor:
                neighbor.append(v)
    return neighbor


def mark(joints):
    joint_pcd = o3d.geometry.PointCloud()
    joint_pcd.points = o3d.utility.Vector3dVector(joints)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.get_render_option().point_size = 20.0


    vis.add_geometry(joint_pcd)
    vis.remove_geometry(joint_pcd)
    
    vis.run()
    vis.destroy_window()
    mark_list = vis.get_picked_points()
    if len(mark_list)!=2:
        print('Please Select Exact 2 Joint!')
        mark(joints)
    left = joints[mark_list[0]]
    right = joints[mark_list[1]]
    
    if left[0] < 0:
        temp = left
        left = right
        right = temp
    return left, right

if __name__ == '__main__':
    rignet = RigNet()
    rignet.LoadModel()
    handNet = HandNet()
    handNet.LoadModel()


    mesh_file = './data/239.obj'
    mesh,_,_ = sf.ReadObj(mesh_file)
    joints,adj = rignet.Run(mesh_file)

    left, right = mark(joints)
    left_hand_mesh = CutWrist_plane(mesh, joints, adj, left)
    left_hand_mesh, pivot_L, scale_L = sf.NormalizeMesh(left_hand_mesh)
    sf.WriteObj('./data/left_hanf.obj', left_hand_mesh.vertices, left_hand_mesh.triangles)
    # joints,adj = handNet.Run(mesh_file)

    view = sf.CreateDisplayWindow()
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(len(mesh_ls.lines))])
    view.add_geometry(mesh_ls)
    # sf.DrawVertices(view, joints)
    # sf.DrawSkeleton(view,joints,adj)
    ctr = view.get_view_control()
    # ctr.set_front([0.0001,1,0])
    view.run()
    view.destroy_window()
