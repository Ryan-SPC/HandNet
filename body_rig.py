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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def nms_meanshift(pts_in, density, bandwidth):
    """
    NMS to extract modes after meanshift. Code refers to sci-kit-learn.
    :param pts_in: input points
    :param density: density at each point
    :param bandwidth: bandwidth used in meanshift. Used here as neighbor region for NMS
    :return: extracted clusters.
    """
    Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
    sorted_ids = np.argsort(density)[::-1]
    unique = np.ones(len(sorted_ids), dtype=np.bool)
    dist = np.sqrt(Y)
    for i in sorted_ids:
        if unique[i]:
            neighbor_idxs = np.argwhere(dist[:, i] <= bandwidth)
            unique[neighbor_idxs.squeeze()] = 0
            unique[i] = 1  # leave the current point as unique
    pts_in = pts_in[unique]
    return pts_in


def meanshift_cluster(pts_in, bandwidth, weights=None, max_iter=15):
    """
    Meanshift clustering
    :param pts_in: input points
    :param bandwidth: bandwidth
    :param weights: weights per pts indicting its importance in the clustering
    :return: points after clustering
    """
    diff = 1e10
    num_iter = 1
    while diff > 1e-3 and num_iter < max_iter:
        Y = np.sum(((pts_in[np.newaxis, ...] - pts_in[:, np.newaxis, :]) ** 2), axis=2)
        K = np.maximum(bandwidth**2 - Y, np.zeros(Y.shape))
        if weights is not None:
            K = K * weights
        row_sums = K.sum(axis=0, keepdims=True)
        P = K / (row_sums + 1e-10)
        P = P.transpose()
        pts_in_prim = 0.3 * (np.matmul(P, pts_in) - pts_in) + pts_in
        diff = np.sqrt(np.sum((pts_in_prim - pts_in)**2))
        pts_in = pts_in_prim
        num_iter += 1
    return pts_in

def get_id_list(folder):
    files = [f for f in os.listdir(folder) if isfile(join(folder, f))]
    id_list = []
    for file in files:
        name = file.split('.')[0]
        id = name.split('_')[0]
        if id not in id_list:
            id_list.append(id)
    return id_list

def find_root_id(root_point, joints):
    min_dinst = 1000000
    min_id = 0
    for i in range(len(joints)):
        joint = joints[i]
        d = np.linalg.norm(joint-root_point)
        if d < min_dinst:
            min_id = i
            min_dinst = d
    return min_id
def create_input_data(mesh_file):
    # load obj
    # mesh_file = './data/obj/'+id+'.obj'
    mesh, mesh_v, mesh_f = Mesh.ReadObj(mesh_file)
    mesh.compute_vertex_normals()
    mesh_vertex_normals = np.asarray(mesh.vertex_normals)
    # mesh_v, translate, scale = Mesh.NormalizeVertice(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(Vector3d(mesh_v), Vector3i(mesh_f))
    Mesh.WriteObj(mesh_file.replace(".obj", "_normalized.obj"), mesh_v, mesh_f)

    # v_data -> [x, y, z, nx, ny, nz]
    v_data = np.concatenate((mesh_v, mesh_vertex_normals), axis=1)
    v_data = torch.from_numpy(v_data).float()

    print("     gathering topological edges.")
    tpl_e = Mesh.get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=mesh_v.shape[0])

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = Mesh.CalculateSufaceGeodesic(mesh)

    print("     gathering geodesic edges.")
    geo_e = Mesh.get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=mesh_v.shape[0])

    # batch
    batch = torch.zeros(len(v_data), dtype=torch.long)

    # voxel
    if not os.path.exists(mesh_file.replace('.obj', '_normalized.binvox')):
        if platform == "linux" or platform == "linux2":
            os.system("./binvox -d 88 -pb " + mesh_file.replace(".obj", "_normalized.obj"))
        elif platform == "win32":
            os.system("binvox.exe -d 88 " + mesh_file.replace(".obj", "_normalized.obj"))
        else:
            raise Exception('Sorry, we currently only support windows and linux.')

    with open(mesh_file.replace('.obj', '_normalized.binvox'), 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)

    data = Data(x=v_data[:, 3:6], pos=v_data[:, 0:3], tpl_edge_index=tpl_e, geo_edge_index=geo_e, batch=batch)
    data = data.to(device)
    return mesh, data, vox

def joint_shift(data, joint_net):
    data_displacement = joint_net(data)
    distance = data_displacement.data.cpu().numpy()
    np.savetxt('./data/save/distance.txt', distance)

    y_pred = data_displacement + data.pos
    y_pred = y_pred.data.cpu().numpy()
    np.savetxt('./data/save/shift.txt', y_pred)
    return y_pred

def joint_cluster(shift_joints, data, joint_mask_net):
    # run joint mask net
    joint_prob = joint_mask_net(data)
    joint_prob_sigmoid = torch.sigmoid(joint_prob)
    joint_prob_sigmoid = joint_prob_sigmoid.data.cpu().numpy()
    np.savetxt('./data/body_save/att.txt', joint_prob_sigmoid)

# fintune
    bandwidth = 0.07
    threshold =  1e-6

    y_pred = meanshift_cluster(shift_joints,bandwidth, joint_prob_sigmoid)
    y_pred_np = y_pred

    Y_dist = np.sum(((y_pred_np[np.newaxis, ...] - y_pred_np[:, np.newaxis, :]) ** 2), axis=2)
    density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
    density = np.sum(density, axis=0)
    density_sum = np.sum(density)
    y_pred = y_pred_np[density / density_sum > threshold]
    
    density = density[density / density_sum > threshold]
    pred_joints = nms_meanshift(y_pred_np, density, bandwidth)
    # pred_joints = y_pred
    # pred_joints, _ = flip(pred_joints)
    # np.savetxt('./data/save/cluster.txt', pred_joints)
    return pred_joints

class BoneNode:
    def __init__(self, old_id, new_id) -> None:
        self.old_id = old_id
        self.new_id = new_id
        self.parnet = None
        self.children = []
        self.is_to_root = False
    def Check_to_root(self):
        if self.parnet == None:
            self.is_to_root = True
            return True
        if self.parnet.Check_to_root():
            self.is_to_root = True
        return self.is_to_root

def my_tree_generate(cost_matrix, root_id, joints):
    joint_count = cost_matrix.shape[0]
    # print(cos_matrix.shape[0])
    #sort by distance
    root_pos = joints[root_id]
    # distances = []
    # for pos in joints:
    #     d = np.linalg.norm(pos-root_pos)
    #     distances.append(d)
        
    # sort_indexs = np.argsort(distances)
    # old_to_new = np.zeros(joint_count)
    # for i in range(joint_count):
    #     old = sort_indexs[i]
    #     old_to_new[old] = i


    adj = np.zeros((joint_count, joint_count))
    for i in range(joint_count):
        min_cost = 1000000000000000000
        # if i == root_id:
        #     continue
        connect_index = -1
        for j in range(0,i):
            if j == i:
                continue
            if cost_matrix[i][j]<min_cost:
                min_cost = cost_matrix[i][j]
                connect_index = j
        if connect_index == -1: # not find
            continue
        adj[i][connect_index] = 1
        adj[connect_index][i] = 1

    return adj


def sort_joint(joints, root_id):
    joint_count = joints.shape[0]
    root_pos = joints[root_id]
    distances = []
    for pos in joints:
        d = np.linalg.norm(pos-root_pos)
        distances.append(d)
        
    sort_indexs = np.argsort(distances)
    # sort_d = np.sort(distances)
    # print(sort_indexs[0], ',', root_id)
    # print(sort_indexs)
    # print(sort_d)
    
    sort_list = []
    for i in range(joint_count):
        sort_list.append(joints[sort_indexs[i]])
    sort_joints = np.array(sort_list)
    return(sort_joints)



if __name__ == '__main__':
    id_list = get_id_list('./data/obj')

    print("Start")

    print("loading all networks...")

    joint_net = JointPredNet(out_channels=3, input_normal= False, arch='jointnet')
    joint_net.to(device)
    joint_net.eval()
    joint_distance_net_checkpoint = torch.load('checkpoints/pretrain_jointnet/model_best.pth.tar')
    joint_net.load_state_dict(joint_distance_net_checkpoint['state_dict'])

    joint_mask_net = JointPredNet(out_channels=1, input_normal= False, arch='masknet')
    joint_mask_net.to(device)
    joint_mask_net.eval()
    joint_mask_net_checkpoint = torch.load('checkpoints/pretrain_masknet/model_best.pth.tar')
    joint_mask_net.load_state_dict(joint_mask_net_checkpoint['state_dict'])

    # bone_net = BONENET()
    # bone_net.to(device)
    # bone_net.eval()
    # bone_net_checkpoint = torch.load('checkpoints/hand_bonenet/model_best.pth.tar')
    # bone_net.load_state_dict(bone_net_checkpoint['state_dict'])

    id = '239'
# load obj
    mesh_file = './data/239.obj'
    mesh, data, vox = create_input_data(mesh_file)
    joints = joint_shift(data, joint_net)

    joints = joint_cluster(joints, data,joint_mask_net)
    

    # show 
        
        
    view = sf.CreateDisplayWindow()
    mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mesh_ls.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for i in range(len(mesh_ls.lines))])
    view.add_geometry(mesh_ls)
    sf.DrawVertices(view, joints,radius=0.004)
    # sf.DrawSkeleton(view,pred_joints,adj)

    ctr = view.get_view_control()
    ctr.set_front([0.0001,1,0])
    view.run()

    # ctr = view.get_view_control()
    # ctr.set_front([0.0001,1,0])
    # view.run()
    # view.capture_screen_image('./shot/'+id+'.png', True)
    # view.clear_geometries()


   
    
    # sf.WriteVerticeInObj('./data/out.obj', y_pred)


