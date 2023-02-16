import torch
import numpy as np
import open3d as o3d
import os

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
from utils.mst_utils import sample_on_bone, inside_check, increase_cost_for_outside_bone, primMST
from utils.tree_utils import TreeNode
from utils.meanshift_utils import nms_meanshift, meanshift_cluster

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HandNet:
    def __init__(self) -> None:
        self.is_loaded = False
        self.root_point = np.array([ -0.5, 0.32, -0.12202])
        pass

    def LoadModel(self):
        print("Load HandNet Models!")
        self.joint_net = JointPredNet(out_channels=3, input_normal= False, arch='jointnet')
        self.joint_net.to(device)
        self.joint_net.eval()
        joint_distance_net_checkpoint = torch.load('checkpoints/hand_jointnet/model_best.pth.tar')
        self.joint_net.load_state_dict(joint_distance_net_checkpoint['state_dict'])

        self.joint_mask_net = JointPredNet(out_channels=1, input_normal= False, arch='masknet')
        self.joint_mask_net.to(device)
        self.joint_mask_net.eval()
        joint_mask_net_checkpoint = torch.load('checkpoints/hand_masknet/model_best.pth.tar')
        self.joint_mask_net.load_state_dict(joint_mask_net_checkpoint['state_dict'])

        self.bone_net = BONENET()
        self.bone_net.to(device)
        self.bone_net.eval()
        bone_net_checkpoint = torch.load('checkpoints/hand_bonenet/model_best.pth.tar')
        self.bone_net.load_state_dict(bone_net_checkpoint['state_dict'])
        self.is_loaded = True
    
    def joint_shift(self, data, joint_net):
        data_displacement = joint_net(data)
        distance = data_displacement.data.cpu().numpy()
        # np.savetxt('./data/save/distance.txt', distance)
        y_pred = data_displacement + data.pos
        y_pred = y_pred.data.cpu().numpy()
        # np.savetxt('./data/save/shift.txt', y_pred)
        return y_pred
    


    def joint_cluster(self, shift_joints, data, joint_mask_net):
        # run joint mask net
        joint_prob = joint_mask_net(data)
        joint_prob_sigmoid = torch.sigmoid(joint_prob)
        joint_prob_sigmoid = joint_prob_sigmoid.data.cpu().numpy()
    # fintune
        # bandwidth = 0.045 #239
        # threshold =  1e-5
        bandwidth = 0.1
        threshold =  1e-4

        y_pred = meanshift_cluster(shift_joints,bandwidth, joint_prob_sigmoid)
        y_pred_np = y_pred

        Y_dist = np.sum(((y_pred_np[np.newaxis, ...] - y_pred_np[:, np.newaxis, :]) ** 2), axis=2)
        density = np.maximum(bandwidth ** 2 - Y_dist, np.zeros(Y_dist.shape))
        density = np.sum(density, axis=0)
        density_sum = np.sum(density)
        y_pred = y_pred_np[density / density_sum > threshold]
        
        density = density[density / density_sum > threshold]
        pred_joints = nms_meanshift(y_pred_np, density, bandwidth)
        # pred_joints, _ = flip(pred_joints)
        # np.savetxt('./data/save/cluster.txt', pred_joints)
        return pred_joints

    def sort_joint(self, joints, root_id):
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
    
    def create_input_data(self, mesh_file):
        # load obj
        # mesh_file = './data/obj/'+id+'.obj'
        mesh, mesh_v, mesh_f = Mesh.ReadObj(mesh_file)
        mesh.compute_vertex_normals()
        mesh_vertex_normals = np.asarray(mesh.vertex_normals)
        mesh_v, translate, scale = Mesh.NormalizeVertice(mesh_v)
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
    
    def generate_bone_net_input(self, data, vox, pred_joints):
        pairs = list(it.combinations(range(pred_joints.shape[0]), 2))
        pair_attr = []
        for pr in pairs:
            dist = np.linalg.norm(pred_joints[pr[0]] - pred_joints[pr[1]])
            bone_samples = sample_on_bone(pred_joints[pr[0]], pred_joints[pr[1]])
            bone_samples_inside, _ = inside_check(bone_samples, vox)
            outside_proportion = len(bone_samples_inside) / (len(bone_samples) + 1e-10)
            attr = np.array([dist, outside_proportion, 1])
            pair_attr.append(attr)
        pairs = np.array(pairs)
        pair_attr = np.array(pair_attr)
        pairs = torch.from_numpy(pairs).float()
        pair_attr = torch.from_numpy(pair_attr).float()
        pred_joints = torch.from_numpy(pred_joints).float()
        joints_batch = torch.zeros(len(pred_joints), dtype=torch.long)
        pairs_batch = torch.zeros(len(pairs), dtype=torch.long)

        data.joints = pred_joints
        data.pairs = pairs
        data.pair_attr = pair_attr
        data.joints_batch = joints_batch
        data.pairs_batch = pairs_batch
        data = data.to(device)
        return data

    def predict_connection(self, data, bone_net, root_id, vox):
        with torch.no_grad():
            connect_prob, _ = bone_net(data, permute_joints=False)
            connect_prob = torch.sigmoid(connect_prob)

        pair_idx = data.pairs.long().data.cpu().numpy()
        prob_matrix = np.zeros((len(data.joints), len(data.joints)))
        prob_matrix[pair_idx[:, 0], pair_idx[:, 1]] = connect_prob.data.cpu().numpy().squeeze()
        prob_matrix = prob_matrix + prob_matrix.transpose()

        cost_matrix = -np.log(prob_matrix + 1e-10)
        pred_joints = data.joints.data.cpu().numpy()
        cost_matrix = increase_cost_for_outside_bone(cost_matrix, pred_joints, vox)


        parent, key = primMST(cost_matrix, root_id)
        joint_count = len(parent)
        adj = np.zeros((joint_count, joint_count))

        for i in range(joint_count):
            p_id = parent[i]
            if p_id == -1:
                continue
            adj[i][p_id] = 1
            adj[p_id][i] = 1
        return adj
    
    def Run(self, mesh_file):
        if not self.is_loaded:
            print('Model is not loaded!')
            return
        
        mesh, data, vox = self.create_input_data(mesh_file)
        
        joints = self.joint_shift(data, self.joint_net)
        joints = self.joint_cluster(joints, data, self.joint_mask_net)


        min_dinst = 1000000
        min_id = 0
        for i in range(len(joints)):
            joint = joints[i]
            d = np.linalg.norm(joint-self.root_point)
            if d < min_dinst:
                min_id = i
                min_dinst = d
        root_id =  min_id
        joints = self.sort_joint(joints, root_id)

        data = self.generate_bone_net_input(data, vox, joints)
        adj = self.predict_connection(data, self.bone_net, root_id, vox)

        # view
        view = sf.CreateDisplayWindow()
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_ls.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(len(mesh_ls.lines))])
        view.add_geometry(mesh_ls)
        sf.DrawVertices(view, joints)
        sf.DrawSkeleton(view,joints,adj)

        ctr = view.get_view_control()
        ctr.set_front([0.0001,1,0])
        view.run()

        return joints, adj


        

if __name__ == '__main__':
    net = HandNet()
    net.LoadModel()
    id = '8290L'
    mesh_file = './data/obj/'+id+'.obj'
    net.Run(mesh_file)
    pass


