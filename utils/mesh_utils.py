import open3d as o3d
import numpy as np
import time
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra


def Vector3d(np_points):
    return o3d.utility.Vector3dVector(np_points)

def Vector3i(np_points):
    return o3d.utility.Vector3iVector(np_points)

def WriteObj(file_path, v, f):
    mesh = o3d.geometry.TriangleMesh(Vector3d(v),Vector3i(f))
    o3d.io.write_triangle_mesh(file_path, mesh)

def ReadObj(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    v = f = None
    if mesh:
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
    return mesh, v, f

def NormalizeVertice(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale

def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        if len(neighbor_ids)!=0:
            neighbor_ids = np.concatenate(neighbor_ids, axis=0)
            edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index

def get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index

def CalculateSufaceGeodesic(mesh):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    time1 = time.time()
    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                    return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    time2 = time.time()
    print('surface geodesic calculation: {} seconds'.format((time2 - time1)))
    return surface_geodesic