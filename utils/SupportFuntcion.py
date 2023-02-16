import open3d as o3d
import numpy as np

def CreateDisplayWindow():
    view = o3d.visualization.Visualizer()
    view.create_window()
    return view

def Vector3d(np_points):
    return o3d.utility.Vector3dVector(np_points)

def Vector3i(np_points):
    return o3d.utility.Vector3iVector(np_points)

def Mesh(v_np, f_np):
    '''
    Open3d Mesh
    '''
    
    return o3d.geometry.TriangleMesh(Vector3d(v_np),Vector3i(f_np))

def drawCone(bottom_center, top_position, radius = 0.007, color=[0.6, 0.6, 0.9]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=np.linalg.norm(top_position - bottom_center)+1e-6)
    line1 = np.array([0.0, 0.0, 1.0])
    line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center)+1e-6)
    v = np.cross(line1, line2)
    c = np.dot(line1, line2) + 1e-8
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
    if np.abs(c + 1.0) < 1e-4: # the above formula doesn't apply when cos(∠(𝑎,𝑏))=−1
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    T = bottom_center + 5e-3 * line2
    #print(R)
    cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    return cone

def drawSphere(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def DrawVertices(view, vertices, radius = 0.008, color = [1.0, 0.0, 0.0]):
    for v in vertices:
        view.add_geometry(drawSphere(v, radius, color))

def DrawSkeleton(view, joint, adj_mat, radius = 0.007, color = [0.6,0.6,0.9]):
    for i in range(adj_mat.shape[0]):
        for j in range(i, adj_mat.shape[0]):
            if adj_mat[i,j] == 1:
                parent = joint[i]
                child = joint[j]
                view.add_geometry(drawCone(parent,child,radius,color))


# Here normalize means to fit this mesh into a box of (-0.5, 0, -0.5) to (0.5, 1, 0.5). 
def NormalizeMesh(mesh):
    v = np.asarray(mesh.vertices)

    # find the longest distance of x y z
    dims = [max(v[:, 0]) - min(v[:, 0]),
            max(v[:, 1]) - min(v[:, 1]),
            max(v[:, 2]) - min(v[:, 2])]

    # scale = one divide to the longest distance
    scale = 1.0 / max(dims)

    # find the center point, but since the bounding of y-axis is 0~1, the pivot of y is y_min
    pivot = np.array([  (min(v[:, 0]) + max(v[:, 0])) / 2, 
                        min(v[:, 1]),
                        (min(v[:, 2]) + max(v[:, 2])) / 2])

    # Shift all vertices according to pivots
    v[:, 0] -= pivot[0]
    v[:, 1] -= pivot[1]
    v[:, 2] -= pivot[2]

    # Scale the pivot
    v *= scale
    normalize_mesh = o3d.geometry.TriangleMesh(Vector3d(v),Vector3i(mesh.triangles))
    return normalize_mesh, pivot, scale

def ReadObj(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    v = f = None
    if mesh:
        v = np.array(mesh.vertices)
        f = np.array(mesh.triangles)
    return mesh, v, f


def WriteObjM(file_path, mesh):
    o3d.io.write_triangle_mesh(file_path, mesh, write_triangle_uvs=True)

def WriteObj(file_path, v, f):
    mesh = o3d.geometry.TriangleMesh(Vector3d(v),Vector3i(f))
    o3d.io.write_triangle_mesh(file_path, mesh, write_triangle_uvs=True)

def WriteVerticeInObj(file_path, v):
    file = open(file_path, 'w')
    for i in range(len(v)):
        file.write('v ' + str(v[i,0]) + ' ' + str(v[i,1]) + ' ' + str(v[i,2])+ '\n')

    file.close()




def ConstructCube(len = 1.0):
    v = np.array([  0,0,0,
                    len,0,0,
                    len,len,0,
                    0,len,0,
                    0,len,len,
                    0,0,len,
                    len,0,len,
                    len,len,len])
    f = np.array([  0,3,2,
                    2,1,0,
                    0,1,6,
                    6,5,0,
                    2,7,6,
                    6,1,2,
                    2,3,4,
                    4,7,2,
                    4,3,0,
                    0,5,4,
                    4,5,6,
                    6,7,4])
    f = f.reshape(12,3)

    v = v.reshape(8,3)
    return v,f
