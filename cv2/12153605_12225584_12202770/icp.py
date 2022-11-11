from SupplementalCode.example import *
from matching import *
import numpy as np
from numpy import linalg as LA
import open3d as o3d
import random
import argparse
import time


MATCHING_FN = {
    'euclidean': "euclidean_matching",
    'kdtree': "kdtree_matching",
    'zbuffer': "zbuffer_matching"
}


def c2h(mat):
    return np.vstack([mat, np.ones((1, mat.shape[1]))])


def h2c(mat):
    return mat[:-1] / mat[-1]


def calc_RMS(X, Y):
    return np.sqrt(np.mean(np.sum((X - Y)**2, axis=0)))


class point_sets:
    def __init__(self, source, target, sampling=None, n_samples=None, N=4):
        self.source = source
        self.target = target
        self.sampling = sampling
        self.n_samples = n_samples

        self.Ns = source.shape[1]
        self.Nt = target.shape[1]

        if sampling == 'uniform':
            self.uf_sidc = np.random.choice(self.Ns, n_samples, replace=False)
            self.uf_tidc = np.random.choice(self.Nt, n_samples, replace=False)
        elif sampling == 'multiresolution':
            self.N = N
            self.dS = self.source.shape[1] // 150
            self.dT = self.target.shape[1] // 150

    def get_sets(self, res_step=False):
        # Return points as is
        if self.sampling is None:
            return self.source, self.target
        # Sample N from both clouds and return same at every iteration.
        elif self.sampling == 'uniform':
            return self.source[:, self.uf_sidc], self.target[:, self.uf_tidc]
        # Return N random samples at each iteration.
        elif self.sampling == 'random':
            src_idc = random.sample(range(self.Ns), self.n_samples)
            tgt_idc = random.sample(range(self.Nt), self.n_samples)
            return self.source[:, src_idc], self.target[:, tgt_idc]
        # Return subsamples according to multiresolution scheme.
        elif self.sampling == 'multiresolution':
            if res_step:
                self.dS = self.dS // self.N
                self.dT = self.dT // self.N
                if self.dS < 1:
                    self.dS = 1
                if self.dT < 1:
                    self.dT = 1
            return self.source[:, ::self.dS], self.target[:, ::self.dT]


def ICP(source, target, tolerance=1e-4, sampling=None, n_samples=None,
        matching="euclidean", N=4, mr_tolerance=3e-3):
    matching_fn = MATCHING_FN[matching]
    D = source.shape[0]
    R = np.eye(D)
    t = np.zeros((D, 1))
    RMS = np.inf
    set_loader = point_sets(source, target, sampling, n_samples, N)
    i = 1
    res_step = False
    while True:
        src, tgt = set_loader.get_sets(res_step=res_step)
        # Transform source point and find correspondences
        src, tgt_sorted, P = eval(matching_fn + '(src, tgt, R, t)')
        # Compute RMS and stop if converged
        RMS_1 = RMS
        RMS = calc_RMS(P, tgt_sorted)
        print(f"({i}) RMS:", RMS)
        if abs(RMS_1 - RMS) < tolerance:
            print("CONVERGED")
            break
        elif abs(RMS_1 - RMS) < mr_tolerance:
            res_step = True
        # Compute covariance matrix and get SVD
        Cs = np.mean(src, axis=1, keepdims=True)
        Ct = np.mean(tgt_sorted, axis=1, keepdims=True)
        X = src - Cs
        Y = tgt_sorted - Ct
        S = X @ Y.T
        U, _, Vt = LA.svd(S)

        # Update the transformation
        diag = np.diag([*np.ones(U.shape[1] - 1), LA.det(Vt.T@U.T)])

        R = Vt.T @ diag @ U.T
        t = Ct - R @ Cs
        i += 1
    return R, t


def draw(A1, A2, A3):
    """
    This function can be used to visualize point clouds.
    If given A1, A2 and A3 will be colored Green, Red and Blue
    respectively. Set argument to None to ignore it.
    """
    R = np.array(np.array([1, 0, 0], dtype=np.float64)).reshape(3,1)
    G = np.array(np.array([0, 1, 0], dtype=np.float64)).reshape(3,1)
    B = np.array(np.array([0, 0, 1], dtype=np.float64)).reshape(3,1)

    vis_src = o3d.geometry.PointCloud()
    vis_tgt = o3d.geometry.PointCloud()
    vis_trs = o3d.geometry.PointCloud()

    clouds = []
    if A1 is not None:
        vis_src.points = o3d.utility.Vector3dVector(A1.T)
        vis_src = vis_src.paint_uniform_color(G)
        clouds.append(vis_src)
    if A2 is not None:
        vis_tgt.points = o3d.utility.Vector3dVector(A2.T)
        vis_tgt = vis_tgt.paint_uniform_color(R)
        clouds.append(vis_tgt)
    if A3 is not None:
        vis_trs.points = o3d.utility.Vector3dVector(A3.T)
        vis_trs = vis_trs.paint_uniform_color(B)
        clouds.append(vis_trs)

    o3d.visualization.draw_geometries(clouds)
    return None

def open_human_data(file):
    pcd = o3d.io.read_point_cloud(file)
    # ## convert into ndarray

    pcd_arr = np.asarray(pcd.points)

    z_values = pcd_arr[:,2]
    median = np.median(z_values)
    filter = (median+0.5 > z_values) & (z_values > median-0.5)

    pcd_arr_cleaned = pcd_arr[filter]

    return pcd_arr_cleaned.T

def human_reconstruction():
    # set stepsize
    S = 1
    N = 49-S
    R_list = []
    t_list = []
    cloud_list = []

    for i in range(0,N,S):
        # adjacent frames implementation
        source_points = open_human_data("Data/data/" + '{0:010d}'.format(i) + ".pcd")
        target_points = open_human_data("Data/data/" + '{0:010d}'.format(i+S) + ".pcd")

        Ns = source_points.shape[1]
        Nt = target_points.shape[1]

        if args.n_samples is not None:
            if Nt < args.n_samples:
                raise ValueError(f"Number of samples larger than target size ({Nt}).")
            elif Ns < args.n_samples:
                raise ValueError(f"Number of samples larger than source size ({Ns}).")



        R, t = ICP(source_points, target_points, tolerance=args.tolerance,
                   sampling=args.sampling, n_samples=args.n_samples,
                   matching=args.matching)

        R_list.append(R)
        t_list.append(t)


        # apply the found transformation to all precious clouds to maintain
        # the same coordinate system
        for i, cloud in enumerate(cloud_list):
            cloud_list[i] = R @ cloud + t

        # add the new cloud
        cur_transformed = R @ source_points + t
        cloud_list.append(cur_transformed)

    result = cloud_list[0]
    for cloud in cloud_list[1:]:
        result = np.hstack((result, cloud))

    # sample the result for efficient viewing.
    sample_res = True

    if sample_res:
        Ns = result.shape[1]
        n_samples = Ns//(10//S)
        print(Ns)
        src_idc = random.sample(range(Ns), n_samples)
        result = result[:,src_idc]

    # draw
    vis_src = o3d.geometry.PointCloud()

    vis_src.points = o3d.utility.Vector3dVector(result.T)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    o3d.visualization.draw_geometries([vis_src, frame])

def human_rec_merge():
    # merge on the go implementation
    S = 4
    N = 49-S
    t_list = []
    R_list = []
    cloud_list = []
    #
    for i in range(0,N,S):
        target_points = open_human_data("Data/data/" + '{0:010d}'.format(i+S) + ".pcd")

        # now we keep expanding the same result, so no iteration on the source
        if i == 0:
            source_points = open_human_data("Data/data/" + '{0:010d}'.format(i) + ".pcd")
    #
        Ns = source_points.shape[1]
        Nt = target_points.shape[1]
    #
        if args.n_samples is not None:
            if Nt < args.n_samples:
                raise ValueError(f"Number of samples larger than target size ({Nt}).")
            elif Ns < args.n_samples:
                raise ValueError(f"Number of samples larger than source size ({Ns}).")

        # get as much samples as possible to ensure some overlap
        sampling = 'uniform'
        tolerance=1e-6
        n_samples = min(Ns, Nt)

        R, t = ICP(source_points, target_points, tolerance=tolerance,
                   sampling=sampling, n_samples=n_samples,
                   matching=args.matching)
                   #
        R_list.append(R)
        t_list.append(t)

        # transform the source and combine
        cur_transformed = R @ source_points + t

        source_points = np.hstack((cur_transformed, target_points))

    result = source_points

    # sample the result for better viewing
    sample_res = True

    if sample_res:
        Ns = result.shape[1]
        n_samples = Ns//(10//S)
        src_idc = random.sample(range(Ns), n_samples)
        result = result[:,src_idc]

    # draw
    vis_src = o3d.geometry.PointCloud()
    #
    vis_src.points = o3d.utility.Vector3dVector(result.T)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.025)
    #
    o3d.visualization.draw_geometries([vis_src, frame])
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="euclidean", type=str)
    parser.add_argument("--tolerance", default=1e-4, type=float)
    parser.add_argument("--sampling", default=None, type=str)
    parser.add_argument("--n_samples", default=None, type=int)
    args = parser.parse_args()

    source_points, target_points = open_wave_data()
    # print(source_points.shape, target_points.shape)
    # source_points = source_points[:,::2]
    # target_points = target_points[:,::2]
    # draw(source_points, target_points, None)

    Ns = source_points.shape[1]
    Nt = target_points.shape[1]
    if args.n_samples is not None:
        if Nt < args.n_samples:
            raise ValueError(f"Number of samples larger than target size ({Nt}).")
        elif Ns < args.n_samples:
            raise ValueError(f"Number of samples larger than source size ({Ns}).")
    start = time.time()
    R, t = ICP(source_points, target_points, tolerance=args.tolerance,
               sampling=args.sampling, n_samples=args.n_samples,
               matching=args.matching)
    finish = time.time()
    print("Elapsed time (sec): {}".format(finish - start))
    source_transformed = R @ source_points + t
    draw(None, target_points, source_transformed)

    # 3.1
    # human_reconstruction()

    #  3.2
    # human_rec_merge()
