import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from slam_vo.datasets.euroc import CameraIntrinsics

EUROC_CAM = CameraIntrinsics(fx=458.654, fy=457.296, cx=367.215, cy=248.375, width=752, height=480, k1=0, k2=0, p1=0, p2=0)
K = EUROC_CAM.K.astype(np.float64)

rng = np.random.default_rng(42)
pts3d = rng.uniform([-3,-2,4],[3,2,10],size=(100,3))
R1,_ = cv2.Rodrigues(np.array([0.08,0.15,0.03]))
T_cw1 = np.eye(4); T_cw1[:3,:3]=R1; T_cw1[:3,3]=[0.5,0.05,0.0]

def project(T, pts):
    p = (T[:3,:3]@pts.T + T[:3,3:4])
    pi = K@p
    return (pi[:2]/pi[2]).T.astype(np.float32)

pts0 = project(np.eye(4), pts3d) + rng.normal(0,0.3,(100,2)).astype(np.float32)
pts1 = project(T_cw1, pts3d) + rng.normal(0,0.3,(100,2)).astype(np.float32)

E, e_mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
print(f"E.shape={E.shape}, e_mask.sum()={e_mask.sum()}")

# recoverPose without mask first
n_inliers1, R1o, t1, pm1 = cv2.recoverPose(E, pts0, pts1, K)
print(f"Without mask: n_inliers={n_inliers1}, pm1 unique={np.unique(pm1)}")

# recoverPose with mask
n_inliers2, R2o, t2, pm2 = cv2.recoverPose(E, pts0, pts1, K, mask=e_mask.copy())
print(f"With mask: n_inliers={n_inliers2}, pm2 unique={np.unique(pm2.ravel())}")
print(f"pm2 shape={pm2.shape}, dtype={pm2.dtype}")
print(f"pm2[:5]={pm2.ravel()[:5]}")
