import cv2 as cv
import numpy as np
from scipy.linalg import sqrtm, cholesky
import matplotlib.pyplot as plt
import pyvista as pv


def observation_matrix(images):
    keypoints_sets = []
    matched_points = []
    sift = cv.SIFT_create()
    keypoints0, descriptors0 = sift.detectAndCompute(images[0], None)
    keypoints_sets.append(keypoints0)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    for i in range(1,len(images)-2):
        keypoints1, descriptors1 = sift.detectAndCompute(images[i], None)
        keypoints_sets.append(keypoints1)
        matches = flann.knnMatch(descriptors0,descriptors1,k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        matched_points.append([(keypoints0[m.queryIdx].pt, keypoints1[m.trainIdx].pt) for m in good_matches])
        
        descriptors0 = descriptors1
        keypoints0 = keypoints1

    F = len(images) # no. of frames 
    N = max([len(x) for x in matched_points])

    W = np.zeros((2 * F, N))
    
    for i, matches in enumerate(matched_points):
        for j, (pt1, pt2) in enumerate(matches):
            W[2 * i, j] = pt1[0] # x coordinate 
            W[2 * i + 1, j] = pt1[1]  # y coordinate
    
    W_mean = np.mean(W, axis=1, keepdims=True)
    W_tilde = W - W_mean
    
    return W_tilde

def orthogonality_constraint(W):
    U, Sigma, VT = np.linalg.svd(W)
    U_prime = U[:, :3]
    Sigma_prime = np.diag(Sigma.flatten()[:3])
    V_prime = VT[:3, :]
    R_hat = U_prime @ sqrtm(Sigma_prime)
    S_hat = sqrtm(Sigma_prime) @ V_prime
    # Metric constraint - solving for Q
    L = []
    Ro = []
    for i in range(R_hat.shape[0] // 2):
        r1 = R_hat[2 * i]
        r2 = R_hat[2 * i + 1]
        
        #orthonormality constraint
        L.append([r1[0] ** 2, 2 * r1[0] * r1[1], 2 * r1[0] * r1[2], r1[1] ** 2, 2 * r1[1] * r1[2], r1[2] ** 2])
        L.append([r2[0] ** 2, 2 * r2[0] * r2[1], 2 * r2[0] * r2[2], r2[1] ** 2, 2 * r2[1] * r2[2], r2[2] ** 2])
        L.append([r1[0] * r2[0], r1[0] * r2[1] + r1[1] * r2[0], r1[0] * r2[2] + r1[2] * r2[0], r1[1] * r2[1],
                  r1[1] * r2[2] + r1[2] * r2[1], r1[2] * r2[2]])
        
        Ro.append(1)
        Ro.append(1)
        Ro.append(0)
    
    L = np.array(L)
    Ro = np.array(Ro)
    QQT = np.linalg.lstsq(L, Ro, rcond=None)[0]
    
    LT = np.array([[QQT[0], QQT[1], QQT[2]],
                   [QQT[2], QQT[3], QQT[4]],
                   [QQT[4], QQT[5], QQT[0]]])
    print(LT)
    # LT = np.array([[QQT[0], QQT[1], QQT[2]],
    #                [QQT[1], QQT[3], QQT[4]],
    #                [QQT[2], QQT[4], QQT[5]]])

    # Q is obtained through cholesky

    Q = cholesky(LT)
    
    R = R_hat @ Q
    S = np.linalg.inv(Q) @ S_hat
    
    return R, S    

def plot_cloud(S):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(S[0, :], S[1, :], S[2, :], c='r', marker='.')
    ax.set_title('Reconstructed 3D Structure')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()  

def d3_mesh(S):
    cloud = pv.PolyData(S.reshape(S.shape[1],S.shape[0]))
    cloud.plot()
    volume = cloud.delaunay_3d(alpha=100)
    shell = volume.extract_geometry()
    shell.plot()


if __name__ == '__main__':
    # image_files = ['./input_data/r1 (1).jpeg','./input_data/r1 (2).jpeg','./input_data/r1 (3).jpeg']
    # images = [cv.imread(file, cv.COLOR_BGR2GRAY) for file in image_files]
    cam = cv.VideoCapture('vid2.mp4')
    ret, frame = cam.read()
    images = []
    while ret:
        ret, frame = cam.read()
        images.append(frame)
    
    cam.release()

    W = observation_matrix(images)

    R, S = orthogonality_constraint(W)
    print(S.shape)
    plot_cloud(S)
    d3_mesh(S)
