import trimesh
import torch
import numpy as np
import json
from smplx import SMPLX

# used to parse the json
def load_landmark_coords(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# find the best-fitting transformation that aligns one set of 3D points to another
# this function does not deform the body or move the joints, it just moves the entire mesh!!!
def umeyama_alignment(X, Y): # X is source, Y is target
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    X0 = X - mu_X
    Y0 = Y - mu_Y
    U, S, Vt = np.linalg.svd(X0.T @ Y0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = mu_Y - R @ mu_X
    return R, t # R: rotate SMPL to face same direction, t: fit the SMPL translation

def apply_transform(vertices, R, t):
    return (vertices @ R.T) + t

# load landmarks
character_landmarks = load_landmark_coords("index/character.json")
smpl_landmarks = load_landmark_coords("index/smpl.json")
smpl_index = load_landmark_coords("index/smpl_index.json")

common_keys = list(set(character_landmarks.keys()) & set(smpl_landmarks.keys()))


"""this step can get the transformation info to align them"""
char_pts = np.array([character_landmarks[i] for i in common_keys])
smpl_pts = np.array([smpl_landmarks[i] for i in common_keys])

R, t = umeyama_alignment(smpl_pts, char_pts) # important step to align the SMPL mesh with the character mesh
# then convert them to torch
R = torch.tensor(R, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.float32)

new_smpl_model = SMPLX(model_path='models/smplx', gender='neutral', batch_size=1)

# initialize pose and shape
pose = torch.zeros([1, 63], dtype=torch.float32, requires_grad=True)
shape = torch.zeros([1, 10], dtype=torch.float32, requires_grad=True)

""" make the smpl be A pose first (maybe unnecessary)"""
# left_shoulder_idx = 15 # left shoulder
# right_shoulder_idx = 16 # right shoulder

# # rotate -45 degrees around z for left shoulder and 45 degrees for right shoulder
# angle_rad = np.radians(55)
# pose[0, left_shoulder_idx * 3 : left_shoulder_idx * 3 + 3] = torch.tensor([0.0, 0.0, -angle_rad])
# pose[0, right_shoulder_idx * 3 : right_shoulder_idx * 3 + 3] = torch.tensor([0.0, 0.0, angle_rad])


"""
Optimization-Based Pose Fitting
Loss: 1/n * ∑||SMPL vertex - Character vertex||²
"""
# optimizer using Adam
optimizer = torch.optim.Adam([shape, pose], lr=0.1)

# target character landmarks
char_pts_torch = torch.tensor(char_pts, dtype=torch.float32)

# get the indices from the smpl_index.json
smpl_indices = torch.tensor([smpl_index[i] for i in common_keys], dtype=torch.long)

# optimization loop !!!!!!!!!
for i in range(320):
    smpl_output = new_smpl_model(
        global_orient = pose[:, :3].reshape(1, 1, 3),
        body_pose = pose.reshape(1, 21, 3),
        betas = shape,
        pose2rot = True
    )

    vertices = smpl_output.vertices[0]

    vertices = apply_transform(vertices, R, t) # apply the transformation to the vertices only once

    model_landmarks = vertices[smpl_indices]
    loss = torch.mean((model_landmarks - char_pts_torch) ** 2)
    # add regularization term to loss
    reg = 0.0001 * torch.mean(shape ** 2) + 0.001 * torch.mean(pose ** 2)
    loss += reg

    optimizer.zero_grad()
    loss.backward() # backward propagation
    optimizer.step() # update pose and shape parameters

    # show progress
    if i % 2 == 0:
        print(f"step {i}, loss: {loss.item():.6f}")


faces = new_smpl_model.faces
mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces)
mesh.export('result_smpl.obj') # final output
print("done!")
