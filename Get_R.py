#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get_R.py

Compute a rotation matrix that rotates a given 3D vector onto the negative Z-axis
using Rodrigues' rotation formula.

Steps:
1. Compute the magnitude of the original vector
2. Normalize the vector
3. Compute the rotation axis via cross product
4. Compute the rotation angle via dot product
5. Construct the rotation matrix using Rodrigues' formula
6. Verify the rotation result
"""

import numpy as np
from numpy.linalg import norm


# =============================================================================
# Input Vector
# =============================================================================

# Original vector (example)
v = np.array([1615.0, 666.0, 2487.0], dtype=float)

# Other sample vectors (for reference):
# [1615.0,  666.0, 2487.0]
# [2672.0, -127.0, 2643.0]
# [ 810.0, -183.0,  574.0]
# [ 826.0,  -45.0,  483.0]


# =============================================================================
# Vector Normalization
# =============================================================================

# Magnitude of the vector (used as rotated Z value)
a = norm(v)
print(f"旋转后的a值为: {a}")

# Target unit vector (negative Z-axis)
z_axis = np.array([0.0, 0.0, -1.0], dtype=float)

# Unit vector of the original vector
v_unit = v / a


# =============================================================================
# Rotation Axis and Angle
# =============================================================================

# Rotation axis (cross product between v and target axis)
axis = np.cross(v_unit, z_axis)
axis = axis / norm(axis)

# Rotation angle between v and target axis
angle = np.arccos(np.dot(v_unit, z_axis))


# =============================================================================
# Rodrigues Rotation Formula
# =============================================================================

# Skew-symmetric matrix of rotation axis
K = np.array(
    [
        [0.0,        -axis[2],  axis[1]],
        [axis[2],     0.0,     -axis[0]],
        [-axis[1],    axis[0],  0.0],
    ],
    dtype=float,
)

# Identity matrix
I = np.eye(3)

# Rotation matrix
R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

print("\n旋转矩阵为:")
print(R)


# =============================================================================
# Verification
# =============================================================================

# Apply rotation to original vector
rotated_v = R @ v

print("\n验证旋转结果（应接近[0, 0, a]）:")
print(rotated_v)
