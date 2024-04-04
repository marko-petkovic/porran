"""Functions related to coordinate systems and transformations thereof.
"""

import numpy as np
from numpy import cos, sin


def transform_vectors(
    positions: np.ndarray,
    matrix: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    vector: np.ndarray = np.array([0, 0, 0]),
) -> np.ndarray:
    """Apply a transformation matrix and translation vector to 3D
    vectors (atom positions/direction vectors, etc.).

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n, 3), where n is the number of atoms. Number of
        dimensions can be higher. Last dimension is considered to be
        coordinate values.
    matrix : np.ndarray, optional
        3x3 transformation matrix. Defaults to identity matrix.
    vector : np.ndarray, optional
        3x1 translation vector. Defaults to 0-vector.

    Returns
    -------
    np.ndarray
        transformed vectors
    """
    # do x W' instead of W x'
    return np.add(np.matmul(positions, matrix.transpose()), vector)


def transform_in_steps(
    positions: np.ndarray,
    dst: np.ndarray,
    translate: float = 0,
) -> np.ndarray:
    """Transform a set of points, aligning the z-axis to dst, by first
    rotating around the z-axis, then toward dst and finally moving by
    translate in the direction of dst.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n, 3), where n is the number of atoms.
    dst : np.ndarray
        3D vector to rotate to.
    translate : float, optional
        move positions by this much into the direction of dst, after
        rotation. Defaults to 0.

    Returns
    -------
    np.ndarray
        (n,3) transformed positions
    """
    # first destination in x,y-plane
    first_dst = np.array([dst[0], dst[1], 0])
    # rotation matrix around z-axis (from x direction to first_dst)
    mat1 = rotation_matrix_from_vectors(src=np.array([1, 0, 0]), dst=first_dst)
    # rotation matrix to dst
    mat2 = rotation_matrix_from_vectors(src=np.array([0, 0, 1]), dst=dst)
    vec = translate * dst / np.linalg.norm(dst)
    # apply transformations
    steps = [{"matrix": mat1}, {"matrix": mat2}, {"vector": vec}]
    for step in steps:
        positions = transform_vectors(positions=positions, **step)
    return positions


def rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Find the rotation matrix that aligns src to dst.
    Taken from stackoverflow:
    https://stackoverflow.com/questions/45142959

    Parameters
    ----------
    src : np.ndarray
        A 3d "source" vector
    dst : np.ndarray
        A 3d "destination" vector

    Returns
    -------
    np.ndarray
        A transform matrix (3x3) which when applied to src, aligns it
        with dst.
    """
    # unit lengths
    a = (src / np.linalg.norm(src)).reshape(3)
    b = (dst / np.linalg.norm(dst)).reshape(3)
    v = np.cross(a, b)
    if np.any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / np.power(s, 2))
    return np.eye(3)  # cross of all zeros only occurs on identical directions


def rotation_axis_angle(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the axis to rotate around and the angle to rotate by to
    align src to dst.

    Parameters
    ----------
    src : np.ndarray
        Direction vector to rotate from. Shape (3,)
    dst : np.ndarray
        Direction vector to rotate to. Shape (3,)

    Returns
    -------
    tuple[np.ndarray, float]
        axis to rotate around, angle to rotate by [rad]
    """
    v = np.cross(src, dst)
    a = np.arccos(np.dot(src, dst) / (np.linalg.norm(src) * np.linalg.norm(dst)))
    return v, a


def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Construct a rotation matrix to rotate around an axis by an angle.

    Args:
        axis (np.ndarray): axis to rotate around
        angle (float): angle to rotate by [rad]

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    u = (axis / np.linalg.norm(axis)).reshape(3)  # normalize
    return (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * np.cross(np.eye(3), u)  # cross product matrix
        + (1 - np.cos(angle)) * np.outer(u, u)
    )


def fractional_to_cartesian_matrix(
    a: float, b: float, c: float, al: float, be: float, ga: float
) -> np.ndarray:
    """Construct a matrix to go from fractional to cartesian coordinates
    Transformation matrix taken from Wikipedia:
    https://en.wikipedia.org/wiki/Fractional_coordinates
    NOTE: there care conflicting definitions of the cell dimensions!
    For example, see also: http://ic50.org/fractorth/frorth.pdf

    Args:
        a (float): a
        b (float): b
        c (float): c
        al (float): alpha [rad]
        be (float): beta [rad]
        ga (float): gamma [rad]

    Returns:
        np.ndarray: 3x3 rotation matrix
    """

    # define functions to keep matrix somewhat readable
    def cot(x):
        return cos(x) / sin(x)

    def csc(x):
        return 1 / sin(x)

    # construct matrix
    a11 = (
        a
        * sin(be)
        * np.sqrt(1 - (cot(al) * cot(be) - csc(al) * csc(be) * cos(ga)) ** 2)
    )
    a12 = 0
    a13 = 0

    a21 = a * csc(al) * cos(ga) - a * cot(al) * cos(be)
    a22 = b * sin(al)
    a23 = 0

    a31 = a * cos(be)
    a32 = b * cos(al)
    a33 = c

    return np.array(
        [
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33],
        ]
    )


def cartesian_to_fractional_matrix(
    a: float, b: float, c: float, al: float, be: float, ga: float
) -> np.ndarray:
    """Construct a matrix to go from cartesian to fractional coordinates
    Inverse of fractional to cartesian matrix.

    Args:
        a (float): a
        b (float): b
        c (float): c
        al (float): alpha [rad]
        be (float): beta [rad]
        ga (float): gamma [rad]

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    return np.linalg.inv(
        fractional_to_cartesian_matrix(a=a, b=b, c=c, al=al, be=be, ga=ga)
    )


def get_matrices(
    a: float, b: float, c: float, al: float, be: float, ga: float
) -> tuple[np.ndarray, np.ndarray]:
    """Get matrices to go from fractional to cartesian coordinates and
    vice versa.

    Args:
        a (float): a
        b (float): b
        c (float): c
        al (float): alpha [rad]
        be (float): beta [rad]
        ga (float): gamma [rad]

    Returns:
        tuple[np.ndarray]: (fractional to cartesian matrix, cartesian to
        fractional matrix)
    """
    return (
        fractional_to_cartesian_matrix(a=a, b=b, c=c, al=al, be=be, ga=ga),
        cartesian_to_fractional_matrix(a=a, b=b, c=c, al=al, be=be, ga=ga),
    )
