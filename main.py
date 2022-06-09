import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

TEST_ARRAY = np.array([
    [[-1, 0, 1], [0, 0, 1], [1, 0, 1]],
    [[-1, 0, 0], [0, 2, 0], [1, 0, 0]],
    [[-1, 0, -1], [0, 0, -1], [1, 0, -1]]
])


def get_triangles_for_element(arr, row, col):
    s = arr.shape[0]
    connected = [(row - 1, col), (row - 1, col + 1), (row, col + 1), (row + 1, col), (row + 1, col - 1), (row, col - 1)]
    poss_triangles = [((row, col), connected[n], connected[(n + 1) % 6]) for n in range(6)]
    t_list = []
    for tri in poss_triangles:
        elems = [0 <= n < s for pt in tri for n in pt]
        if all(elems):
            t_list.append(tri)

    return t_list


def get_vectors_from_triangle(arr, tri_elems):
    triangle = [np.array(arr[i][j]) for i, j in tri_elems]
    v_xy = triangle[1] - triangle[0]
    v_xz = triangle[2] - triangle[0]
    return v_xy, v_xz


def calculate_grad(arr, tri_elems):
    v_xy, v_xz = get_vectors_from_triangle(arr, tri_elems)
    if area_of_triangle(v_xy, v_xz) != 0:
        grad_x_area = (np.dot(v_xy, v_xy) * v_xz - np.dot(v_xy, v_xz) * v_xy) / (2 * area_of_triangle(v_xy, v_xz))
    else:
        grad_x_area = np.array([0, 0, 0])
    return grad_x_area


def area_of_triangle(x, y):
    try:
        area = 0.5*math.sqrt(np.dot(x, x)*np.dot(y, y)-np.dot(x, y)**2)
        return area
    except ValueError:
        return 0


def get_all_triangles(arr):
    s = arr.shape[0]
    all_triangles = []
    for i in range(0, s, 3):
        for j in range(0, s, 3):
            tris = [k for gr in [get_triangles_for_element(arr, i+n, j+n) for n in range(3)] for k in gr]
            all_triangles += tris
    return all_triangles


def get_total_area(arr):
    all_triangles = get_all_triangles(arr)
    return sum(area_of_triangle(*get_vectors_from_triangle(arr, triangle)) for triangle in all_triangles)


def construct_matrix(x_str, y_str, z_str, domain, n=5):
    x = eval("lambda t: " + x_str)
    y = eval("lambda t: " + y_str)
    z = eval("lambda t: " + z_str)
    a = domain[0]
    b = domain[1]
    large_sep = (b - a) / 4
    sep = large_sep / n
    mat = np.zeros((n+1, n+1, 3))
    try:
        assert np.allclose((x(a), y(a), z(a)), (x(b), y(b), z(b)))
    except AssertionError:
        print(f"a: {(x(a), y(a), z(a))}")
        print(f"b: {(x(b), y(b), z(b))}")
        print("Parametric is not a closed loop")
        return

    for k in range(0, n):
        r0_val = a + sep*k
        cn_val = a + large_sep + sep*k
        rn_val = a + 2*large_sep + sep*k
        c0_val = a + 3*large_sep + sep*k
        mat[0][k] = (x(r0_val), y(r0_val), z(r0_val))
        mat[k][n] = (x(cn_val), y(cn_val), z(cn_val))
        mat[n][n-k] = (x(rn_val), y(rn_val), z(rn_val))
        mat[n-k][0] = (x(c0_val), y(c0_val), z(c0_val))
    for i in range(1, n):
        for j in range(1, n):
            mat[i][j] = [np.infty, np.infty, np.infty]

    bound_min = np.min(mat, axis=(0, 1))

    for i in range(1, n):
        for j in range(1, n):
            mat[i][j] = [-np.infty, -np.infty, -np.infty]

    bound_max = np.max(mat, axis=(0, 1))

    for i in range(1, n):
        for j in range(1, n):
            mat[i][j] = (bound_max - bound_min) * np.random.random_sample(3) + bound_min

    return mat


def grad_desc(input_array, iters, alpha):
    for iter_num in range(iters):
        area = get_total_area(input_array)
        print(iter_num, "\t\t", area)
        if area > 10000:
            break
        #  0.125*math.exp(-(1/250)*iter_num)
        input_array = iterate_once(input_array, alpha)
    return input_array


def iterate_once(input_array, alpha):
    grad_matrix = np.zeros(input_array.shape)
    for i in range(1, input_array.shape[0] - 1):
        for j in range(1, input_array.shape[0] - 1):
            triangles = get_triangles_for_element(input_array, i, j)
            grad_elem = [calculate_grad(input_array, tri) for tri in triangles]
            grad_matrix[i][j] = (alpha * sum(grad_elem))
    return np.add(input_array, grad_matrix)


def get_trimesh(arr):
    tris = get_all_triangles(arr)
    tri_mesh_points = []
    for triangle_indices in tris:
        tri_in_mesh = []
        for point in triangle_indices:
            tri_in_matrix = arr[point[0]][point[1]]
            tri_in_mesh.append(tri_in_matrix)
        tri_mesh_points.append(tri_in_mesh)
    return tri_mesh_points


def create_graphic(funcs, dom, num_iterations, detail, alpha):
    arr = construct_matrix(*funcs, dom, n=detail)

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    t_pts = np.linspace(*dom, 1000)
    x = np.vectorize(eval("lambda t: " + funcs[0]))
    y = np.vectorize(eval("lambda t: " + funcs[1]))
    z = np.vectorize(eval("lambda t: " + funcs[2]))
    x_pts = x(t_pts)
    y_pts = y(t_pts)
    z_pts = z(t_pts)
    ax.plot3D(x_pts, y_pts, z_pts, "red")

    arr = grad_desc(arr, num_iterations, alpha)

    tri_mesh = get_trimesh(arr)

    mesh = Poly3DCollection(tri_mesh, alpha=0.5)
    mesh.set_edgecolor("black")
    ax.add_collection3d(mesh)
    plt.show()


f = ("(2+math.cos(4*t))*math.cos(t)", "(2+math.cos(4*t))*math.sin(t)", "math.sin(4*t)")
domain = (0, 2*math.pi)
num_iters = 10000
n = 20
a = 0.02

create_graphic(f, domain, num_iters, n, a)
