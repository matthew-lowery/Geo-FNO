import numpy as np

def compute_div_tri_mesh(f, 
                         mesh_points,  # n_points, 2 (ndim)
                         mesh_element_indices, # n_triangles, 3
                     ): 

    mesh_elements = mesh_points[mesh_element_indices] 
    centroids = np.mean(mesh_elements, axis=1)
    edge_indices = mesh_element_indices[:, [[0, 1], [1, 2], [2, 0]]] ### ntris,3,2
    edges = mesh_points[edge_indices] #### ### ntris, 3, 2, 2
    edge_tangents = edges[:,:,1] - edges[:,:,0] #### AB, BC, CA
    edge_lengths = np.linalg.norm(edge_tangents, axis=-1) ### ntris, 3

    edge_normals = np.stack([-edge_tangents[..., 1], edge_tangents[..., 0]], axis=-1)

    ### this makes them agnostic to the vertex ordering (CW vs. CCW)
    direction_test_edges = -edge_tangents[:, [2, 0, 1]] #### AC, BA, CB
    edge_normals = np.where((edge_normals * direction_test_edges).sum(axis=-1)[...,None] > 0, -edge_normals, edge_normals) ## if this is positive, flip

    unit_edge_normals = edge_normals / np.linalg.norm(edge_normals, axis=-1, keepdims=True)
    scaled_edge_normals = unit_edge_normals * edge_lengths[...,None] ### ntris, 3, 2

    vector_field_edges = f[edge_indices] 
    vector_field_midp_on_edge = np.mean(vector_field_edges, axis=2)
    vf_dot_edge_normal = np.sum(vector_field_midp_on_edge*scaled_edge_normals, axis=-1)
    sum_over_triangle = np.sum(vf_dot_edge_normal, axis=-1) 
    
    ### triangle areas
    x1, y1 = mesh_elements[:, 0, 0], mesh_elements[:, 0, 1]
    x2, y2 = mesh_elements[:, 1, 0], mesh_elements[:, 1, 1]
    x3, y3 = mesh_elements[:, 2, 0], mesh_elements[:, 2, 1]
    triangle_areas = 0.5 * np.abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))

    divergences = sum_over_triangle / triangle_areas ### 
    return divergences, centroids

### np vmap equivalent? 
