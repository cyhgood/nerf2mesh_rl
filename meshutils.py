import numpy as np
import pymeshlab as pml

def isotropic_explicit_remeshing(verts, faces):

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    # ms.apply_coord_taubin_smoothing()
    ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] isotropic explicit remesh: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces
    

def decimate_mesh(verts, faces, target, backend='pymeshlab', remesh=False, optimalplacement=True):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == 'pyfqmr':
        import pyfqmr
        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:

        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, 'mesh') # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.Percentage(1))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), optimalplacement=optimalplacement)

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(f'[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_masked_trigs(verts, faces, mask, dilation=5):
    # mask: 0 == keep, 1 == remove

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, f_scalar_array=mask) # mask as the quality
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # select faces
    ms.compute_selection_by_condition_per_face(condselect='fq == 0') # select kept faces
    # dilate to aviod holes...
    for _ in range(dilation):
        ms.apply_selection_dilatation()
    ms.apply_selection_inverse(invfaces=True) # invert

    # delete faces
    ms.meshing_remove_selected_faces()

    # clean unref verts
    ms.meshing_remove_unreferenced_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh mask trigs: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces    


def remove_masked_verts(verts, faces, mask):
    # mask: 0 == keep, 1 == remove

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, v_scalar_array=mask) # mask as the quality
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # select verts
    ms.compute_selection_by_condition_per_vertex(condselect='q == 1')

    # delete verts and connected faces
    ms.meshing_remove_selected_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh mask verts: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def remove_selected_verts(verts, faces, query='(x < 1) && (x > -1) && (y < 1) && (y > -1) && (z < 1 ) && (z > -1)'):

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # select verts
    ms.compute_selection_by_condition_per_vertex(condselect=query)

    # delete verts and connected faces
    ms.meshing_remove_selected_vertices()

    # extract
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh remove verts: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces

def clean_mesh(verts, faces, v_pct=1, min_f=8, min_d=5, repair=True, remesh=True):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices() # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(threshold=pml.Percentage(v_pct)) # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces() # faces defined by the same verts
    ms.meshing_remove_null_faces() # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pml.Percentage(min_d))
    
    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.Percentage(1))

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


def subdivide_longest_edges(verts, faces, candidates, return_event=False,
                            candidate_order=None, max_added_faces=None):
    """Split requested longest edges and return each output face's input index.

    Conflicting requests are processed in face index order. An edge is skipped
    when any incident face has already been split in this round.
    """
    candidates = np.asarray(candidates, dtype=bool)
    if candidates.shape != (len(faces),):
        raise ValueError('candidates must contain one boolean per face')

    edge_faces = {}
    for face_id, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_faces.setdefault(edge, []).append(face_id)

    split_faces = {}
    new_vertices = list(verts)
    requested_edges = set()
    edge_results = {}
    executed_edges = []
    edge_events = []
    request_status = {}
    if max_added_faces is not None and max_added_faces < 0:
        raise ValueError('max_added_faces must be nonnegative')
    if candidate_order is None:
        candidate_order = np.flatnonzero(candidates)
    else:
        candidate_order = np.asarray(candidate_order, dtype=np.int64)
        if (candidate_order.ndim != 1 or len(np.unique(candidate_order)) != len(candidate_order)
                or np.any(candidate_order < 0) or np.any(candidate_order >= len(faces))
                or not np.all(candidates[candidate_order])):
            raise ValueError('candidate_order must contain unique candidate face indices')
    processed = []
    added_faces = 0
    additions = []
    considered_count = 0
    for face_id in candidate_order:
        if max_added_faces is not None and added_faces >= max_added_faces:
            processed.append(int(face_id))
            request_status[int(face_id)] = 'budget'
            continue
        processed.append(int(face_id))
        considered_count += 1
        face = faces[face_id]
        edges = [(int(face[i]), int(face[(i + 1) % 3])) for i in range(3)]
        lengths = [np.sum((verts[a] - verts[b]) ** 2) for a, b in edges]
        edge = tuple(sorted(edges[int(np.argmax(lengths))]))
        if edge in requested_edges:
            request_status[int(face_id)] = edge_results[edge]
            continue
        requested_edges.add(edge)
        incident_faces = edge_faces[edge]
        if any(index in split_faces for index in incident_faces):
            edge_results[edge] = 'conflict'
            request_status[int(face_id)] = 'conflict'
            continue
        if max_added_faces is not None and added_faces + len(incident_faces) > max_added_faces:
            edge_results[edge] = 'budget'
            request_status[int(face_id)] = 'budget'
            continue
        edge_results[edge] = 'executed'
        request_status[int(face_id)] = 'executed'
        executed_edges.append(list(edge))
        edge_events.append({'edge': list(edge), 'requester_face': int(face_id),
                            'affected_faces': list(map(int, incident_faces)),
                            'added_faces': len(incident_faces)})
        added_faces += len(incident_faces)
        additions.append(added_faces)
        midpoint_id = len(new_vertices)
        new_vertices.append((verts[edge[0]] + verts[edge[1]]) * 0.5)
        for index in incident_faces:
            split_faces[index] = (edge, midpoint_id)

    new_faces = []
    parent_face_id = []
    for face_id, face in enumerate(faces):
        if face_id not in split_faces:
            new_faces.append(face)
            parent_face_id.append(face_id)
            continue
        edge, midpoint_id = split_faces[face_id]
        for i in range(3):
            a, b, opposite = int(face[i]), int(face[(i + 1) % 3]), int(face[(i + 2) % 3])
            if tuple(sorted((a, b))) == edge:
                new_faces.extend(((a, midpoint_id, opposite), (midpoint_id, b, opposite)))
                parent_face_id.extend((face_id, face_id))
                break

    result = (np.asarray(new_vertices, dtype=verts.dtype).reshape(-1, 3),
              np.asarray(new_faces, dtype=faces.dtype).reshape(-1, 3),
              np.asarray(parent_face_id, dtype=np.int64))
    if return_event:
        successful = {face for face, status in request_status.items() if status == 'executed'}
        passive = sorted(set(split_faces) - successful)
        return result + ({'requested_faces': processed,
                          'request_status': request_status,
                          'executed_edges': executed_edges,
                          'edge_events': edge_events,
                          'passive_faces': passive, 'added_faces': added_faces,
                          'considered_count': considered_count,
                          'cumulative_added_faces': additions},)
    return result


def decimate_and_refine_mesh(verts, faces, mask, decimate_ratio=0.1, refine_size=0.01, refine_remesh_size=0.02):
    # verts: [N, 3]
    # faces: [M, 3]
    # mask: [M], 0 denotes do nothing, 1 denotes decimation, 2 denotes subdivision

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces, f_scalar_array=mask)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!

    # decimate and remesh
    ms.compute_selection_by_condition_per_face(condselect='fq == 1')
    if decimate_ratio > 0:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int((1 - decimate_ratio) * (mask == 1).sum()), selected=True)

    if refine_remesh_size > 0:
        ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(refine_remesh_size), selectedonly=True)

    # repair
    ms.set_selection_none(allfaces=True)
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    
    # refine 
    if refine_size > 0:
        ms.compute_selection_by_condition_per_face(condselect='fq == 2')
        ms.meshing_surface_subdivision_midpoint(threshold=pml.AbsoluteValue(refine_size), selected=True)

        # ms.meshing_isotropic_explicit_remeshing(iterations=3, targetlen=pml.AbsoluteValue(refine_size), selectedonly=True)

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(f'[INFO] mesh decimating & subdividing: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}')

    return verts, faces


# in meshutils.py
def select_bad_and_flat_faces_by_normal(verts, faces, usear=False, aratio=0.02, nfratio_bad=120, nfratio_flat=5):
    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh')

    ms.compute_selection_bad_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_bad)
    m = ms.current_mesh()
    bad_faces_mask = m.face_selection_array()
    # print('bad_faces_mask cnt: ', sum(bad_faces_mask * 1.0))
    ms.set_selection_none(allfaces=True)

    ms.compute_selection_bad_faces(usear=usear, aratio=aratio, usenf=True, nfratio=nfratio_flat)
    m = ms.current_mesh()
    flat_faces_mask = m.face_selection_array() == False  # reverse
    # print('flat_faces_mask cnt: ', sum(flat_faces_mask * 1.0))
    ms.set_selection_none(allfaces=True)

    return bad_faces_mask, flat_faces_mask
