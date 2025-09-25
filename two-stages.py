import polyscope as ps
import trimesh
from trimesh import Trimesh
import numpy as np
from py3dbp import Packer, Bin, Item
from trimesh.primitives import Box
import os

def load_furniture(name = "stefan", combined = [], y_height = 0.5):
    folder = f"data/{name}"
    parts = []
    files = os.listdir(folder)
    obj_files = [f"{folder}/{file}" for file in files if file.endswith('.obj')]
    obj_files.sort()
    for filename in obj_files:
        mesh = trimesh.load(filename,  force='mesh')
        parts.append(mesh)

    full_asssembly = trimesh.util.concatenate(parts)
    full_assembly_height = full_asssembly.bounds[1][1] - full_asssembly.bounds[0][1]
    scale = y_height / full_assembly_height
    parts = [part.apply_scale(scale) for part in parts]
    full_assembly = full_asssembly.apply_scale(scale)

    ps.register_surface_mesh("full_assembly", full_asssembly.vertices, full_asssembly.faces, enabled=False)

    furniture = []
    if combined == []:
        furniture = parts
    else:
        for part_ids in combined:
            part_list = [parts[part_id] for part_id in part_ids]
            furniture.append(trimesh.util.concatenate(part_list))
    return furniture

def load_boxes(boxes_names):
    boxes = []
    for name in boxes_names:
        if name == "M1":
            boxes.append(trimesh.primitives.Box(extents=[370, 250, 210]).apply_scale(1E-3))
        elif name == "M2":
            boxes.append(trimesh.primitives.Box(extents=[430, 350, 260]).apply_scale(1E-3))
        elif name == "M4":
            boxes.append(trimesh.primitives.Box(extents=[730, 550, 460]).apply_scale(1E-3))
    return boxes

def get_transformation(boxA, boxB):
    boxA_extents = boxA.extents
    boxB_extents = boxB.extents

    centroidA = np.mean(boxA.bounds, axis=0)
    centroidB = np.mean(boxB.bounds, axis=0)

    R = np.zeros((3, 3))
    extents = boxB_extents.copy()
    for id in range(3):
        dist = np.abs(extents - boxA_extents[id])
        min_ind = np.argmin(dist)
        extents[min_ind] = 0.0
        R[min_ind, id] = 1.0

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centroidB - R @ centroidA
    return T

packer = Packer()
#packer.add_bin(Bin('M4', 0.73, 0.55, 0.46, 100))
packer.add_bin(Bin('min van', 1.8, 1.2, 1.2, 100))
#packer.add_bin(Bin('5t van', 4.5, 1.8, 1.8, 100))

ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_ground_plane_mode("none")


# furniture_parts = load_furniture("bekvam", combined=[[0, 1, 2, 3, 4, 5, 6, 7]], y_height=0.5) * 1
# furniture_parts = load_furniture("bekvam", combined=[], y_height=0.5) * 9
# furniture_parts = load_furniture("bekvam", combined=[], y_height=0.5) * 9

furniture_parts = load_boxes(["M4"] * 5)
furniture_parts += load_boxes(["M2"] * 10)

#furniture_parts = load_furniture("sjalland", combined=[[2, 5 ,9], [3, 8, 4], [0], [1], [6], [7], [10]], y_height=0.73)
furniture_parts += load_furniture("lantliv", combined=[], y_height=0.68) * 2
#furniture_parts = load_furniture("lantliv", combined=[[3], [0, 1, 2, 4, 5, 6, 7]]) * 2


furniture_part_bboxs = []
for id, obj in enumerate(furniture_parts):

    bbox = obj.bounding_box_oriented.copy()
    #ps.register_surface_mesh(f"part {id}", obj.vertices, obj.faces)
    #ps.register_surface_mesh(f"bbox {id}", bbox.vertices, bbox.faces)

    T = bbox.transform
    T = np.linalg.inv(T)

    bbox = bbox.apply_transform(T)
    furniture_parts[id] = furniture_parts[id].apply_transform(T)

    size = bbox.extents.copy()
    furniture_part_bboxs.append(bbox.copy())

    #size = np.sort(size)[::-1]
    packer.add_item(Item(f"{id}", size[0], size[1], size[2], 1))

packer.pack(bigger_first = True)

dimension = np.array([packer.bins[0].width, packer.bins[0].height, packer.bins[0].depth], dtype=np.float64)
box = Box(extents=dimension)
box = box.apply_translation(dimension / 2.0)
container = ps.register_surface_mesh("truck", box.vertices, box.faces)
container.set_transparency(0.3)

for b in packer.bins:
    print(":::::::::::", b.string())
    for id, item in enumerate(b.items):
        dimension = np.array(item.get_dimension(), dtype=np.float64)
        position =  np.array(item.position, dtype=np.float64)
        box = Box(extents=dimension)
        part_id = int(item.name)
        box = box.apply_translation(position + dimension / 2.0)
        #ps.register_surface_mesh(f"bin pack box {id}", box.vertices, box.faces)

        T = get_transformation(furniture_part_bboxs[part_id], box)
        part = Trimesh(furniture_parts[part_id].vertices, furniture_parts[part_id].faces)
        part = part.apply_transform(T)
        ps.register_surface_mesh(f"pack part {part_id}", part.vertices, part.faces)
    print(f"{len(packer.bins[0].items)}/{len(packer.items)}")

ps.show()