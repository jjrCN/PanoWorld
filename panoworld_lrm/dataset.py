import os
import json
import random
import traceback
import numpy as np
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import repeat
from scipy.spatial.transform import Rotation as R


def get_local_rotation_matrix(x_angle, y_angle, z_angle):
    """
    Generate a local 4x4 rotation matrix from the given Euler angles in degrees.
    """
    r_matrix = R.from_euler('xyz', [x_angle, y_angle, z_angle], degrees=True).as_matrix()

    local_c2w = np.eye(4)
    local_c2w[:3, :3] = r_matrix
    return local_c2w

def crop_and_resize(target_size, fxfycxcy, square_crop):
    target_width, target_height = target_size
    fx, fy, cx, cy, h, w = fxfycxcy

    # squre crop
    if square_crop:
        min_size = min(w, h)
        start_h = (h - min_size) // 2
        start_w = (w - min_size) // 2
        cx -= start_w
        cy -= start_h


    if square_crop:
        min_size = min(w, h)
        new_fx = fx * (target_width / min_size)
        new_fy = fy * (target_height / min_size)
        new_cx = cx * (target_width / min_size)
        new_cy = cy * (target_height / min_size)
    else:
        new_fx = fx * (target_width / w)
        new_fy = fy * (target_height / h)
        new_cx = cx * (target_width / w)
        new_cy = cy * (target_height / h)

    return [new_fx, new_fy, new_cx, new_cy]

def resize_pano(image, depth, mask, target_size):
    target_width, target_height = target_size

    resized_image = cv2.resize(image, (target_width, target_height))
    if depth is not None:
        resized_depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        resized_depth = resized_depth[:, :, np.newaxis]
    else:
        resized_depth = None

    if mask is not None:
        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask[:, :, np.newaxis]
    else:
        resized_mask = None

    return resized_image, resized_depth, resized_mask

class Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.viewpoint_max_view = config.data.get("viewpoint_max_view", 12)
        self.panoworld_mode = bool(config.data.get("panoworld_mode", False))
        self.pano_image_name = config.data.get("pano_image_name", "panoImage_2048.png")
        self.panoworld_start_image = config.data.get("panoworld_start_image", self.pano_image_name)

        data_path_text = config.data.data_path
        data_root_dir = config.data.get("root_data_dir", config.data.get("root_data_die", ""))

        self.input_view_lists = []
        self.target_view_lists = []
        if self.panoworld_mode:
            with open(data_path_text, 'r') as f:
                self.data_path_scene = f.readlines()
            self.data_path_scene = [x.strip() for x in self.data_path_scene]
            self.data_path_scene = [os.path.join(data_root_dir, x) for x in self.data_path_scene if len(x) > 0]
            total_count_scene = len(self.data_path_scene)
            print(f"Finish load data from str, total: {total_count_scene}\n")
            self.data_path = self.build_panoworld_batches(self.data_path_scene)
            print(f"Finish build panoworld inference batches, total: {len(self.data_path)}\n")

        else:
            with open(data_path_text, 'r') as f:
                self.data_path = f.readlines()
            self.data_path = [x.strip() for x in self.data_path]
            self.data_path = [os.path.join(data_root_dir, x) for x in self.data_path if len(x) > 0]
            total_count = len(self.data_path)
            print(f"Finish load data from str, total: {total_count}\n")

        # if not config.get("inference", False):
        #     np.random.shuffle(self.data_path)

    def pano_stem(self):
        return self.pano_image_name.split(".")[0]

    def lrm_pano_name(self):
        return self.pano_stem() + "_lrm.png"

    def refined_pano_name(self):
        return self.pano_image_name

    def view_dir(self, viewpoints_path, view_name):
        return os.path.join(viewpoints_path, "viewpoints", view_name)

    def join_view_names(self, view_names):
        return "-".join(view_names)

    def load_view_translation(self, viewpoints_path, view_name):
        extrinsics_path = os.path.join(self.view_dir(viewpoints_path, view_name), "extrinsics.txt")
        c2w = np.loadtxt(extrinsics_path)
        return c2w[:3, 3]

    def nearest_completed_view(self, viewpoints_path, target_view, completed_views):
        target_translation = self.load_view_translation(viewpoints_path, target_view)
        best_view = None
        best_dist = None
        for view_name in completed_views:
            view_translation = self.load_view_translation(viewpoints_path, view_name)
            dist = np.linalg.norm(target_translation - view_translation)
            if best_dist is None or dist < best_dist:
                best_view = view_name
                best_dist = dist
        return best_view

    def panoworld_input_image_name(self, viewpoints_path, view_name, initial_view):
        if view_name == initial_view:
            return self.panoworld_start_image
        return self.refined_pano_name()

    def append_panoworld_batch(
        self,
        batches,
        data_path,
        viewpoints_path,
        initial_view,
        input_views,
        input_room_ids,
        target_views,
        is_final=False,
    ):
        if len(input_views) == 0 or len(target_views) == 0:
            return
        if not is_final and len(target_views) != 1:
            raise ValueError("Intermediate PanoWorld batches must have exactly one target view.")

        ref_view = ""
        ref_view_dir = ""
        ref_pano_path = ""
        if not is_final:
            ref_view = self.nearest_completed_view(viewpoints_path, target_views[0], input_views)
            ref_view_dir = self.view_dir(viewpoints_path, ref_view)
            ref_image_name = (
                self.panoworld_start_image
                if ref_view == initial_view
                else self.refined_pano_name()
            )
            ref_pano_path = os.path.join(ref_view_dir, ref_image_name)

        batches.append({
            "data_path": data_path,
            "viewpoints_path": viewpoints_path,
            "input_views": list(input_views),
            "input_room_ids": list(input_room_ids),
            "input_image_names": [
                self.panoworld_input_image_name(viewpoints_path, view_name, initial_view)
                for view_name in input_views
            ],
            "target_views": list(target_views),
            "target_dirs": [
                self.view_dir(viewpoints_path, view_name)
                for view_name in target_views
            ],
            "ref_view": ref_view,
            "ref_view_dir": ref_view_dir,
            "ref_pano_path": ref_pano_path,
            "is_final": is_final,
        })

    def build_panoworld_batches(self, data_paths):
        batches = []
        for data_path in data_paths:
            map_json = json.load(open(data_path, 'r'))
            viewpoints_path = os.path.dirname(data_path)
            groups = []
            for room_id, (map_key, map_values) in enumerate(map_json.items()):
                groups.append((map_key, list(map_values), room_id))
            if len(groups) == 0:
                continue

            initial_view = groups[0][0]
            completed_views = [initial_view]
            completed_room_ids = [groups[0][2]]

            for group_idx, (map_key, map_values, room_id) in enumerate(groups):
                if group_idx > 0:
                    self.append_panoworld_batch(
                        batches,
                        data_path,
                        viewpoints_path,
                        initial_view,
                        completed_views,
                        completed_room_ids,
                        [map_key],
                        is_final=False,
                    )
                    completed_views.append(map_key)
                    completed_room_ids.append(room_id)

                for map_value in map_values:
                    self.append_panoworld_batch(
                        batches,
                        data_path,
                        viewpoints_path,
                        initial_view,
                        completed_views,
                        completed_room_ids,
                        [map_value],
                        is_final=False,
                    )
                    completed_views.append(map_value)
                    completed_room_ids.append(room_id)

            self.append_panoworld_batch(
                batches,
                data_path,
                viewpoints_path,
                initial_view,
                completed_views,
                completed_room_ids,
                completed_views,
                is_final=True,
            )

        return batches


    def __len__(self):
        return len(self.data_path)

    def process_frames(self, frames):
        fxfycxcy_list = []

        resize_h = self.config.data.get("resize_h", -1)
        patch_size = self.config.model.patch_size * self.config.model.get("patch_factor", 2)
        square_crop = self.config.data.square_crop

        resize_w = resize_h
        resize_h = int(round(resize_h / patch_size)) * patch_size #
        resize_w = int(round(resize_w / patch_size)) * patch_size #
        for frame in frames:
            fxfycxcyhw = [frame["fx"], frame["fy"], frame["cx"], frame["cy"], frame["h"], frame["w"]]
            fxfycxcy = crop_and_resize((resize_w, resize_h), fxfycxcyhw, square_crop)
            fxfycxcy_list.append(fxfycxcy)
        intrinsics = torch.tensor(fxfycxcy_list, dtype=torch.float32)  # (num_frames, 4)
        c2ws = np.stack([np.array(frame["c2w"]) for frame in frames])
        c2ws = torch.from_numpy(c2ws).float()
        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        return intrinsics, c2w_bucket

    def process_pano_frames(self, frames):
        image_list = []
        depth_list = []
        mask_list = []
        resize_h_pano = self.config.data.get("resize_h_pano", -1)
        patch_size = self.config.model.patch_size * self.config.model.get("patch_factor", 2)
        filter_top_down = self.config.data.get("filter_top_down", False)

        resize_w_pano = int(resize_h_pano * 2) # 512
        resize_h_pano = int(round(resize_h_pano / patch_size)) * patch_size # 256
        resize_w_pano = int(round(resize_w_pano / patch_size)) * patch_size # 512
        for frame in frames:
            image = np.array(Image.open(frame["image_path"]))
            h, w = image.shape[:2]
            if w/h != 2:
                return None, None, None, None, False

            if "depth_path" in frame and os.path.exists(frame["depth_path"]):
                depth = np.array(Image.open(frame["depth_path"])) # [h, w]
                depth = depth[:, :, np.newaxis] # [h, w, 1]
            else:
                depth = np.zeros((h, w, 1), dtype=image.dtype)

            if "mask_path" in frame and os.path.exists(frame["mask_path"]):
                mask = np.array(Image.open(frame["mask_path"]))/255
                if len(mask.shape) == 2:
                    mask = mask[:, :, np.newaxis]
            else:
                mask = np.ones((h, w, 1), dtype=image.dtype)
                if filter_top_down:
                    mask[:int(h//5), :, :] = 0
                    mask[int(4*h//5):, :, :] = 0

            depth_scale = 1.0
            if "depth_scale" in frame:
                depth_scale = frame["depth_scale"]
            image, depth, mask = resize_pano(image, depth, mask, (resize_w_pano, resize_h_pano))
            depth = depth * 1.0 / depth_scale # Convert back to meters

            image_list.append(torch.from_numpy(image / 255.0).permute(2, 0, 1).float())  # (3, resize_h, resize_w)
            depth_list.append(torch.from_numpy(depth).permute(2, 0, 1).float())  # (1, resize_h, resize_w)
            mask_list.append(torch.from_numpy(mask).permute(2, 0, 1).float()) # (1, resize_h, resize_w)

        images = torch.stack(image_list, dim=0)
        depths = torch.stack(depth_list, dim=0)
        masks = torch.stack(mask_list, dim=0) # (v, 1, resize_h, resize_w)
        c2ws = np.stack([np.array(frame["c2w"]) for frame in frames])
        c2ws = torch.from_numpy(c2ws).float()

        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        return images, depths, masks, c2w_bucket, True

    def __getitem__(self, idx):
        try:
            # Load the perspective-view metadata for the current scene
            data_path = self.data_path[idx]
            panoworld_item = None
            if self.panoworld_mode:
                panoworld_item = data_path
                data_path = panoworld_item["data_path"]
            data_path_class = data_path.split("/")[-1]

            viewpoints_path = None
            view_name_list = []
            room_id_list = []
            input_image_name_list = []
            target_view_name_list = []
            if data_path_class == "map.json" or data_path_class == "map_eval_12.json" or data_path_class == "map_eval.json":
                map_json = json.load(open(data_path, 'r'))
                room_id = 0
                for map_key in map_json.keys():
                    view_name_list.append(map_key)
                    room_id_list.append(room_id)
                    for map_value in map_json[map_key]:
                        view_name_list.append(map_value)
                        room_id_list.append(room_id)
                    room_id += 1
                viewpoints_path = os.path.dirname(data_path)
                input_image_name_list = [self.pano_image_name] * len(view_name_list)
            elif "map_panoworld" in data_path_class and self.panoworld_mode:
                viewpoints_path = panoworld_item["viewpoints_path"]
                view_name_list = panoworld_item["input_views"]
                room_id_list = panoworld_item["input_room_ids"]
                input_image_name_list = panoworld_item["input_image_names"]
                target_view_name_list = panoworld_item["target_views"]
            else:
                print(f"error loading data_path_class: {data_path_class}")
                return self.__getitem__(random.randint(0, len(self) - 1))

            # Load all panorama frames
            frames_pano = []
            for view_name, image_name in zip(view_name_list, input_image_name_list):
                frame_pano = {}
                frame_pano["c2w"] = np.loadtxt(os.path.join(viewpoints_path, "viewpoints", view_name, "extrinsics.txt"))
                frame_pano["image_path"] = os.path.join(viewpoints_path, "viewpoints", view_name, image_name)
                frame_pano["mask_path"] = os.path.join(viewpoints_path, "viewpoints", view_name, "pano_mask.png")
                frame_pano["view_name"] = view_name
                frames_pano.append(frame_pano)

            num_input_frames = len(frames_pano)
            if not self.panoworld_mode and num_input_frames > self.viewpoint_max_view:
                num_input_frames = self.viewpoint_max_view

            # get input frames_pano range
            input_frames_pano_idx = list(range(0, len(frames_pano))) # Panorama views
            if self.panoworld_mode:
                input_frame_idx = input_frames_pano_idx
            else:
                random_indices = np.random.choice(len(input_frames_pano_idx), num_input_frames, replace=False)
                input_frame_idx = [input_frames_pano_idx[i] for i in random_indices]
            input_frame_room_id = [room_id_list[i] for i in input_frame_idx]

            input_frames = [frames_pano[i] for i in input_frame_idx]
            input_source_view_name = self.join_view_names([frame["view_name"] for frame in input_frames])
            input_gt_pano_paths = "||".join([frame["image_path"] for frame in input_frames])
            if self.panoworld_mode:
                render_view_name_list = target_view_name_list
            else:
                render_view_name_list = [frame["view_name"] for frame in input_frames]
            input_frames_view_name = self.join_view_names(render_view_name_list)

            input_images, input_depths, input_masks, input_c2ws, succ_status = self.process_pano_frames(input_frames)
            if succ_status == False:
                print(f"error succ_status: {succ_status}, data_path: {data_path}")
                return self.__getitem__(random.randint(0, len(self) - 1))

            pose_variations = [(0, 0, 0), (0, -270, 0), (0, -180, 0), (0, -90, 0), (90, 0, 0), (-90, 0, 0)]
            # Load all perspective-view frames
            frames = []
            for view_name in render_view_name_list:
                for angles in pose_variations:
                    local_rot_mat = get_local_rotation_matrix(*angles)
                    base_c2w = np.loadtxt(os.path.join(viewpoints_path, "viewpoints", view_name, "extrinsics.txt"))
                    new_c2w = base_c2w @ local_rot_mat
                    frame_data = {}
                    frame_data["c2w"] = new_c2w
                    frame_data["fx"] = self.config.data.resize_h / 2
                    frame_data["fy"] = self.config.data.resize_h / 2
                    frame_data["cx"] = self.config.data.resize_h / 2
                    frame_data["cy"] = self.config.data.resize_h / 2
                    frame_data["h"] = self.config.data.resize_h
                    frame_data["w"] = self.config.data.resize_h
                    frames.append(frame_data)
            num_target_frames = len(frames)

            target_intr, target_c2ws = self.process_frames(frames)

            # Reject scenes with excessively large translations
            if (target_c2ws[:, :3, 3] > 1e3).any():
                print(f"encounter large translation in target poses: {target_c2ws[:, :3, 3].max()}")
                assert False
            if (input_c2ws[:, :3, 3] > 1e3).any():
                print(f"encounter large translation in input poses: {input_c2ws[:, :3, 3].max()}")
                assert False

            # Camera poses must not contain NaNs
            if any(torch.isnan(torch.det(target_c2ws[:, :3, :3]))):
                print(f"encounter nan in target poses: {target_c2ws[:, :3, :3]}")
                assert False
            if any(torch.isnan(torch.det(input_c2ws[:, :3, :3]))):
                print(f"encounter nan in input poses: {input_c2ws[:, :3, :3]}")
                assert False

            # Verify that each rotation matrix has determinant 1
            if not torch.allclose(torch.det(target_c2ws[:, :3, :3]), torch.det(target_c2ws[:, :3, :3]).new_tensor(1.0)):
                print(f"det of target poses not equal to 1")
                assert False
            if not torch.allclose(torch.det(input_c2ws[:, :3, :3]), torch.det(input_c2ws[:, :3, :3]).new_tensor(1.0)):
                print(f"det of input poses not equal to 1")
                assert False

            # normalize input camera poses
            position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
            forward_avg = input_c2ws[:, :3, 2].mean(0) # (3,)
            down_avg = input_c2ws[:, :3, 1].mean(0) # (3,)

            # --- Safeguard 1: check whether forward_avg is too small ---
            if torch.norm(forward_avg) < 1e-6:
                # If camera directions cancel out completely, fall back to the default Z axis
                forward_avg = torch.tensor([0.0, 0.0, 1.0], device=input_c2ws.device).float()
            else:
                forward_avg = F.normalize(forward_avg, dim=0)

            # --- Safeguard 2: check down_avg and apply Gram-Schmidt orthogonalization ---
            # First, try to compute an orthogonalized down vector
            down_avg_ortho = down_avg - down_avg.dot(forward_avg) * forward_avg

            if torch.norm(down_avg_ortho) < 1e-6:
                # This means either:
                # 1. the original down_avg is zero, or
                # 2. the original down_avg is parallel to forward_avg
                # In either case, create a fallback vector that is not parallel to forward_avg.

                # Try the Y axis first
                fallback_down = torch.tensor([0.0, 1.0, 0.0], device=input_c2ws.device).float()
                # If forward is nearly aligned with Y, switch to the X axis instead
                if torch.abs(torch.dot(forward_avg, fallback_down)) > 0.99:
                    fallback_down = torch.tensor([1.0, 0.0, 0.0], device=input_c2ws.device).float()

                # Orthogonalize again using the fallback direction
                down_avg_ortho = fallback_down - fallback_down.dot(forward_avg) * forward_avg
                down_avg = F.normalize(down_avg_ortho, dim=0)
            else:
                # Standard normalization path
                down_avg = F.normalize(down_avg_ortho, dim=0)

            # Compute the right vector; the safeguards above ensure this cross product is safe
            right_avg = torch.cross(down_avg, forward_avg, dim=0)

            # Build the normalization transform
            pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
            pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)

            # Invert the transform; the matrix should be orthogonal here, so inversion is stable
            pos_avg_inv = torch.inverse(pos_avg)

            input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), input_c2ws)
            target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), target_c2ws)

            if torch.isnan(input_c2ws).any() or torch.isinf(input_c2ws).any():
                print("encounter nan or inf in input poses")
                assert False

            if torch.isnan(target_c2ws).any() or torch.isinf(target_c2ws).any():
                print("encounter nan or inf in target poses")
                assert False

            input_depths_mask = (input_depths > 0) * (input_masks > 0)
            input_room_ids = torch.tensor(input_frame_room_id).long()

            ret_dict = {
                "input_images": input_images,  # (num_input, 3, resize_pano_h, resize_pano_w)
                "input_depths": input_depths, # (num_input, 1, resize_pano_h, resize_pano_w)
                "input_depths_mask": input_depths_mask, # (num_input, 1, resize_h, resize_w)
                "input_masks": (input_masks > 0), # (num_input, 1, resize_pano_h, resize_pano_w)
                "input_c2ws": input_c2ws,  # (num_input, 4, 4)
                "target_fxfycxcy": target_intr,  # (num_target, 4)
                "target_c2ws": target_c2ws, # (num_target, 4, 4)
                "input_room_ids": input_room_ids,
                "input_target_scene_name": viewpoints_path.split("/")[-1],
                "input_view_names": input_frames_view_name,
                "input_source_view_names": input_source_view_name,
                "input_gt_pano_paths": input_gt_pano_paths,
            }
            if self.panoworld_mode:
                ret_dict.update({
                    "panoworld_target_view_names": "||".join(panoworld_item["target_views"]),
                    "panoworld_target_dir_names": "||".join(panoworld_item["target_dirs"]),
                    "panoworld_target_dir_name": panoworld_item["target_dirs"][0] if len(panoworld_item["target_dirs"]) == 1 else "",
                    "panoworld_ref_view": panoworld_item["ref_view"],
                    "panoworld_ref_view_dir_name": panoworld_item["ref_view_dir"],
                    "panoworld_ref_pano_path": panoworld_item["ref_pano_path"],
                    "panoworld_is_final": panoworld_item["is_final"],
                })

        except:
            traceback.print_exc()
            print(f"error loading data: {self.data_path[idx]}")
            if self.panoworld_mode:
                raise
            return self.__getitem__(random.randint(0, len(self) - 1))

        return ret_dict
