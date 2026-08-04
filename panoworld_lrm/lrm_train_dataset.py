import os
import json
import random
from collections import defaultdict
import traceback
import numpy as np
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from einops import repeat


def resolve_manifest_path(manifest_path):
    manifest_path = os.path.expanduser(str(manifest_path))
    return manifest_path if os.path.isabs(manifest_path) else os.path.abspath(manifest_path)


def resolve_data_entry(root_data_dir, entry):
    entry = os.path.expanduser(str(entry).strip())
    if not entry:
        return ""
    if os.path.isabs(entry):
        return entry
    if not root_data_dir:
        return entry
    return os.path.join(os.path.expanduser(str(root_data_dir)), entry)


def is_config_list(value):
    return isinstance(value, (list, tuple))


def normalize_root_dirs(root_data_dir, data_path_text):
    if is_config_list(data_path_text):
        if is_config_list(root_data_dir):
            if len(root_data_dir) != len(data_path_text):
                raise ValueError(
                    "data.root_data_dir and data.data_path must have the same length "
                    f"when both are lists, got {len(root_data_dir)} and {len(data_path_text)}."
                )
            return list(root_data_dir)
        return [root_data_dir] * len(data_path_text)

    if is_config_list(root_data_dir):
        if len(root_data_dir) != 1:
            raise ValueError(
                "data.root_data_dir is a list but data.data_path is a single manifest. "
                "Use a single root_data_dir string or a one-element list."
            )
        return root_data_dir[0]

    return root_data_dir


def pick_and_shuffle_indices(nums, max_N):
    if not nums:
        return []

    # 1. Group indices with the same id.
    # Dictionary format: {id: [index1, index2, ...]}.
    index_groups = defaultdict(list)
    for i, num in enumerate(nums):
        index_groups[num].append(i)

    first_pick_indices = []
    remaining_indices = []

    # 2. Randomly pick one index per id and put the rest into the remaining pool.
    for num, indices in index_groups.items():
        # Randomly pick one index.
        chosen_index = random.choice(indices)
        first_pick_indices.append(chosen_index)

        # Add unselected indices to the candidate pool.
        for idx in indices:
            if idx != chosen_index:
                remaining_indices.append(idx)

    # 3. Randomly pick N indices from the remaining pool.
    if max_N > len(first_pick_indices):
        N = max_N - len(first_pick_indices)
        # For robustness, if N exceeds the remaining pool size, use all remaining indices.
        if N > len(remaining_indices):
            N = len(remaining_indices)
    else:
        first_pick_indices_sample = random.sample(first_pick_indices, max_N)
        random.shuffle(first_pick_indices_sample)
        return first_pick_indices_sample

    actual_n = min(N, len(remaining_indices))
    second_pick_indices = random.sample(remaining_indices, actual_n)

    # 4. Concatenate and shuffle.
    final_result = first_pick_indices + second_pick_indices
    random.shuffle(final_result)

    return final_result

def pick_and_shuffle_indices_forward(nums, max_N):
    if not nums or max_N <= 0:
        return []

    # 1. Group indices with the same id.
    # Dictionary format: {id: [index1, index2, ...]}.
    index_groups = defaultdict(list)
    for i, num in enumerate(nums):
        index_groups[num].append(i)

    first_pick_indices = []

    # 2. Shuffle unique ids to randomly choose id groups.
    unique_nums = list(index_groups.keys())
    random.shuffle(unique_nums)

    # 3. Draw indices from the randomly ordered id groups.
    for num in unique_nums:
        # Get all indices for the current id.
        indices = index_groups[num]

        # Compute how many more indices are needed to reach max_N.
        needed = max_N - len(first_pick_indices)

        if len(indices) >= needed:
            # If the current id group has enough indices for the remaining quota,
            # randomly sample the needed amount and stop.
            picked_indices = random.sample(indices, needed)
            first_pick_indices.extend(picked_indices)
            break
        else:
            # Otherwise, shuffle and add all indices from this id group, then continue.
            picked_indices = list(indices)
            random.shuffle(picked_indices)
            first_pick_indices.extend(picked_indices)

    # 4. Optionally shuffle the returned max_N indices.
    # Without this, indices would remain grouped by the sampled id order.
    random.shuffle(first_pick_indices)

    return first_pick_indices

def crop_and_resize(image, depth, target_size, fxfycxcy, square_crop):
    target_width, target_height = target_size
    fx, fy, cx, cy, h, w = fxfycxcy

    # squre crop
    croped_depth = None
    if square_crop:
        min_size = min(w, h)
        start_h = (h - min_size) // 2
        start_w = (w - min_size) // 2
        croped_image = image[start_h:start_h+min_size, start_w:start_w+min_size, :]
        if depth is not None:
            croped_depth = depth[start_h:start_h+min_size, start_w:start_w+min_size, :]
        cx -= start_w
        cy -= start_h
    else:
        croped_image = image
        if depth is not None:
            croped_depth = depth

    resized_image = cv2.resize(croped_image, (target_width, target_height))
    if croped_depth is not None:
        resized_depth = cv2.resize(croped_depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        resized_depth = resized_depth[:, :, np.newaxis]
    else:
        resized_depth = None

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

    return resized_image, resized_depth, [new_fx, new_fy, new_cx, new_cy]

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
        self.evaluation = config.get("evaluation", False)
        self.absoluate_max_num_input_frames = config.data.get("absoluate_max_num_input_frames", 12)
        self.view_select_dict = config.data.get("view_select_dict", {})
        self.sample_target_images = config.data.get("sample_target_images", 6)

        if self.evaluation and "data_eval" in config:
            self.config.data.update(config.data_eval)

        data_path_text = config.data.data_path
        data_root_dir = config.data.get("root_data_dir", "")
        data_root_dir = normalize_root_dirs(data_root_dir, data_path_text)
        if isinstance(data_path_text, list):
            print("Load data from list\n")
            self.data_path = []
            for data_path_text_single, data_root_dir_single in zip(data_path_text, data_root_dir):
                data_path_text_single = str(data_path_text_single)
                manifest_path = resolve_manifest_path(data_path_text_single)
                data_count = 0
                multi_sample = 1
                if "data_front3d" in data_path_text_single:
                    multi_sample = 200
                elif "data_realsee3d" in data_path_text_single:
                    multi_sample = 50
                print(f"load data from: {manifest_path}, root_data_dir: {data_root_dir_single}, multi_sample: {multi_sample}")
                data_path_list = []
                with open(manifest_path, 'r') as f:
                    for x in f:
                        data_path_single_x = resolve_data_entry(data_root_dir_single, x)
                        if not data_path_single_x: # Skip empty lines.
                            continue
                        data_path_list.append(data_path_single_x)
                        data_count += 1
                data_count_resample = 0
                for i in range(multi_sample):
                     self.data_path.extend(data_path_list)
                     data_count_resample += data_count
                print(f"Finish load data from: {manifest_path}, total: {data_count_resample}\n")
            total_count = len(self.data_path)
            print(f"Finish load data from list, total: {total_count}\n")
        elif isinstance(data_path_text, str):
            print("Load data from str\n")
            manifest_path = resolve_manifest_path(data_path_text)
            with open(manifest_path, 'r') as f:
                self.data_path = f.readlines()
            self.data_path = [resolve_data_entry(data_root_dir, x) for x in self.data_path]
            self.data_path = [x for x in self.data_path if len(x) > 0]
            total_count = len(self.data_path)
            print(f"Finish load data from: {manifest_path}, root_data_dir: {data_root_dir}, total: {total_count}\n")
        else:
            print("data_path class error.\n")
            raise NotImplementedError

        if not config.get("inference", False):
            np.random.shuffle(self.data_path)


    def __len__(self):
        return len(self.data_path)

    def process_frames(self, frames):
        fxfycxcy_list = []
        image_list = []
        depth_list = []

        resize_h = self.config.data.get("resize_h", -1)
        patch_size = self.config.model.patch_size * self.config.model.get("patch_factor", 2)
        square_crop = self.config.data.square_crop

        resize_w = resize_h
        resize_h = int(round(resize_h / patch_size)) * patch_size #
        resize_w = int(round(resize_w / patch_size)) * patch_size #
        for frame in frames:
            image = np.array(Image.open(os.path.join(frame["image_base_dir"], frame["file_path"])))
            if "depth_file_path" in frame and os.path.exists(os.path.join(frame["image_base_dir"], frame["depth_file_path"])) and frame["depth_file_path"] != "":
                # Load the perspective depth map in z-depth convention.
                depth = np.array(Image.open(os.path.join(frame["image_base_dir"], frame["depth_file_path"]))) # [h, w]
                depth = depth[:, :, np.newaxis] # [h, w, 1]
            else:
                h, w = image.shape[:2]
                depth = np.zeros((h, w, 1), dtype=image.dtype)
            depth_scale = frame["depth_scale"]
            fxfycxcyhw = [frame["fx"], frame["fy"], frame["cx"], frame["cy"], frame["h"], frame["w"]]
            image, depth, fxfycxcy = crop_and_resize(image, depth, (resize_w, resize_h), fxfycxcyhw, square_crop)
            depth = depth * 1.0 / depth_scale # Convert depth values back to meters.
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(torch.from_numpy(image / 255.0).permute(2, 0, 1).float())  # (3, resize_h, resize_w)
            depth_list.append(torch.from_numpy(depth).permute(2, 0, 1).float())  # (1, resize_h, resize_w)
        intrinsics = torch.tensor(fxfycxcy_list, dtype=torch.float32)  # (num_frames, 4)
        images = torch.stack(image_list, dim=0)
        depths = torch.stack(depth_list, dim=0)
        c2ws = np.stack([np.array(frame["c2w"]) for frame in frames])
        c2ws = torch.from_numpy(c2ws).float()
        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        return images, depths, intrinsics, c2w_bucket

    def process_pano_frames(self, frames):
        image_list = []
        depth_list = []
        mask_list = []
        resize_h_pano = self.config.data.get("resize_h_pano", -1)
        patch_size = self.config.model.patch_size * self.config.model.get("patch_factor", 2)

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

            depth_scale = frame["depth_scale"]
            image, depth, mask = resize_pano(image, depth, mask, (resize_w_pano, resize_h_pano))
            depth = depth * 1.0 / depth_scale # Convert depth values back to meters.

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
            # read config
            input_frame_select_type = self.config.data.input_frame_select_type
            target_frame_select_type = self.config.data.target_frame_select_type
            num_input_frames = self.config.data.num_input_frames # Panorama images.
            num_target_frames = self.config.data.get("num_target_frames", 0) # Perspective images.

            # Load the first-view perspective JSON.
            data_path = self.data_path[idx] # Path to the first-view perspective transforms.json.
            data_path_class = data_path.split("/")[-1]

            if "Realsee3D" in data_path:
                dataset_class = "Realsee3D"
            else:
                dataset_class = "front3d"

            viewpoints_path = None
            scene_name_list = []
            target_frames_view_name = "None"
            target_render_save_path = []
            room_id_list = []
            if data_path_class == "map.json" or data_path_class == "map_eval_12.json" or data_path_class == "map_eval.json":
                map_json = json.load(open(data_path, 'r'))
                room_id = 0
                for map_key in map_json.keys():
                    scene_name_list.append(map_key)
                    room_id_list.append(room_id)
                    for map_value in map_json[map_key]:
                        scene_name_list.append(map_value)
                        room_id_list.append(room_id)
                    room_id += 1
                viewpoints_path = os.path.dirname(data_path)
            elif data_path_class == "transforms.json":
                dir_path = os.path.dirname(data_path) # First-view directory path.
                true_scene_name = data_path.split("/")[-4] # Scene name.
                scene_name_ref = data_path.split("/")[-2] # View name.
                # Get the list of other viewpoints.
                viewpoints_path = data_path.split("/viewpoints/")[0]
                map_path = os.path.join(viewpoints_path, "map.json")
                map_json = json.load(open(map_path, 'r'))
                scene_name_list_all = []
                for key_tmp in map_json.keys():
                    scene_name_list_all.append(key_tmp)
                    scene_name_list_all.extend(map_json[key_tmp])
                scene_name_list = random.sample(scene_name_list_all, 1)
                room_id_list.append(0)
            else:
                print(f"error loading data_path_class: {data_path_class}")
                return self.__getitem__(random.randint(0, len(self) - 1))

            # Load all perspective frames.
            frames = []
            for scene_name in scene_name_list:
                data_json_path = os.path.join(viewpoints_path, "viewpoints", scene_name, "transforms.json")
                data_json = json.load(open(data_json_path, 'r'))
                frames_perceptive_all = []
                for i in range(len(data_json['frames'])):
                    frame_data = data_json['frames'][i]
                    frame_data["fx"] = data_json["fl_x"]
                    frame_data["fy"] = data_json["fl_y"]
                    frame_data["cx"] = data_json["cx"]
                    frame_data["cy"] = data_json["cy"]
                    frame_data["h"] = data_json["h"]
                    frame_data["w"] = data_json["w"]
                    frame_data["c2w"] = np.array(frame_data["transform_matrix"])
                    if "depth_scale" in data_json:
                        frame_data["depth_scale"] = data_json["depth_scale"]
                    else:
                        frame_data["depth_scale"] = 1.0
                    frame_data["image_base_dir"] = os.path.join(viewpoints_path, "viewpoints", scene_name)
                    frames_perceptive_all.append(frame_data)

                # Sample a fixed number of perspective views for each panorama.
                idx_sample = np.random.choice(len(frames_perceptive_all), self.sample_target_images, replace=False)
                idx_sample = np.sort(idx_sample)
                frames_perceptive_sample = [frames_perceptive_all[i] for i in idx_sample]
                frames.extend(frames_perceptive_sample)

            # Load all panorama frames.
            frames_pano = []
            for scene_name in scene_name_list:
                frame_pano = {}
                frame_pano["c2w"] = np.loadtxt(os.path.join(viewpoints_path, "viewpoints", scene_name, "extrinsics.txt"))

                depth_scale_path = os.path.join(viewpoints_path, "viewpoints", scene_name, "depth_scale.txt")

                if os.path.exists(depth_scale_path):
                    frame_pano["depth_scale"] = np.loadtxt(depth_scale_path)
                else:
                    frame_pano["depth_scale"] = 1.0

                if dataset_class == "Realsee3D":
                    frame_pano["image_path"] = os.path.join(viewpoints_path, "viewpoints", scene_name, "panoImage_1600.jpg")
                else:
                    frame_pano["image_path"] = os.path.join(viewpoints_path, "viewpoints", scene_name, "panoImage_2048.png")

                frame_pano["depth_path"] = os.path.join(viewpoints_path, "viewpoints", scene_name, "depth_image.png") # Euclidean-distance depth; panoramas only provide this depth type.

                frame_pano["mask_path"] = os.path.join(viewpoints_path, "viewpoints", scene_name, "pano_mask.png")
                frame_pano["view_name"] = scene_name
                frames_pano.append(frame_pano)

            if num_input_frames == -1:
                max_num = len(frames_pano)
                if max_num > self.absoluate_max_num_input_frames:
                    max_num = self.absoluate_max_num_input_frames
                num_input_frames = np.random.randint(1, max_num+1)
            elif num_input_frames == 0:
                num_input_frames = len(frames_pano)
                if num_input_frames > self.absoluate_max_num_input_frames:
                    num_input_frames = self.absoluate_max_num_input_frames

            # get input frames_pano range
            input_frames_pano_idx = list(range(0, len(frames_pano))) # Panorama images.
            # get input frames
            if input_frame_select_type == 'random':
                # 1. Randomly generate non-repeated indices from the source list.
                random_indices = np.random.choice(len(input_frames_pano_idx), num_input_frames, replace=False)
                # 2. Gather values from the source list using the sampled indices.
                input_frame_idx = [input_frames_pano_idx[i] for i in random_indices]
                input_frame_room_id = [room_id_list[i] for i in random_indices]
            elif input_frame_select_type == 'first':
                input_frame_idx = [input_frames_pano_idx[i] for i in range(0, num_input_frames)]
                input_frame_room_id = [room_id_list[i] for i in range(0, num_input_frames)]
            elif input_frame_select_type == 'random_whole_room':
                random_rate = np.random.rand()
                if random_rate < self.config.data.get("whole_room_rate", 0.0):
                    random_indices = pick_and_shuffle_indices(room_id_list, self.absoluate_max_num_input_frames)
                    num_input_frames = len(random_indices)
                elif random_rate >= self.config.data.get("whole_room_rate", 0.0) and random_rate < 0.75:
                    random_indices = pick_and_shuffle_indices_forward(room_id_list, self.absoluate_max_num_input_frames)
                    num_input_frames = len(random_indices)
                else:
                    random_indices = np.random.choice(len(input_frames_pano_idx), num_input_frames, replace=False)
                input_frame_idx = [input_frames_pano_idx[i] for i in random_indices]
                input_frame_room_id = [room_id_list[i] for i in random_indices]
            else:
                raise NotImplementedError

            # get target frames range
            target_frames_idx = list(range(0, len(frames))) # Perspective images.
            target_frame_idx = []
            if num_target_frames == 0:
                num_target_frames = num_input_frames * 6 # Use the perspective views corresponding to the input panoramas.
                for input_idx in input_frame_idx:
                    for i in range(self.sample_target_images):
                        target_frame_idx.append(self.sample_target_images * input_idx + i)
            elif num_target_frames == -1:
                num_target_frames = np.random.randint(1, self.sample_target_images*len(frames_pano)+1) # Randomly sample the number of target perspective views from the maximum pool.

            # get target frames
            if len(target_frame_idx) == 0:
                if target_frame_select_type == 'random':
                    target_frame_idx = np.random.choice(target_frames_idx, num_target_frames, replace=False)
                elif target_frame_select_type == 'uniform':
                    target_frame_idx = np.linspace(0, len(target_frames_idx) - 1, num_target_frames, dtype=int)
                else:
                    raise NotImplementedError
            # target_frame_idx = sorted(target_frame_idx)
            if num_target_frames != len(target_frame_idx):
                print(f"error num_target_frames = {num_target_frames}, not equal to len(target_frame_idx), data_path: {data_path}")
                return self.__getitem__(random.randint(0, len(self) - 1))

            input_frames = [frames_pano[i] for i in input_frame_idx]
            input_frames_view_name = ""
            for input_idx in range(len(input_frames)):
                target_render_save_path.append(os.path.join(viewpoints_path, "viewpoints", input_frames[input_idx]["view_name"], "panoImage_1600_panoworld.jpg"))
                if input_idx == len(input_frames) - 1:
                    input_frames_view_name += input_frames[input_idx]["view_name"]
                else:
                    input_frames_view_name += (input_frames[input_idx]["view_name"] + "-")

            input_images, input_depths, input_masks, input_c2ws, succ_status = self.process_pano_frames(input_frames)
            if succ_status == False:
                print(f"error succ_status: {succ_status}, data_path: {data_path}")
                return self.__getitem__(random.randint(0, len(self) - 1))
            target_frames = [frames[i] for i in target_frame_idx]
            target_images, target_depths, target_intr, target_c2ws = self.process_frames(target_frames)

            # Reject extremely large scenes.
            if (target_c2ws[:, :3, 3] > 1e3).any():
                print(f"encounter large translation in target poses: {target_c2ws[:, :3, 3].max()}")
                assert False
            if (input_c2ws[:, :3, 3] > 1e3).any():
                print(f"encounter large translation in input poses: {input_c2ws[:, :3, 3].max()}")
                assert False

            # Poses must not contain NaNs.
            if any(torch.isnan(torch.det(target_c2ws[:, :3, :3]))):
                print(f"encounter nan in target poses: {target_c2ws[:, :3, :3]}")
                assert False
            if any(torch.isnan(torch.det(input_c2ws[:, :3, :3]))):
                print(f"encounter nan in input poses: {input_c2ws[:, :3, :3]}")
                assert False

            # Validate that rotation matrices have determinant 1.
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

            # --- Guard 1: check whether forward_avg is too small. ---
            if torch.norm(forward_avg) < 1e-6:
                # If camera directions cancel out, fall back to the default z axis.
                forward_avg = torch.tensor([0.0, 0.0, 1.0], device=input_c2ws.device).float()
            else:
                forward_avg = F.normalize(forward_avg, dim=0)

            # --- Guard 2: check down_avg and apply Gram-Schmidt. ---
            # First compute the orthogonalized down vector.
            down_avg_ortho = down_avg - down_avg.dot(forward_avg) * forward_avg

            if torch.norm(down_avg_ortho) < 1e-6:
                # Reaching this branch means:
                # 1. the original down_avg itself is zero,
                # 2. or the original down_avg is parallel to forward_avg.
                # Use a fallback down vector that is not parallel to forward.

                # Try the y axis first.
                fallback_down = torch.tensor([0.0, 1.0, 0.0], device=input_c2ws.device).float()
                # If forward is close to the y axis, switch to the x axis.
                if torch.abs(torch.dot(forward_avg, fallback_down)) > 0.99:
                    fallback_down = torch.tensor([1.0, 0.0, 0.0], device=input_c2ws.device).float()

                # Orthogonalize again.
                down_avg_ortho = fallback_down - fallback_down.dot(forward_avg) * forward_avg
                down_avg = F.normalize(down_avg_ortho, dim=0)
            else:
                # Normalize normally.
                down_avg = F.normalize(down_avg_ortho, dim=0)

            # Compute the right vector; the two guards above make the cross product safe.
            right_avg = torch.cross(down_avg, forward_avg, dim=0)

            # Build the transform matrix.
            pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
            pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)

            # Invert the orthogonal transform matrix.
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
            target_depths_mask = (target_depths > 0)
            input_image_indices = input_frame_idx + target_frame_idx
            input_image_indices = torch.tensor(input_frame_idx).long().unsqueeze(-1)
            target_image_indices = torch.tensor(target_frame_idx).long().unsqueeze(-1)
            input_room_ids = torch.tensor(input_frame_room_id).long()

            ret_dict = {
                "input_images": input_images,  # (num_input, 3, resize_pano_h, resize_pano_w)
                "target_images": target_images,  # (num_target, 3, resize_h, resize_w)
                "input_depths": input_depths, # (num_input, 1, resize_pano_h, resize_pano_w)
                "input_depths_mask": input_depths_mask, # (num_input, 1, resize_h, resize_w)
                "input_masks": (input_masks > 0), # (num_input, 1, resize_pano_h, resize_pano_w)
                "target_depths": target_depths, # (num_target, 1, resize_h, resize_w)
                "target_depths_mask": target_depths_mask, # (num_target, 1, resize_h, resize_w)
                "target_fxfycxcy": target_intr,  # (num_target, 4)
                "input_c2ws": input_c2ws,  # (num_input, 4, 4)
                "target_c2ws": target_c2ws, # (num_target, 4, 4)
                "input_image_indexs": input_image_indices,
                "input_room_ids": input_room_ids,
                "target_image_indexs": target_image_indices,
                "input_target_scene_name": viewpoints_path.split("/")[-1],
                "input_view_names": input_frames_view_name,
                "target_view_names": target_frames_view_name, # Set to "None".
                "target_render_save_path": target_render_save_path, # Set to "None".
                "input_data_path": data_path
            }

        except:
            traceback.print_exc()
            print(f"error loading data: {self.data_path[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return ret_dict
