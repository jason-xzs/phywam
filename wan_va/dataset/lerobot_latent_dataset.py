# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    with Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.use_phys_memory = getattr(config, 'use_phys_memory', False)
        cfg_phys_memory_path = getattr(config, 'phys_memory_path', None)
        if cfg_phys_memory_path is None:
            phys_memory_dirname = getattr(config, 'phys_memory_dirname', 'physics_features3.4')
            self.phys_memory_path = Path(repo_id) / phys_memory_dirname
        else:
            self.phys_memory_path = Path(cfg_phys_memory_path)
        self.phys_memory_dim = int(getattr(config, 'phys_memory_dim', 0) or 0)
        self.phys_memory_block_size = int(getattr(config, 'phys_memory_block_size', 16) or 16)
        self.phys_memory_align_mode = getattr(config, 'phys_memory_align_mode', 'per_latent_frame')
        if self.phys_memory_align_mode not in ('per_latent_frame', 'phase_tokens'):
            raise ValueError(f"Unsupported phys_memory_align_mode: {self.phys_memory_align_mode}")
        self.phys_memory_max_tokens = int(getattr(config, 'phys_memory_max_tokens', 8) or 8)
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def _build_phys_memory_candidates(self, episode_chunk, episode_index, start_frame, end_frame):
        if self.phys_memory_path is None:
            return []
        root = Path(self.phys_memory_path)
        names = [
            f"episode_{episode_index:06d}.npz",
            f"episode_{episode_index:06d}.npy",
            f"episode_{episode_index:06d}_{start_frame}_{end_frame}.npz",
            f"episode_{episode_index:06d}_{start_frame}_{end_frame}.npy",
        ]
        candidates = []
        for name in names:
            candidates.append(root / f"chunk-{episode_chunk:03d}" / name)
            candidates.append(root / name)
        return candidates

    def _find_phys_memory_file(self, episode_chunk, episode_index, start_frame, end_frame):
        for file_path in self._build_phys_memory_candidates(
            episode_chunk, episode_index, start_frame, end_frame
        ):
            if file_path.exists():
                return file_path
        return None

    def _load_phys_memory_payload(self, file_path):
        if file_path.suffix == '.npz':
            payload = np.load(file_path)
            if 'features' in payload:
                features = payload['features']
            elif 'phys_feat' in payload:
                features = payload['phys_feat']
            else:
                raise KeyError(f"Missing 'features' in physics memory file: {file_path}")
            block_starts = payload['block_starts'] if 'block_starts' in payload else None
            block_ends = payload['block_ends'] if 'block_ends' in payload else None
            block_size = int(payload['block_size']) if 'block_size' in payload else self.phys_memory_block_size
            origin = int(payload['start_frame']) if 'start_frame' in payload else 0
        else:
            features = np.load(file_path)
            block_starts = None
            block_ends = None
            block_size = self.phys_memory_block_size
            origin = 0
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None]
        if features.ndim != 2:
            raise ValueError(f"Physics memory features should be [T, D], got {features.shape}: {file_path}")
        if self.phys_memory_dim > 0 and features.shape[-1] != self.phys_memory_dim:
            raise ValueError(
                f"Physics memory dim mismatch for {file_path}: expected {self.phys_memory_dim}, got {features.shape[-1]}"
            )
        return features, block_starts, block_ends, block_size, origin

    def _load_phys_memory(self, episode_index, start_frame, end_frame, latent_frame_ids):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        file_path = self._find_phys_memory_file(episode_chunk, episode_index, start_frame, end_frame)
        if file_path is None:
            raise FileNotFoundError(
                f"Physics memory not found for episode={episode_index}, start={start_frame}, end={end_frame}. "
                f"Checked under phys_memory_path={self.phys_memory_path}"
            )
        features, block_starts, block_ends, block_size, origin = self._load_phys_memory_payload(file_path)
        if self.phys_memory_align_mode == 'phase_tokens':
            return self._pack_phase_phys_memory(
                features, block_starts, block_ends, block_size, origin,
                start_frame, end_frame, latent_frame_ids)
        frame_ids = np.asarray(latent_frame_ids, dtype=np.int64)
        if features.shape[0] == len(frame_ids):
            aligned = features
        elif block_starts is not None and block_ends is not None:
            block_ends = np.asarray(block_ends, dtype=np.int64)
            indices = np.searchsorted(block_ends, frame_ids, side='right')
            indices = np.clip(indices, 0, features.shape[0] - 1)
            aligned = features[indices]
        else:
            indices = (frame_ids - int(origin)) // max(int(block_size), 1)
            indices = np.clip(indices, 0, features.shape[0] - 1)
            aligned = features[indices]
        return torch.from_numpy(np.asarray(aligned, dtype=np.float32))

    def _pack_phase_phys_memory(
        self, features, block_starts, block_ends, block_size, origin,
        start_frame, end_frame, latent_frame_ids,
    ):
        max_tokens = max(int(self.phys_memory_max_tokens), 1)
        token_count = min(features.shape[0], max_tokens)
        padded = np.zeros((max_tokens, features.shape[-1]), dtype=np.float32)
        mask = np.zeros((max_tokens,), dtype=np.bool_)
        spans = np.zeros((max_tokens, 2), dtype=np.int64)

        padded[:token_count] = features[:token_count]
        mask[:token_count] = True
        if block_starts is not None and block_ends is not None:
            starts = np.asarray(block_starts, dtype=np.int64)[:token_count]
            ends = np.asarray(block_ends, dtype=np.int64)[:token_count]
        else:
            starts = (
                np.arange(features.shape[0], dtype=np.int64) * max(int(block_size), 1)
                + int(origin)
            )[:token_count]
            ends = (
                starts + max(int(block_size), 1)
            )[:token_count]

        raw_frame_ids = np.asarray(latent_frame_ids, dtype=np.int64)
        latent_frame_count = max((len(raw_frame_ids) - 1) // 4 + 1, 1)
        latent_key_frames = raw_frame_ids[::4][:latent_frame_count]
        if len(latent_key_frames) == 0:
            latent_key_frames = np.asarray([int(start_frame)], dtype=np.int64)
            latent_frame_count = 1
        span_starts = np.searchsorted(latent_key_frames, starts, side='left')
        span_ends = np.searchsorted(latent_key_frames, ends, side='left')

        spans[:token_count, 0] = np.clip(
            span_starts, 0, max(latent_frame_count - 1, 0))
        spans[:token_count, 1] = np.clip(span_ends, 0, latent_frame_count)
        spans[:token_count, 1] = np.maximum(
            spans[:token_count, 1], spans[:token_count, 0] + 1)
        spans[:token_count, 1] = np.clip(spans[:token_count, 1], 1, latent_frame_count)
        if features.shape[0] > max_tokens:
            spans[token_count - 1, 1] = max(
                spans[token_count - 1, 1], latent_frame_count)
        return {
            'feat': torch.from_numpy(padded),
            'mask': torch.from_numpy(mask),
            'spans': torch.from_numpy(spans),
        }

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        if self.use_phys_memory:
            phys_file = self._find_phys_memory_file(episode_chunk, episode_index, start_frame, end_frame)
            if phys_file is None:
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        if self.use_phys_memory:
            phys_mem = data_dict['phys_mem_feat']
            if isinstance(phys_mem, dict):
                out_dict['phys_mem_feat'] = phys_mem['feat']
                out_dict['phys_mem_mask'] = phys_mem['mask']
                out_dict['phys_mem_spans'] = phys_mem['spans']
            else:
                out_dict['phys_mem_feat'] = phys_mem
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        if self.use_phys_memory:
            ori_data_dict['phys_mem_feat'] = self._load_phys_memory(
                episode_index,
                local_start_frame,
                local_end_frame,
                latent_frame_ids,
            )
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
