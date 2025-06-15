import time
import os
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import sys
import logging
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch.optim.lr_scheduler import StepLR
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--train_users", nargs='+', required=True)
parser.add_argument("--test_user", type=str, required=True)
parser.add_argument("--log_dir", type=str, required=True)  # TensorBoard日志保存路径
parser.add_argument("--model_dir", type=str, required=True)  # 模型保存路径
args = parser.parse_args()

train_users = args.train_users
test_users = [args.test_user]
log_dir = args.log_dir
model_dir = args.model_dir

# 创建模型目录（如果不存在）
os.makedirs(model_dir, exist_ok=True)

# 将 train_log.txt 保存到 model_dir 下
log_file_path = os.path.join(model_dir, "train_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

print = logging.info  # 替换内置 print 为 logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quaternion_to_rotation_matrix(quaternion):
    """将四元数 (w, x, y, z) 转换为旋转矩阵"""
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    R = np.zeros((quaternion.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R

def sliding_window(data, window_size, step_size):
    num_samples, num_features = data.shape
    windows = []
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

def load_multimodal_by_user_split(base_dir, train_users, test_users, window_size=30, step_size=1):
    user_emg_stats = {}
    for user in train_users + test_users:
        all_emg_user = []
        for sample_id in range(1, 21):
            sample_name = str(sample_id)
            emg_path = os.path.join(base_dir, "EMG_Press", "EMG", user, "aligned_emg_to_imu",
                                    f"aligned_emg_{sample_name}.csv")
            emg = pd.read_csv(emg_path).iloc[:, 1:].values
            all_emg_user.append(emg)

        all_emg_user = np.concatenate(all_emg_user, axis=0)
        mean = np.mean(all_emg_user, axis=0)
        std = np.std(all_emg_user, axis=0) + 1e-8  # 防止除以0
        user_emg_stats[user] = (mean, std)

    def process_sample(user, sample_name):
        emg_path = os.path.join(base_dir, "EMG_Press", "EMG", user, "aligned_emg_to_imu",
                                f"aligned_emg_{sample_name}.csv")
        emg = pd.read_csv(emg_path).iloc[:, 1:].values
        mean, std = user_emg_stats[user]
        emg = (emg - mean) / std

        pressure_dir = os.path.join(base_dir, "EMG_Press", "Pressure", user, sample_name,
                                    "aligned_pressure_log1p")
        pressures = []
        for f in range(1, 6):
            f_path = os.path.join(pressure_dir, f"aligned_pressure_f{f}_log1p.csv")
            finger_data = pd.read_csv(f_path).iloc[:, 3:].values
            pressures.append(finger_data)
        pressure = np.hstack(pressures)

        return pressure, emg

    pressures_all = []
    # 用于存储三个数据集
    train_emg, train_pressure, train_imu, train_keypoints = [], [], [], []
    fine_emg, fine_pressure, fine_imu, fine_keypoints = [], [], [], []
    test_emg, test_pressure, test_imu, test_keypoints = [], [], [], []

    all_users = train_users + test_users
    for user in all_users:
        for sample_id in range(1, 21):
            sample_name = str(sample_id)
            pressure, _ = process_sample(user, sample_name)
            pressures_all.append(pressure)

    pressures_concat = np.concatenate(pressures_all, axis=0)
    pressure_min = pressures_concat.min(axis=0)
    pressure_max = pressures_concat.max(axis=0)

    normalization_params = {
        "min": pressure_min,
        "max": pressure_max
    }

    def normalize_pressure(pressure):
        return (pressure - pressure_min) / (pressure_max - pressure_min + 1e-8)

    # 对每个用户和动作，做数据划分
    for user in train_users:
        for sample_id in range(1, 21):  # 每个动作1到20
            sample_name = str(sample_id)
            pressure, emg = process_sample(user, sample_name)
            pressure = normalize_pressure(pressure)

            wt6_path = os.path.join(base_dir, "IMU", "IMU", user, "WT6", f"{sample_name}_WT6.csv")
            wt1_path = os.path.join(base_dir, "IMU", "IMU", user, "WT1", f"{sample_name}_WT1.csv")
            wt6 = pd.read_csv(wt6_path).iloc[:, 1:].values
            wt1 = pd.read_csv(wt1_path).iloc[:, 1:].values
            wt1[:, :3] -= wt6[:, :3]
            wt1_rot = quaternion_to_rotation_matrix(wt1[:, 3:7])
            wt6_rot = quaternion_to_rotation_matrix(wt6[:, 3:7])
            wt6_rot_inv = np.transpose(wt6_rot, axes=(0, 2, 1))
            aligned_rot = np.einsum("bij,bjk->bik", wt6_rot_inv, wt1_rot)
            aligned_rot_flat = aligned_rot.reshape(aligned_rot.shape[0], -1)
            wt6_rot_flat = wt6_rot.reshape(wt6_rot.shape[0], -1)
            imu_feature = np.hstack([wt1[:, :3], aligned_rot_flat, wt6[:, :3], wt6_rot_flat])

            keypoints_path = os.path.join(base_dir, "video", "keypoints", user,
                                          f"{sample_name}_keypoints_smoothed.csv")
            keypoints = pd.read_csv(keypoints_path).iloc[:, 1:].values
            keypoints = keypoints.reshape(-1, 21, 3)

            # 直接全部作为训练数据
            imu_train = imu_feature
            emg_train = emg
            pressure_train = pressure
            keypoints_train = keypoints

            # sliding window
            imu_train = sliding_window(imu_train, window_size, step_size)
            emg_train = sliding_window(emg_train, window_size, step_size)
            pressure_train = sliding_window(pressure_train, window_size, step_size)
            keypoints_train = sliding_window(keypoints_train.reshape(keypoints_train.shape[0], -1), window_size, step_size)
            keypoints_train = keypoints_train.reshape(-1, window_size, 21, 3)

            train_emg.append(emg_train)
            train_imu.append(imu_train)
            train_pressure.append(pressure_train)
            train_keypoints.append(keypoints_train)

    for user in test_users:
        for sample_id in range(1, 21):  # 每个动作1到20
            sample_name = str(sample_id)
            pressure, emg = process_sample(user, sample_name)
            pressure = normalize_pressure(pressure)

            wt6_path = os.path.join(base_dir, "IMU", "IMU", user, "WT6", f"{sample_name}_WT6.csv")
            wt1_path = os.path.join(base_dir, "IMU", "IMU", user, "WT1", f"{sample_name}_WT1.csv")
            wt6 = pd.read_csv(wt6_path).iloc[:, 1:].values
            wt1 = pd.read_csv(wt1_path).iloc[:, 1:].values
            wt1[:, :3] -= wt6[:, :3]
            wt1_rot = quaternion_to_rotation_matrix(wt1[:, 3:7])
            wt6_rot = quaternion_to_rotation_matrix(wt6[:, 3:7])
            wt6_rot_inv = np.transpose(wt6_rot, axes=(0, 2, 1))
            aligned_rot = np.einsum("bij,bjk->bik", wt6_rot_inv, wt1_rot)
            aligned_rot_flat = aligned_rot.reshape(aligned_rot.shape[0], -1)
            wt6_rot_flat = wt6_rot.reshape(wt6_rot.shape[0], -1)
            imu_feature = np.hstack([wt1[:, :3], aligned_rot_flat, wt6[:, :3], wt6_rot_flat])

            keypoints_path = os.path.join(base_dir, "video", "keypoints", user,
                                          f"{sample_name}_keypoints_smoothed.csv")
            keypoints = pd.read_csv(keypoints_path).iloc[:, 1:].values
            keypoints = keypoints.reshape(-1, 21, 3)

            split_idx = int(len(imu_feature) * 0.2)

            imu_fine, imu_test = imu_feature[:split_idx], imu_feature[split_idx:]
            emg_fine, emg_test = emg[:split_idx], emg[split_idx:]
            pressure_fine, pressure_test = pressure[:split_idx], pressure[split_idx:]
            keypoints_fine, keypoints_test = keypoints[:split_idx], keypoints[split_idx:]

            # sliding window
            imu_fine = sliding_window(imu_fine, window_size, step_size)
            emg_fine = sliding_window(emg_fine, window_size, step_size)
            pressure_fine = sliding_window(pressure_fine, window_size, step_size)
            keypoints_fine = sliding_window(keypoints_fine.reshape(keypoints_fine.shape[0], -1), window_size, step_size)
            keypoints_fine = keypoints_fine.reshape(-1, window_size, 21, 3)

            fine_emg.append(emg_fine)
            fine_imu.append(imu_fine)
            fine_pressure.append(pressure_fine)
            fine_keypoints.append(keypoints_fine)

    return {
        "train": (
            np.concatenate(train_emg, axis=0),
            np.concatenate(train_imu, axis=0),
            np.concatenate(train_pressure, axis=0),
            np.concatenate(train_keypoints, axis=0)
        ),
        "fine": (
            np.concatenate(fine_emg, axis=0),
            np.concatenate(fine_imu, axis=0),
            np.concatenate(fine_pressure, axis=0),
            np.concatenate(fine_keypoints, axis=0)
        ),
        "normalization_params": normalization_params
    }

class MultiModalDataset(Dataset):
    def __init__(self, emg, imu, pressure, keypoints):
        self.emg = emg
        self.imu = imu
        self.pressure = pressure
        self.keypoints = keypoints

    def __len__(self):
        return self.emg.shape[0]

    def __getitem__(self, idx):
        emg = torch.tensor(self.emg[idx], dtype=torch.float32)
        imu = torch.tensor(self.imu[idx], dtype=torch.float32)
        pressure = torch.tensor(self.pressure[idx], dtype=torch.float32)
        keypoints = torch.tensor(self.keypoints[idx], dtype=torch.float32)

        return imu, emg, pressure, keypoints

# PositionalEncoding类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class TimeEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers):
        super(TimeEncoder, self).__init__()

        # 增加 CNN 模块
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ).to(device)

        # 原始输入投影和 Transformer 编码器
        self.input_projection = nn.Linear(d_model, d_model).to(device)  # 输入数据经过 CNN 后通道为 d_model
        self.positional_encoding = PositionalEncoding(d_model).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True).to(device),
            num_layers=num_layers
        )

    def forward(self, x):
        # 转换为 CNN 期望的输入形状 (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)  # 将 (batch_size, seq_len, input_dim) 转换为 (batch_size, input_dim, seq_len)

        # 通过 CNN 提取局部特征
        x = self.cnn(x)  # 输出形状: (batch_size, d_model, seq_len / 2)

        # 转换回 Transformer 输入形状 (batch_size, seq_len / 2, d_model)
        x = x.transpose(1, 2)

        # 投影到 Transformer 的维度
        x = self.input_projection(x)

        # 添加位置编码
        x = self.positional_encoding(x)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        return x

# Cross Attention模块
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.output = nn.Linear(d_model, d_model)

        self.attn_weights_last = None  # 👉 新增：存储最后一次forward的注意力权重

    def forward(self, query, key, value):
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        attn_output, attn_weights = self.attention(Q, K, V)
        self.attn_weights_last = attn_weights  # 👉 保存注意力权重
        output = self.output(attn_output)
        return output, attn_weights

class EMGEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=2, dropout=0.2):
        super(EMGEncoder, self).__init__()

        # LSTM 模块替代 Transformer 编码器
        self.lstm = nn.LSTM(input_dim, d_model, num_layers=num_layers, batch_first=True, dropout=dropout)

        # 添加位置编码 (如果需要的话)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        # LSTM 编码
        x, _ = self.lstm(x)

        # 可选: 加入位置编码
        x = self.positional_encoding(x)

        return x

class MultiTaskIMUEMGDecoderModel(nn.Module):
    def __init__(self, imu_dim, emg_dim, d_model, n_heads, num_layers):
        super().__init__()
        self.imu_encoder = TimeEncoder(imu_dim, d_model, n_heads, num_layers)
        self.emg_encoder = EMGEncoder(emg_dim, d_model, num_layers)

        self.cross_attention_imu_to_emg = CrossAttention(d_model, n_heads)#IMU to EMG 的attention　TODO
        self.cross_attention_emg_to_imu = CrossAttention(d_model, n_heads)

        self.fusion_fc = nn.Linear(2 * d_model, d_model)

        # Decoder layers
        decoder_layer_pressure = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        decoder_layer_pose = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)

        self.pressure_decoder = nn.TransformerDecoder(decoder_layer_pressure, num_layers=2)
        self.pose_decoder = nn.TransformerDecoder(decoder_layer_pose, num_layers=2)

        # Task-specific FC layers
        self.pressure_heads = nn.ModuleList([  # 只保留压力任务
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.ReLU()
            ) for _ in range(5)
        ])

        self.pose_fc = nn.Sequential(  # 姿态解码部分
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 21 * 3)  # 输出所有21个关节的3D坐标
        )

        self.imu_pose_fc = nn.Linear(d_model, 21 * 3)
        self.emg_pose_fc = nn.Linear(d_model, 21 * 3)

        self.imu_pressure_heads = nn.ModuleList([  # IMU压力任务
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.ReLU()
            ) for _ in range(5)
        ])

        self.emg_pressure_heads = nn.ModuleList([  # EMG压力任务
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.ReLU()
            ) for _ in range(5)
        ])

        self.query_proj = nn.Linear(d_model, d_model)
        self.enhance_fc = nn.Linear(d_model + 5 * 3, d_model)

        self.cross_attention_output_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 5),  # 预测5指压力
            nn.ReLU()
        )

    def forward(self, imu, emg):
        imu_feat = self.imu_encoder(imu)
        emg_feat = self.emg_encoder(emg)

        imu_to_emg, _ = self.cross_attention_imu_to_emg(imu_feat, emg_feat, emg_feat)# 删除cross_attention, TODO
        emg_to_imu, _ = self.cross_attention_emg_to_imu(emg_feat, imu_feat, imu_feat)
        cross_pred = self.cross_attention_output_fc(imu_to_emg)

        fused_feat = torch.cat([imu_to_emg, emg_to_imu], dim=-1)
        fused_feat = self.fusion_fc(fused_feat)

        # ----- 首先解码姿态 -----
        query_input = self.query_proj(fused_feat)
        pose_decoded = self.pose_decoder(query_input, fused_feat)
        pose_out = self.pose_fc(pose_decoded).view(fused_feat.size(0), fused_feat.size(1), 21, 3)

        # ----- 仅选择五个指尖的姿态 -----
        # 选取五个指尖的关节索引：thumb_tip(4), index_finger_tip(8), middle_finger_tip(12), ring_finger_tip(16), pinky_tip(20)
        finger_tip_indices = [4, 8, 12, 16, 20]
        pose_tips = pose_out[:, :, finger_tip_indices, :]  # 只取这五个指尖的姿态

        # ----- 将五个指尖姿态特征与融合特征结合后解码压力 -----
        enhanced_feat = torch.cat([fused_feat, pose_tips.view(fused_feat.size(0), fused_feat.size(1), -1)], dim=-1)
        enhanced_feat = self.enhance_fc(enhanced_feat)

        decoded = self.pressure_decoder(enhanced_feat, fused_feat)
        pressure_outs = []
        for i in range(5):
            out = self.pressure_heads[i](decoded)
            pressure_outs.append(out)
        pressure_out = torch.cat(pressure_outs, dim=-1)

        # 输出姿态和压力预测
        imu_pose_out = self.imu_pose_fc(imu_feat).view(imu_feat.size(0), imu_feat.size(1), 21, 3)
        emg_pose_out = self.emg_pose_fc(emg_feat).view(emg_feat.size(0), emg_feat.size(1), 21, 3)

        imu_pressure_out = torch.cat(
            [head(imu_feat) for head in self.imu_pressure_heads], dim=-1
        )

        emg_pressure_out = torch.cat(
            [head(emg_feat) for head in self.emg_pressure_heads],
            dim=-1
        )

        return {
            "pose_final": pose_out,
            "pressure_final": pressure_out,
            "pose_imu": imu_pose_out,
            "pose_emg": emg_pose_out,
            "pressure_imu": imu_pressure_out,
            "pressure_emg": emg_pressure_out,
            "cross_attn_pred": cross_pred
        }

def multitask_loss(output_dict, keypoints_gt, pressure_gt, criterion, criterion_pressure, lambda_weights):
    # 主任务输出
    pose_final = output_dict['pose_final']
    pressure_final = output_dict['pressure_final']

    # 辅助输出
    pose_imu = output_dict['pose_imu']
    pose_emg = output_dict['pose_emg']
    pressure_imu = output_dict['pressure_imu']
    pressure_emg = output_dict['pressure_emg']
    cross_attn_pred = output_dict["cross_attn_pred"]

    # 姿态损失
    loss_pose_final = criterion(pose_final, keypoints_gt)
    loss_pose_imu = criterion(pose_imu, keypoints_gt)
    loss_pose_emg = criterion(pose_emg, keypoints_gt)

    # 压力损失
    loss_press_final_fingers = []
    for i in range(5):
        pred = pressure_final[:, :, i]
        gt = pressure_gt[:, :, i]
        loss = criterion_pressure(pred, gt)
        loss_press_final_fingers.append(loss)
    loss_press_final = sum(loss_press_final_fingers)
    loss_press_imu = criterion_pressure(pressure_imu, pressure_gt)
    loss_press_emg = criterion_pressure(pressure_emg, pressure_gt)
    loss_cross_attn = criterion_pressure(cross_attn_pred, pressure_gt)

    total_loss = (
            lambda_weights['pose'] * loss_pose_final +
            lambda_weights['pose_imu'] * loss_pose_imu +
            lambda_weights['pose_emg'] * loss_pose_emg +
            lambda_weights['press'] * loss_press_final +
            lambda_weights['press_imu'] * loss_press_imu +
            lambda_weights['press_emg'] * loss_press_emg +
            lambda_weights['cross_attn'] * loss_cross_attn
    )

    # 在loss_dict中返回tensor版本
    loss_dict = {
        "total_loss": total_loss,
        "loss_pose_final": loss_pose_final,
        "loss_pose_imu": loss_pose_imu,
        "loss_pose_emg": loss_pose_emg,
        "loss_press_final": loss_press_final,
        "loss_press_imu": loss_press_imu,
        "loss_press_emg": loss_press_emg,
        "loss_cross_attn": loss_cross_attn
    }

    for i, loss in enumerate(loss_press_final_fingers):
        loss_dict[f"loss_press_f{i + 1}"] = loss

    return total_loss, loss_dict

def train_multitask_model(model, dataloader, optimizer, scheduler, criterion, criterion_pressure, lambda_weights,
                          num_epochs=100, patience=50, log_dir='runs/multitask_train', stage='train', model_dir='models'):
    # TensorBoard Writer，每个阶段创建不同的writer
    writer = SummaryWriter(log_dir=model_dir)  # 使用model_dir作为日志目录
    best_loss = float('inf')
    wait = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_start_time = time.time()

        loss_sums = {
            "total_loss": 0,
            "loss_pose_final": 0,
            "loss_pose_imu": 0,
            "loss_pose_emg": 0,
            "loss_press_final": 0,
            "loss_press_imu": 0,
            "loss_press_emg": 0,
            'loss_press_f1': 0,
            'loss_press_f2': 0,
            'loss_press_f3': 0,
            'loss_press_f4': 0,
            'loss_press_f5': 0,
            'loss_cross_attn': 0
        }

        data_loading_time = 0
        forward_time = 0
        backward_time = 0
        optimizer_time = 0

        batch_start_time = time.time()
        for imu, emg, pressure, keypoints in dataloader:
            data_loading_end_time = time.time()
            data_loading_time += data_loading_end_time - batch_start_time

            imu = imu.to(device, non_blocking=True)
            emg = emg.to(device, non_blocking=True)
            pressure = pressure.to(device, non_blocking=True)
            keypoints = keypoints.to(device, non_blocking=True)

            # Forward pass
            forward_start = time.time()
            optimizer.zero_grad()
            outputs = model(imu, emg)
            loss, loss_dict = multitask_loss(
                outputs, keypoints, pressure, criterion, criterion_pressure, lambda_weights)
            forward_end = time.time()
            forward_time += forward_end - forward_start

            # Backward pass
            backward_start = time.time()
            loss.backward()
            backward_end = time.time()
            backward_time += backward_end - backward_start

            # Optimizer step
            optimizer_start = time.time()
            optimizer.step()
            optimizer_end = time.time()
            optimizer_time += optimizer_end - optimizer_start

            # 更新各项loss
            for key in loss_sums:
                loss_sums[key] += loss_dict[key].item()

            batch_start_time = time.time()

        num_batches = len(dataloader)
        avg_total_loss = loss_sums["total_loss"] / num_batches

        # 在train阶段记录日志
        if stage == 'train':
            for key, value in loss_sums.items():
                writer.add_scalar(f"Loss/{key}", value / num_batches, epoch)
        # 在fine阶段记录日志
        elif stage == 'fine':
            # 在fine阶段记录日志，区分日志
            for key, value in loss_sums.items():
                writer.add_scalar(f"Fine_Loss/{key}", value / num_batches, epoch)

        # TensorBoard记录时间信息
        writer.add_scalar("Time/Data_loading", data_loading_time, epoch)
        writer.add_scalar("Time/Forward_pass", forward_time, epoch)
        writer.add_scalar("Time/Backward_pass", backward_time, epoch)
        writer.add_scalar("Time/Optimizer_step", optimizer_time, epoch)
        writer.add_scalar("Time/Epoch_total", time.time() - epoch_start_time, epoch)

        scheduler.step()

        # 记录参数和梯度到TensorBoard
        for name, param in model.named_parameters():
            writer.add_histogram(f"Parameters/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        print(f"=== Epoch {epoch + 1} ===, "
              f"total_loss = {avg_total_loss:.4f}, "
              f"data_loading={data_loading_time:.2f}s, forward={forward_time:.2f}s, "
              f"backward={backward_time:.2f}s, optimizer={optimizer_time:.2f}s")
        print(f"  ├── Pose Loss       : {loss_sums['loss_pose_final'] / num_batches:.4f}")
        print(f"  ├── Pose IMU        : {loss_sums['loss_pose_imu'] / num_batches:.4f}")
        print(f"  ├── Pose EMG        : {loss_sums['loss_pose_emg'] / num_batches:.4f}")
        print(f"  ├── Cross Attention Loss: {loss_sums['loss_cross_attn'] / num_batches:.4f}")
        print(f"  ├── Pressure (final): {loss_sums['loss_press_final'] / num_batches:.4f}")
        print(f"      ├─ Finger 1     : {loss_sums['loss_press_f1'] / num_batches:.4f}")
        print(f"      ├─ Finger 2     : {loss_sums['loss_press_f2'] / num_batches:.4f}")
        print(f"      ├─ Finger 3     : {loss_sums['loss_press_f3'] / num_batches:.4f}")
        print(f"      ├─ Finger 4     : {loss_sums['loss_press_f4'] / num_batches:.4f}")
        print(f"      └─ Finger 5     : {loss_sums['loss_press_f5'] / num_batches:.4f}")
        print(f"  ├── Pressure IMU    : {loss_sums['loss_press_imu'] / num_batches:.4f}")
        print(f"  └── Pressure EMG    : {loss_sums['loss_press_emg'] / num_batches:.4f}\n")

        # 保存最佳模型与Early Stopping机制
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            wait = 0
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, f"best_model_{stage}.pth"))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    writer.close()

if __name__ == "__main__":
    print("=== Step 1: Load and preprocess data ===")
    base_dir = "../../UIST/data"

    window_size = 30
    step_size = 5
    num_epochs = 100
    num_epochs_fine = 100

    splits = load_multimodal_by_user_split(base_dir, train_users, test_users, window_size, step_size)

    train_emg, train_imu, train_pressure, train_keypoints = splits["train"]
    fine_emg, fine_imu, fine_pressure, fine_keypoints = splits["fine"]
    test_emg, test_imu, test_pressure, test_keypoints = splits["test"]
    normalization_params = splits["normalization_params"]

    # 保存归一化参数
    np.savez(os.path.join(model_dir, "pressure_normalization_params.npz"), **normalization_params)

    # 创建数据加载器
    train_dataset = MultiModalDataset(train_emg, train_imu, train_pressure, train_keypoints)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=20, pin_memory=True)#TODO batch_size

    fine_dataset = MultiModalDataset(fine_emg, fine_imu, fine_pressure, fine_keypoints)
    fine_loader = DataLoader(fine_dataset, batch_size=256, shuffle=True, num_workers=20, pin_memory=True)

    print("=== Step 2: Create and configure model ===")
    model = MultiTaskIMUEMGDecoderModel(
        imu_dim=train_imu.shape[2],
        emg_dim=train_emg.shape[2],
        d_model=128,
        n_heads=4,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_fine = torch.optim.Adam(model.parameters(), lr=0.0001)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler_fine = StepLR(optimizer_fine, step_size=30, gamma=0.5)

    criterion = nn.MSELoss()
    criterion_pressure = nn.SmoothL1Loss()

    lambda_weights = {
        'pose': 1.0,
        'pose_imu': 0.3,
        'pose_emg': 0.3,
        'press': 3.0,  # 提升压力预测的损失权重
        'press_imu': 0.6,  # 提升辅助压力预测权重
        'press_emg': 0.6,
        'cross_attn': 0.5
    }

    print("=== Step 3: Train the model ===")
    train_multitask_model(model, train_loader, optimizer, scheduler, criterion, criterion_pressure, lambda_weights, num_epochs, model_dir=model_dir)
    print("Training complete.")

    print("=== Step 4: Fine tune the model ===")
    train_multitask_model(model, fine_loader, optimizer_fine, scheduler_fine, criterion, criterion_pressure, lambda_weights,
                          num_epochs_fine, stage="fine", model_dir=model_dir)
    print("Training complete.")
