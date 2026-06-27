# SlimeVR 身体 + Manus 手 → G1（GMR IK）

`pico_manus_thread_server.py` 的 SlimeVR 版本：身体来源从 Pico 头显换成 SlimeVR
（VMC/OSC），下游全部保持不变，所以 WBC deploy、`g1_data_server.py`、
`realsense_server.py` 都无需改动即可消费。

```
SlimeVR app ── VMC/OSC UDP :39539 ──► 桌面: slimevr_manus_thread_server.py
Manus 手套 ── ZMQ :8000 ────────────►   ├─ GMR IK → joint_pos29 + pelvis_quat
                                        ├─ PUB :5556  pose（v1: joint_pos/joint_vel/
                                        │             body_quat/frame_index + Dex3 手）
                                        └─ PUB :5570  脖子 [yaw, pitch] JSON
G1 Orin: realsense_server.py --pose-zmq tcp://<桌面>:5570  驱动 2-DOF 脖子
```

pose 消息走 **v1（关节空间）协议** —— 和 `pico_gmr_thread_server.py` 同一套契约。
身体链路：SlimeVR `body_frame` → `apply_bvh_like_coordinate_transform` →
`gmr_rename_and_footmod("nokov")` →
`GMR(src_human="bvh_nokov", tgt_robot="unitree_g1")` → `qpos[7:36]` 重排到 IsaacLab
顺序 + `qpos[3:7]` 作为 pelvis 四元数。

## 依赖

自包含的 SlimeVR 代码在
[external_dependencies/slimevr_body/](external_dependencies/slimevr_body/)
（adapter、FK viewer、BVH 骨架、坐标变换、键盘）。除此之外，桌面 `sonic` 环境还需要：

- **GMR 包** `general_motion_retargeting`，且带有 `bvh_nokov` 源骨架（即真正的 GMR IK
  包，和 `pico_gmr_thread_server.py` 用的同一个 —— **不是** `gmr_shim`）。注意别让 shim
  在 `PYTHONPATH` 上遮蔽真包。
- `python-osc`（SlimeVR VMC 接收）—— `pip install python-osc`
- Manus SDK 在 `tcp://localhost:8000` 推流（运行 Manus `SDKClient`）。

## 运行

### SlimeVR app（操作员）
配对所有 tracker（**必须包含一个头部 tracker**），在 SlimeVR app 里校准/reset，
启用 VMC/OSC 发送、指向桌面 IP 的 `39539` 端口。没有头部 tracker 时，FK 的 `Head`
会回退到 `Neck`，脖子就不会动。

### 桌面
参数不走命令行，全部写在脚本**文件头的 `CONFIG` 区块**，按需手动改（手套 SN、身高、端口等）。改好后无参运行：

```bash
micromamba activate sonic
cd /path/to/GR00T-WholeBodyControl

# Manus SDK（单独终端）→ 在 :8000 推手套数据
~/.../MANUS_Core_*_SDK/SDKClient_Linux/SDKClient_Linux.out

# 先编辑 gear_sonic/scripts/slimevr_manus_thread_server.py 顶部的 CONFIG，再运行：
python gear_sonic/scripts/slimevr_manus_thread_server.py
```
预期日志：`[slimevr] VMC reader ...`、`GMR retargeter ready.`、
`[pose] ZMQ PUB (v1) bound ...`、`[neck] ZMQ PUB bound to port 5570`，随后周期性的
`[slimevr] frame=... gmr=...ms fps~...`。

### G1 Orin（脖子 + 相机）—— 和 Pico 流程完全一致
```bash
sudo chmod 777 /dev/ttyUSB0
python realsense_server.py --zed-only --zmq-bind tcp://0.0.0.0:5558 \
    --enable-neck-motor --pose-zmq tcp://<桌面IP>:5570
```

### 录制 —— 不变
`g1_data_server.py` 加 `--neck-zmq tcp://localhost:5570 --neck-state-zmq
tcp://192.168.123.164:5560`，和之前一样记录脖子的命令值与状态值。

## 控制（在 server 自己的终端里）

和 `g1_data_server.py` 的 `s/q/d` 录制按键相互独立（不同进程/终端，且键位不重合）：

| 按键 | 作用 |
|------|------|
| `k`  | 切换 发送 开/关（gate 机器人） |
| `p`  | 切换 hold（把身体冻结在上一帧姿态） |
| `q`  | 退出 |
| `e`  | 急停 |

`--no-keyboard` 为一直发送、无 gating。`Ctrl-C` 始终可退出。

## 配置常量（脚本文件头 `CONFIG` 区块）

所有参数都在 `gear_sonic/scripts/slimevr_manus_thread_server.py` 顶部，手动编辑后无参运行：

| 常量 | 默认 | 用途 |
|------|------|------|
| `HUMAN_HEIGHT` | `1.60` | GMR 肢体缩放（米） |
| `FORMAT` | `"nokov"` | GMR 源骨架 → `src_human=bvh_<FORMAT>` |
| `POSE_ADDR` | `"tcp://*:5556"` | pose PUB 绑定 |
| `POSE_TOPIC` | `"pose"` | ZMQ topic 前缀 |
| `TARGET_FPS` | `50` | 主循环目标帧率 |
| `NUM_FRAMES_TO_SEND` | `5` | 每次发布的滑窗帧数 |
| `VMC_IP` / `VMC_PORT` | `"0.0.0.0"` / `39539` | VMC 监听地址 |
| `VMC_TIMEOUT_S` | `0.5` | VMC 帧超时（秒） |
| `VMC_BVH_SCALE` | `0.01` | BVH 全局缩放 |
| `BVH_PATH` | `""` | 留空=用自带 `assets/bvh-recording.bvh` |
| `ENABLE_HAND` | `True` | `False`=只发身体 + 脖子 |
| `LEFT_GLOVE_SN` / `RIGHT_GLOVE_SN` | Manus SN | 改成你们的手套 |
| `ENABLE_NECK_PUB` | `True` | `False`=关闭脖子流 |
| `NECK_PUB_PORT` | `5570` | 脖子 `[yaw,pitch]` PUB |
| `NECK_RETARGET_SCALE` | `1.5` | 缩放脖子运动幅度 |
| `ENABLE_KEYBOARD` | `True` | `False`=一直发送、无 gating |

## 脖子说明

脖子是 `Head` 相对 `Spine3` 的 `[yaw, pitch]`（YXZ 欧拉），计算方式与 wire 格式都和
`pico_manus_thread_server.py` 一致。方向在 G1 端用 `NECK_YAW_SIGN` / `NECK_PITCH_SIGN`
调，幅度用脚本 CONFIG 的 `NECK_RETARGET_SCALE` 调（见 `NECK_TELEOP_GUIDE.md`）。
SlimeVR 的 VMC 坐标系和 Pico/Unity 略有差异，首次上机大概率需要翻一个符号。

## 首次上机检查（bring-up）

这套和 humdex 的 sonic 链路逐项对齐过（坐标变换、`gmr_rename("nokov")`、
`GMR(src_human="bvh_nokov", actual_human_height=1.60)`、`offset_to_ground=True`、
`qpos[7:36]→IsaacLab` 重排、归一化 body_quat、v1 payload 含
`timestamp_realtime/heading_increment/catch_up`、滑窗 5 帧、VMC 参数）。剩下几处
**只能上机确认 / 可能需要现场调**，按顺序过一遍：

1. **真 GMR 包可用且不被 shim 遮蔽**：脚本用
   `from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting`
   （和 `pico_gmr_thread_server.py` 同一个真 GMR IK 包）。确保 `PYTHONPATH` 上的
   `external_dependencies/gmr_shim` **不会**抢先解析 `general_motion_retargeting`，
   否则会报找不到 `motion_retarget`。
2. **`FORMAT` 对应的源骨架存在**：默认 `nokov` → `src_human=bvh_nokov`。GMR 包里要有
   这个源骨架配置；若你们 SlimeVR 调好的是别的格式，改 CONFIG 的 `FORMAT`。
3. **头部 tracker 必须戴**：没有头部 tracker 时 FK 的 `Head` 回退到 `Neck`，
   脖子不动。
4. **脖子方向**：VMC 坐标系和 Pico 不同，先慢慢转头，在 G1 端按需翻
   `NECK_YAW_SIGN` / `NECK_PITCH_SIGN`，再用 `NECK_RETARGET_SCALE` 调幅度。
5. **deploy 端手动启动**：本脚本不发 `build_command_message`（与 `pico_gmr` 一致）。
   在 WBC 侧按 `]` 开始控制、再按 `Enter` 使能 ZMQ 流。
6. **gating 与录制不冲突**：本脚本键盘 `k/p/q/e` 在它自己的终端；`g1_data_server.py`
   的 `s/q/d` 在另一个终端，互不影响。先用 `k` 确认能 gate 住机器人再正式操作。
