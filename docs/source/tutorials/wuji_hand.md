# Wuji 手部遥操作与数据采集

本文档介绍如何在 Sonic 遥操作流水线中接入 Wuji 手部硬件。

## 概述

Wuji 手部路径将 Dex3（7-DOF）替换为 Wuji 硬件（每手 20-DOF，5 指 × 4 关节）。身体遥操作链路不变，仅替换手部部分。

```
【输入设备 / 传感器】

  Manus 手套                        RealSense 相机（G1 上）
      │ ZMQ PULL :8000                   │ ZMQ REP :5558
      ▼                                  │
  pico_manus_thread_server.py            │
      │                                  │
      │ ZMQ PUB :5556  topic=pose        │
      │ （身体姿态；hand_joints 填默认张开值）
      ▼                                  │
  C++ deploy（G1）──► G1 身体关节        │
      │ ZMQ PUB :5557  topic=g1_debug    │
      │ （body_q / quat / action）        │
      │                                  │
      │ ZMQ PUB :5559  topic=wuji_hand   │
      │ （26D 手部追踪）                  │
      ▼                                  │
  wuji_hand_server.py  ×2（左 / 右）     │
      │ wuji-retargeting                 │
      │ wujihandpy ──► Wuji 硬件         │
      │ ZMQ PUB :5560  topic=wuji_state  │
      │ （hand state + action，各 20D）   │
      │                                  │
      └──────────────┐  ┌────────────────┘
                     ▼  ▼
               g1_data_server.py
                 SUB :5557  body state/action
                 REQ :5558  camera frames
                 SUB :5560  wuji hand state/action
                     │
                     ▼
               EpisodeWriter
               → data.json + color/*.jpg
```

**完整 ZMQ 端口一览：**

| 端口 | 协议 | 方向 | Topic | 内容 |
|------|------|------|-------|------|
| 8000 | PULL | Manus SDK → pico_manus | — | 原始 Manus 骨骼帧 |
| 5556 | PUB | pico_manus → C++ deploy | `pose` | msgpack：身体姿态 + hand_joints（Wuji 模式填默认张开值） |
| 5557 | PUB | C++ deploy → g1_data_server | `g1_debug` | msgpack：body_q_measured / quat / body_q_action 等 |
| 5558 | REQ/REP | g1_data_server → realsense_server | — | RGB 图像帧请求 |
| 5559 | PUB | pico_manus → wuji_hand_server | `wuji_hand` | msgpack：原始 26D 手部追踪字典（Wuji 模式新增） |
| 5560 | PUB | wuji_hand_server → g1_data_server | `wuji_state` | msgpack：测量状态 + 动作目标（每手 20D，Wuji 模式新增） |

---

## 安装

### 1. 安装 wuji-retargeting（运行 wuji_hand_server.py 的机器上执行）

```bash
cd GR00T-WholeBodyControl/wuji-retargeting
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
cd ..
```

### 2. 安装 wujihandpy

```bash
pip install wujihandpy
```

### 3. （G1 上运行时）将文件同步到机器人

如果 Wuji 手直接接在 G1 上，将以下文件同步到机器人：

```bash
rsync -av GR00T-WholeBodyControl/wuji-retargeting/ unitree@<g1_ip>:~/GR00T-WholeBodyControl/wuji-retargeting/
rsync -av GR00T-WholeBodyControl/wuji_hand_server.py  unitree@<g1_ip>:~/GR00T-WholeBodyControl/
rsync -av GR00T-WholeBodyControl/wuji_hand_real.sh    unitree@<g1_ip>:~/GR00T-WholeBodyControl/
```

然后在 G1 上安装依赖：

```bash
cd ~/GR00T-WholeBodyControl/wuji-retargeting
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e . && cd ..
pip install wujihandpy zmq msgpack numpy
```

---

## 启动流程（Wuji 手在 laptop 上）

按顺序依次启动各终端：

**Terminal 1 — 相机流（G1 上）**
```bash
python realsense_server.py
```

**Terminal 2 — C++ deploy（G1 或 laptop，不变）**
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/run_sim_loop.py

cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim

cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager real

```

**Terminal 3 — 身体遥操（laptop）**
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manus_thread_server.py --manager --hand_type wuji
```

**Terminal 4 — 左手 Wuji 控制（laptop）**
```bash
conda activate humdex
bash wuji_hand_real.sh left
```

**Terminal 5 — 右手 Wuji 控制（laptop）**
```bash
conda activate humdex
bash wuji_hand_real.sh right
```

**Terminal 6 — 数据采集（laptop）**
```bash
source .venv_teleop/bin/activate
python g1_data_server.py --hand-type wuji
```

---

## 启动流程（Wuji 手接在 G1 上）

当 Wuji 手物理安装在 G1 机器人上时，`wuji_hand_server.py` 在 G1 上运行。

**Terminal 3 — 身体遥操（laptop，不变）**
```bash
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manus_thread_server.py --manager --hand_type wuji
# 在 port 5559 绑定 PUB socket，G1 可订阅
```

**Terminal 4 — 左手 Wuji 控制（G1 上）**

先编辑 G1 上的 `wuji_hand_real.sh`，将 `TRACKING_HOST` 改为 laptop IP 192.168.123.222 ，然后：
```bash
conda activate humdex
bash wuji_hand_real.sh left
```

**Terminal 5 — 右手 Wuji 控制（G1 上）**
```bash
conda activate humdex
bash wuji_hand_real.sh right
```

**Terminal 6 — 数据采集（laptop）**
```bash
source .venv_teleop/bin/activate
python g1_data_server.py --hand-type wuji --wuji-state-host 192.168.123.164
```

> `wuji_hand_server.py` 将状态 PUB socket 绑定在 `tcp://*:5560`（监听所有网卡），laptop 通过 G1 的 IP 订阅即可。

---

## 配置说明

### wuji_hand_real.sh 变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HAND_SIDE` | `left` | `left` 或 `right`（也可作为第一个参数传入） |
| `SERIAL_NUMBER` | `""` | Wuji 设备序列号；留空自动选第一个 |
| `TRACKING_HOST` | `localhost` | 运行 `pico_manus_thread_server.py` 的机器 IP（Wuji 在 G1 时改为 laptop IP） |
| `TRACKING_PORT` | `5559` | ZMQ SUB 端口，接收 26D 手部追踪数据 |
| `STATE_PORT` | `5560` | ZMQ PUB 端口，发布测量状态 |
| `TARGET_FPS` | `50` | 控制循环频率（Hz） |
| `SMOOTH_STEPS` | `5` | 每个控制 tick 的插值步数 |
| `CONFIG` | `""` | retarget YAML 路径；留空自动解析 |

### Retarget YAML

默认自动加载：
```
wuji-retargeting/example/config/retarget_manus_<hand_side>.yaml
```
使用自定义配置：
```bash
python wuji_hand_server.py --hand_side left --config /path/to/my_config.yaml
```

---

## 键盘快捷键

`wuji_hand_server.py` 支持单键控制（无需回车）：

| 键 | 功能 |
|----|------|
| `k` | 切换 **follow**（跟随 Manus 追踪）↔ **default**（回零姿态） |
| `p` | 切换 **follow**（跟随 Manus 追踪）↔ **hold**（冻结当前姿态） |
| `Ctrl-C` | 优雅退出：平滑回零 → 关闭电机 → 退出 |

---

## 录制数据格式

使用 `--hand-type wuji` 录制的 episode 采用 `{"metadata": ..., "frames": [...]}` JSON 结构。

每帧 state / action：
- `states["hand_joints"]`：shape `(40,)` — 左手 20D + 右手 20D Wuji 测量关节角
- `actions["hand_joints"]`：shape `(40,)` — 左手 20D + 右手 20D Wuji 目标关节角
- `metadata["hand_type"]`：`"wuji"`

原有 Dex3 episode 不受影响：`hand_joints` 为 `(14,)`，`hand_type` 为 `"dex3"`。所有 `replay*.py` 脚本均已兼容新格式。

---

## 常见问题排查

**wuji_hand_server.py 一直等待追踪数据**
- 确认 `pico_manus_thread_server.py` 使用了 `--hand_type wuji` 启动
- 检查 port 5559 是否可达：`telnet <tracking_host> 5559`
- 检查防火墙：在追踪机器上执行 `sudo ufw allow 5559`

**wujihandpy 找不到设备**
- 列出已连接设备：`python -c "import wujihandpy; print(wujihandpy.list_devices())"`
- 指定序列号：在 `wuji_hand_real.sh` 中设置 `SERIAL_NUMBER=<序列号>`
- 确认 USB 已连接并有权限：`sudo chmod 666 /dev/ttyUSBx`

**g1_data_server.py 录制的 hand_joints 维度是 14 而非 40**
- 确认 `g1_data_server.py` 使用了 `--hand-type wuji`
- 确认 `wuji_hand_server.py` 正在运行并发布在 port 5560
- 当 Wuji 在 G1 上时，检查 `--wuji-state-host` 是否指向正确的 G1 IP

**找不到 retarget YAML**
- 默认路径相对于 `wuji_hand_server.py` 所在目录
- 确认已安装：`pip install -e wuji-retargeting/`
- 使用绝对路径显式指定：`--config /absolute/path/to/retarget_manus_left.yaml`
