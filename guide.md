# 操作手册（遥操作 / 推理）

G1 + 灵巧手的遥操作与策略推理操作步骤。分**遥操作（Teleop）**和**推理（Inference）**
两部分，每部分又分 **G1 板载（on board）** 和 **桌面端（on desktop）**。

- 默认身体来源是 Pico（`pico_manus_thread_server.py`）。换成 **SlimeVR** 见
  [SLIMEVR_MANUS_GUIDE.md](SLIMEVR_MANUS_GUIDE.md)。
- 脖子相关细节见 [NECK_TELEOP_GUIDE.md](NECK_TELEOP_GUIDE.md)。

---

## 1. 遥操作（Teleop）

### 1.1 G1 板载

开放电机串口、强制加载系统 libffi（GStreamer 需要）、激活环境，启动相机 + 脖子 server：

```bash
sudo chmod 777 /dev/ttyUSB0

export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7

conda activate sonic
cd ~/GR00T-WholeBodyControl     # 机器人上按实际路径调整
python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:5558 \
    --enable-pico --pico-ip 192.168.0.242 \
    --enable-neck-motor \
    --pose-zmq tcp://192.168.123.222:5570
```

### 1.2 桌面端

先启动 XRoboToolKit 守护进程：

```bash
sudo bash /opt/apps/roboticsservice/runService.sh
```

启动 WBC deploy：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

设置环境变量并启动身体/手/脖子 server（Pico + Manus）：

```bash
export GR00T_ROOT="$PWD"
export PYTHONPATH="$GR00T_ROOT/external_dependencies/gmr_shim:$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$LD_LIBRARY_PATH"
export PYTHONPATH=$PWD/GMR:$PYTHONPATH
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manus_thread_server.py --use_pico_hand
```

> 用 SlimeVR 身体源时，把这一步换成
> `python gear_sonic/scripts/slimevr_manus_thread_server.py`（见
> [SLIMEVR_MANUS_GUIDE.md](SLIMEVR_MANUS_GUIDE.md)）。

数据录制（脖子命令值 + 状态值）：

```bash
source .venv_teleop/bin/activate
python g1_data_server.py \
    --neck-zmq       tcp://localhost:5570 \
    --neck-state-zmq tcp://192.168.123.164:5560
```

可选的实时 ZED 画面查看：

```bash
python test_viewer.py --server 192.168.123.164 --port 5559 --show-stereo
```

Manus 手套 SDK（在 `:8000` 推流）：

```bash
../MANUS_Core_2.4.0_SDK/SDKClient_Linux$ ./SDKClient_Linux.out
```


---

## 2. 推理（Inference）

### 2.1 G1 板载

```bash
sudo chmod 777 /dev/ttyUSB0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7
conda activate sonic
cd ~/GR00T-WholeBodyControl
python realsense_server.py \
    --zed-only \
    --zmq-bind tcp://0.0.0.0:5558 \
    --enable-neck-motor \
    --pose-zmq tcp://192.168.123.222:5570
```

### 2.2 桌面端

启动 WBC deploy：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

图像客户端：
```bash
python test_viewer.py --server 192.168.123.164 --port 5559 --show-stereo
```

端口转发：

```bash
ssh -L 5000:localhost:5000 nebula101
```

回初始姿态，然后启动策略客户端（仅动作 + 含脖子）：

```bash
python apply_initial_pose.py

python g1_sonic_client.py --action-only --include-neck
```

---

## 3. 原生相机（不用 ZED mini / 不用脖子）

和第 1、2 节的区别：**不挂脖子上的 ZED mini、不驱动脖子电机**，直接用机器人
自带的原生 RealSense 头部相机。相机固定，所以整条脖子链路（`--enable-neck-motor`、
`--pose-zmq`、`pose_publisher.py`、脖子数据录制）全部去掉；客户端也去掉
`--include-neck`（默认走 `RSCamera`，收 3-part RGB/IR/Depth）。

相机 server 换成独立模块 `realsense_native_server.py`（不依赖 pyzed，直接用
pyrealsense2 驱动原生相机），REP 绑 5558，回 `[RGB jpeg, IR L|R jpeg, depth raw]`，
和客户端 `RSCamera`、`realsense_viewer.py` 的契约一致。

### 3.1 遥操作（Teleop）

对应第 1 节，只是相机换原生、去掉脖子。身体源（Pico 或 SlimeVR）和 Manus
手套 server 保持不变。

#### 3.1.1 G1 板载

只启动原生相机 server。录制端只取 RGB（`RealSenseClient` 只用 part 0 当 ego），
所以这里用 `--no-ir --no-depth` 发纯 RGB 最干净，避免 IR 混进 stereo 槽：

```bash
export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7
sudo killall -9 videohub_pc4
conda activate ruohai
cd ~/GR00T-WholeBodyControl
python realsense_native_server.py --zmq-bind tcp://0.0.0.0:5558 --no-ir --no-depth
```

> 相比第 1.1 节去掉了：`sudo chmod 777 /dev/ttyUSB0`（脖子电机串口）、
> `--enable-pico`（给 Pico 头显推 ZED 立体画面）、`--enable-neck-motor`、
> `--pose-zmq`。原生相机不是立体的，Pico 头显里不会有第一视角画面；若用
> SlimeVR 身体源则本来就不依赖头显画面。

#### 3.1.2 桌面端

先启动 XRoboToolKit 守护进程：

```bash
sudo bash /opt/apps/roboticsservice/runService.sh
```

启动 WBC deploy（不变）：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

设置环境变量并启动身体/手 server。**不加 `--enable_neck_pub`**（不发脖子角度）；
其余和第 1.2 节相同：

```bash
export GR00T_ROOT="$PWD"
export PYTHONPATH="$GR00T_ROOT/external_dependencies/gmr_shim:$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$PYTHONPATH"
export LD_LIBRARY_PATH="$GR00T_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64:$LD_LIBRARY_PATH"
export PYTHONPATH=$PWD/GMR:$PYTHONPATH
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manus_thread_server.py --use_pico_hand
```

> 用 SlimeVR 身体源时换成
> `python gear_sonic/scripts/slimevr_manus_thread_server.py`。

数据录制（**去掉两个 `--neck-*` 参数**，其余默认；图像仍从 5558 的原生相机
server 取 RGB）：

```bash
source .venv_teleop/bin/activate
python g1_data_server.py
```

实时画面查看（原生相机用 `realsense_viewer.py`，直连 5558 的 REP）：

```bash
python realsense_viewer.py --server 192.168.123.164 --port 5558
```

> `RSCamera`/`RealSenseClient`（客户端/录制）和 `realsense_viewer.py` 都是 REQ
> 直连这个 REP 口，三者不能同时连同一个口。录制时不要再开 viewer。

Manus 手套 SDK（在 `:8000` 推流，不变）：

```bash
../MANUS_Core_2.4.0_SDK/SDKClient_Linux$ ./SDKClient_Linux.out
```

### 3.2 推理（Inference）

#### 3.2.1 G1 板载

```bash
export LD_PRELOAD=/lib/aarch64-linux-gnu/libffi.so.7
sudo killall -9 videohub_pc4
conda activate ruohai
cd ~/GR00T-WholeBodyControl
python realsense_native_server.py --zmq-bind tcp://0.0.0.0:5558

# 只要 RGB（最轻）：python realsense_native_server.py --no-ir --no-depth
# 列设备：       python realsense_native_server.py --list-devices
```

> 不再需要 `sudo chmod 777 /dev/ttyUSB0`（脖子电机串口），也不需要
> `--enable-pico`（给 VR 头显推 ZED 立体画面）。

#### 3.2.2 桌面端

启动 WBC deploy（不变）：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

图像查看（原生相机用 `realsense_viewer.py`，直连 5558 的 REP）：

```bash
python realsense_viewer.py --server 192.168.123.164 --port 5558 --show-ir --show-depth
```

> 注意：`RSCamera`（推理客户端）和 `realsense_viewer.py` 都是 REQ 直连这个 REP
> 口，两者不能同时连。测相机时用 viewer，跑推理时用客户端。

端口转发：

```bash
ssh -L 5000:localhost:5000 nebula101
```

回初始姿态，然后启动策略客户端（**去掉 `--include-neck`**）：

```bash
python apply_initial_pose.py

python g1_sonic_client.py --action-only
# 或 RTC 版：python psix_rtc_sonic_client.py
```




slimevr -1 +1
pico +1 -1


pc ip: 192.168.123.222
g1 ip: 192.168.123.164
mac ip: 192.168.123.158
windows ip: 192.168.123.177


export CHECKPOINT_DIR=.runs/psix_finetune/psix-sonic-subtask-g1.sonic_psix_neck_rtc.flow1000.cosine.lr5.0e-05.b128.gpus8.2606241510
export CHECKPOINT_DIR=.runs/psix_finetune/psix-sonic-subtask-g1.sonic_psix_neck_rtc.flow1000.cosine.lr5.0e-05.b128.gpus8.2606220503
export CHECKPOINT_STEP=40000
python psix_rtc_sonic_client.py --include-neck
rsync -avz \
  --include='checkpoints/' \
  --include='checkpoints/ckpt_40000/***' \
  --exclude='checkpoints/*' \
  hongyi@nebula100:~/psi/.runs/psix_finetune/psix-sonic-subtask-g1.sonic_psix_neck_no_rtc.flow1000.cosine.lr5.0e-05.b128.gpus8.2606261137/ \
  .
