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
