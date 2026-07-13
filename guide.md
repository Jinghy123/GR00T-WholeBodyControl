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

端口转发/启动server：

```bash
ssh -L 5000:localhost:5000 nebula101

cd /mnt/data/weiduo/heng/DreamZero-private
bash start.sh
```

回初始姿态，然后启动策略客户端（仅动作 + 含脖子）：

```bash
python apply_initial_pose.py

python g1_sonic_client.py --action-only --include-neck

# 或 RTC 版：python psix_rtc_sonic_client.py
bash client.sh
```

---

## 3. 原生相机（不用 ZED mini / 不用脖子）

和第 1、2 节的区别：**不挂脖子上的 ZED mini、不驱动脖子电机**，直接用机器人
自带的原生 RealSense 头部相机。相机固定，所以整条脖子链路（`--enable-neck-motor`、
`--pose-zmq`、`pose_publisher.py`、脖子数据录制）全部去掉；客户端也去掉
`--include-neck`（默认走 `RSCamera`，收 3-part RGB/IR/Depth）。

相机 server 换成独立模块 `realsense_native_server.py`（不依赖 pyzed，直接用
pyrealsense2 驱动原生相机），三路输出**互不抢帧**：

- **REP 5558**：只给录制/推理客户端（`RSCamera`/`RealSenseClient`），回
  `[RGB jpeg, IR L|R jpeg, depth raw]`，契约与原客户端一致，满帧 30fps。
  **REP 是一问一答轮流服务，这个口只能有一个客户端**——多一个连上来（比如
  REQ 模式的 viewer）就各分一半帧，谁都只剩 15fps。
- **PUB 5559**（`--viewer-pub`，默认开）：viewer 专用广播口，内容同上三分片。
  随便开几个 viewer 都不影响录制（viewer 慢了只丢自己的帧）。
- **`--enable-pico`**：把 RGB 用 H.264/TCP 推给 Pico 头显（单目第一视角，
  从 ZED server 移植）。

### 3.1 遥操作（Teleop）

对应第 1 节，只是相机换原生、去掉脖子。身体源（Pico 或 SlimeVR）和 Manus
手套 server 保持不变。

#### 3.1.1 G1 板载

只启动原生相机 server（`--enable-pico` 把第一视角推到 Pico 头显；用
SlimeVR 身体源不需要头显画面时可去掉）。头显画面二选一：

**立体 IR（推荐遥操用，有真实深度感，灰度）**——把 D435i 左右红外拼成
左|右立体帧（1280x480）推给 Pico，格式和 ZED 立体一样。会自动关掉 IR
散斑投射器（否则满屏光点）：

```bash
sudo killall -9 videohub_pc4
conda activate ruohai
cd ~/GR00T-WholeBodyControl
python realsense_native_server.py --no-depth \
    --enable-pico --pico-ip 192.168.0.241 --pico-source ir
```

**单目 RGB（彩色，无立体）**：

```bash
python realsense_native_server.py --no-ir \
    --enable-pico --pico-ip 192.168.0.241
```

> 相比第 1.1 节去掉了：`sudo chmod 777 /dev/ttyUSB0`（脖子电机串口）、
> `--enable-neck-motor`、`--pose-zmq`。Pico 里的第一视角是单目 RGB（原生相机
> 不是立体的），不是 ZED 那种立体画面。
>
> 环境务必用 `ruohai`（系统源码编译的 librealsense 2.58 / v4l2 内核后端，枚举
> 稳定、满帧率）；`vision` 里 pip 的 pyrealsense2 是 RSUSB/libusb 后端，USB2 上
> 枚举不稳、帧率会被压到 15。`killall videohub_pc4` 是先释放占着相机的服务。
> server 已自动关掉 `auto_exposure_priority`（否则暗光下 30→15）；房间很暗想
> 锁死 30 可再加固定曝光，如 `--exposure 8000`。
>
> **USB2 带宽：RGB + depth + IR 三路开不满，最多选两路。** 实测（640x480@30）：
> RGB+depth（`--no-ir`）满帧 30fps；RGB+双IR（`--no-depth`）也满帧 30fps；
> 三路全开掉到 ~22。所以推 IR 立体给 Pico 时必须 `--no-depth`。三路都想满帧
> 只能把相机插到 USB3 口（机身 Bus 02 的 5000M 口）。录制只用 RGB。

**Pico 看不到画面时，按顺序排查（都踩过）：**

1. 看 server 启动命令有没有带 `--enable-pico`——重启 server 时最容易漏掉。
   连上时 server 会打印 `[PicoStreamer] Connected to Pico 192.168.0.241:12345`。
2. G1 上 `ping 192.168.0.241`。不通的话是 G1 的 USB WiFi 网卡（rtl8852bu）
   假死了——nmcli 显示已连接、信号满格，但实际链路不通。重启连接即可恢复：

   ```bash
   sudo nmcli connection down use-psi && sudo nmcli connection up use-psi
   ```

3. 确认推流 TCP 已建立：`ss -tn | grep 12345` 应有一条到
   `192.168.0.241:12345` 的 ESTAB。server 的 Pico 线程每 2 秒自动重连，
   网络恢复后不用重启 server。
4. 画面卡顿但能看：WiFi 网段 RTT 波动大（实测 56–569ms），是无线环境问题，
   不是 server 的问题。

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

实时画面查看：**一律用 `--sub`**（SUB 连 5559 广播口，录制时开着也不掉帧）：

```bash
python realsense_viewer.py --server 192.168.123.164 --sub
```

> 千万别在录制时用不带 `--sub` 的 REQ 模式连 5558——REP 轮流服务，viewer 会
> 抢走录制端一半的帧，两边都只剩 15fps（实测踩过这个坑）。

Manus 手套 SDK（在 `:8000` 推流，不变）：

```bash
../MANUS_Core_2.4.0_SDK/SDKClient_Linux$ ./SDKClient_Linux.out
```

### 3.2 推理（Inference）

#### 3.2.1 G1 板载

```bash
sudo killall -9 videohub_pc4
conda activate ruohai
cd ~/GR00T-WholeBodyControl
python realsense_native_server.py --no-ir

# 只要 RGB（最轻）：python realsense_native_server.py --no-ir --no-depth
# 列设备：       python realsense_native_server.py --list-devices
```

> 不再需要 `sudo chmod 777 /dev/ttyUSB0`（脖子电机串口）；推理不需要
> `--enable-pico`。`--no-ir` 的原因见 3.1.1（USB2 带宽，IR 会把 30fps 拖到 ~22）。

#### 3.2.2 桌面端

启动 WBC deploy（不变）：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real
```

图像查看：**一律用 `--sub`**（SUB 连 5559 广播口，跑推理时开着也不抢帧）。
看 depth 加 `--show-depth`；别用 `--show-ir`（server 是 `--no-ir`）：

```bash
python realsense_viewer.py --server 192.168.123.164 --sub --show-depth
```

> 不带 `--sub` 的 REQ 模式会直连 5558 的 REP 口，和推理客户端各抢一半帧
> （两边都掉到 15fps）。只有单独调试相机、且没有其他客户端时才可以用 REQ 模式。

端口转发/启动server：

```bash
ssh -L 5000:localhost:5000 nebula101

cd /mnt/data/weiduo/heng/DreamZero-private
bash start.sh
```

回初始姿态，然后启动策略客户端（**去掉 `--include-neck`**）：

```bash
python apply_initial_pose.py

python g1_sonic_client.py --action-only
# 或 RTC 版：python psix_rtc_sonic_client.py
bash client.sh
```




## 4. 带脖子 + 脖子上装 RealSense（不用 ZED mini）

和第 1 节的区别只有一个：脖子上的相机从 ZED mini 换成 RealSense，板载 server 用
`realsense_native_server.py`（已内置和 `realsense_server.py` 相同的脖子电机驱动，
`--enable-neck-motor` / `--pose-zmq` / `--neck-state-pub` 同名同义）。
桌面端（deploy、pico_manus_thread_server、g1_data_server）与第 1.2 节**完全相同**：
`g1_data_server.py` 照旧带两个 `--neck-*` 参数，state/action 的脖子两维不变，
录的图自动是 RealSense RGB。

### 4.1 G1 板载

```bash
sudo chmod 777 /dev/ttyUSB0
sudo killall -9 videohub_pc4
conda activate ruohai        # 需要 pip install dynamixel-sdk（一次性）
cd ~/GR00T-WholeBodyControl
python realsense_native_server.py --no-depth \
    --enable-pico --pico-ip 192.168.0.241 --pico-source ir \
    --enable-neck-motor \
    --pose-zmq tcp://192.168.123.222:5570
```

> 头显画面二选一同 3.1.1（`--pico-source ir` 立体灰度需 `--no-depth`；
> 单目彩色用 `--no-ir` 且去掉 `--pico-source ir`）。USB2 带宽限制同 3.1.1。

### 4.2 桌面端

全部照抄第 1.2 节：deploy、`pico_manus_thread_server.py`、
`g1_data_server.py --neck-zmq tcp://localhost:5570 --neck-state-zmq tcp://192.168.123.164:5560`。
实时画面查看用 `realsense_viewer.py --server 192.168.123.164 --sub`（不是 test_viewer.py）。

### 4.3 推理（Inference）

和第 3.2 节的区别：加回整条脖子链路；和第 2 节的区别：ZED 换成脖子上的
RealSense。客户端照常带 `--include-neck`，**代码不用改**：`ZedNeckCamera`
只解第 0 分片的 RGB JPEG，native server 的 3-part 回复第 0 片正是 RGB JPEG，
直接兼容。

#### 4.3.1 G1 板载

```bash
sudo chmod 777 /dev/ttyUSB0
sudo killall -9 videohub_pc4
conda activate ruohai
cd ~/GR00T-WholeBodyControl
python realsense_native_server.py --no-ir \
    --enable-neck-motor \
    --pose-zmq tcp://192.168.123.222:5570
```

> 推理不需要 `--enable-pico`；`--no-ir` 保证 RGB+depth 满 30fps（USB2 带宽
> 见 3.1.1）。启动后确认 `[Neck]` 无红字、开始出帧，再启桌面端客户端。

#### 4.3.2 桌面端

WBC deploy、端口转发、policy server 与第 2.2 节完全相同：

```bash
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq real

# 启动server
ssh -L 5000:localhost:5000 nebula101
cd /mnt/data/weiduo/heng/DreamZero-private
bash start.sh
```

回初始姿态，然后启动策略客户端（**带 `--include-neck`**，checkpoint 是
RealSense 数据训练的再加 `--use-realsense`，见下面分辨率说明）：

```bash
python apply_initial_pose.py # 如果pose teleop时候选择一致则不用跑

# psix（发原图，服务端 resize，无 --use-realsense）：
python psix_rtc_sonic_client.py --include-neck

# 或 RTC 版：
python g1_sonic_client_rtc.py \
    --policy-host 127.0.0.1 --policy-port 5000 \
    --prompt "..." \
    --execution-horizon 12 --inference-delay 10 --guidance-weight 5.0 --kv-scheme stride1 \
    --include-neck --use-realsense

# 或非 RTC 版：
python g1_sonic_client.py --action-only --include-neck --use-realsense

# 或 DZ 版（policy 在 nebula102，48014 端口没有公网映射，必须先转发：
#   ssh -N -f -L 48014:localhost:48014 nebula102
# --host 填别的 IP 连不上，走默认 localhost）：
python dz_client.py --include-neck --use-realsense
```

画面查看照旧 `python realsense_viewer.py --server 192.168.123.164 --sub`。

> 两个注意点：
> - **启动顺序**：等 G1 server 开始出帧再启客户端——刚启动时 REP 可能回空
>   分片，客户端会在 `cvtColor(None)` 处崩掉（ZED 时代同理，不是新问题）。
> - **分辨率 / `--use-realsense`**：RealSense 出 640x480（4:3），ZED 时代的
>   客户端会拉伸到 672x384（16:9），画面被压扁。加 `--use-realsense` 后保持
>   640x480 原样发送。**跟着 checkpoint 走**：ZED 图训练的老 ckpt 不加，
>   RealSense 图训练的新 ckpt 加。各客户端行为：
>
>   | 客户端 | 默认 | `--use-realsense` |
>   |---|---|---|
>   | `g1_sonic_client.py` | include-neck 时 resize 672x384 | resize 640x480 |
>   | `g1_sonic_client_rtc.py` | 同上 | 同上 |
>   | `dz_client.py` | 所有帧 resize 672x384 | 所有帧 640x480 |
>   | `psix_rtc_sonic_client.py` | 发原图，服务端处理 | 不需要此 flag |

### 4.4 Replay（重放录制的 token + neck）

把 4.3.2 的「策略客户端」换成 `replay_token.py`，**不需要** policy server 和
端口转发。支持两种录制格式：`data.json` 目录和 LeRobot parquet 目录
（`data/chunk-*/file-*.parquet`，读 `action.token_state` / `action.hand_cmd` /
`action.neck`）。端口、协议、故障排查细节见 [REPLAY_GUIDE.md](REPLAY_GUIDE.md)。

G1 板载同 4.3.1。桌面端环境用 `sonic`（已装 pyarrow）：

```bash
# 1. WBC deploy 同 4.3.2

# 2. 停掉占 5570 的进程（否则 replay 的 neck PUB bind 失败）
pkill -f pico_manus_thread_server.py; pkill -f pose_publisher.py

# 3. 回到该集的起始姿态（直接用第一帧录的 token，不走 encoder；
#    跑完自动退出并释放 5556/5570）
conda activate sonic
PYTHONPATH=$PWD python apply_initial_pose.py --episode-dir data/<episode_dir>

# 4. replay token + neck
PYTHONPATH=$PWD python replay_token.py --episode-dir data/<episode_dir> --enable-neck
```

replay 的同时录制（相机装在脖子上，录下的画面跟着脖子转）：

```bash
source .venv_teleop/bin/activate
python g1_data_server.py \
    --neck-zmq       tcp://localhost:5570 \
    --neck-state-zmq tcp://192.168.123.164:5560
```

键盘 `s` 开始 / `q` 保存 / `d` 丢弃；replay 开始发帧后按 `s`，结束按 `q`。
输出在 `./data/demonstration_<session>/episode_<N>/`（data.json 格式，
之后可直接再 replay）。录制期间看画面一律用 `realsense_viewer.py --sub`，
别用 REQ 模式连 5558 抢帧。

---

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

---

## WiFi 延迟排查(2026-07-09)

**症状**:ping 不丢包,但每隔整 15 秒出现一组阶梯式尖峰(约 3900→2900→1900→900 ms)。

**原因**:本机开着 GNOME"设置-WiFi"面板(`gnome-control-center wifi`)时,
系统每 15 秒强制扫描一次周围网络;单射频 USB 网卡扫描时离开当前信道约 4 秒,
期间所有流量被缓存,表现为不丢包但延迟爆炸。

**解决**:测延迟 / 遥操作前关掉 WiFi 设置窗口:

```bash
pkill gnome-control-center
```

**相关信息**:

- G1 上没有 wlan0,WiFi 网卡是 `wlxfc23cd913281`(rtl8852bu USB)
- 桌面端 USB WiFi 网卡是 `wlx7419f816d8d1`(Realtek 8811CU)
- G1 已存好 `use-psi_5G` 连接(5G 信道 112,已强制 wpa-psk 避开 WPA3 坑),切换:
  `sudo nmcli connection up use-psi_5G`(ssh 会话里 nmcli 必须 sudo,polkit 限制)
- G1 在 use-psi 网段的 IP:192.168.0.8(DHCP,可能变)
- G1 网卡省电模式默认开着,遥操作如有周期性小抖动可关掉:
  `sudo iw dev wlxfc23cd913281 set power_save off`(重启失效)
- 正常参考值:桌面 ↔ G1 全 WiFi 链路(双方都在 use-psi_5G)约 3-8 ms,0 丢包
