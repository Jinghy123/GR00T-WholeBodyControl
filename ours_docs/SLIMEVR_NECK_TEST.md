# SlimeVR 脖子离线预检（不需要 G1）

在 Mac（或任意一台和 SlimeVR 同局域网的机器）上，独立验证 SlimeVR → 脖子
`[yaw, pitch]` 这条计算链 —— **不需要 G1、不需要 GMR 包、不需要 Manus**。脚本只用
`slimevr_adapter` 读 VMC，复用主 server **逐字相同**的脖子公式（`Head` 相对 `Spine3`
的 YXZ 欧拉），把 yaw/pitch 实时打印出来。

脚本：[external_dependencies/slimevr_body/test_slimevr_neck.py](external_dependencies/slimevr_body/test_slimevr_neck.py)

## 为什么先做这个

整条 SlimeVR pipeline 里，只有「脖子从 `Head`/`Spine3` 算角度」这一处和 SlimeVR 数据
+ 坐标系直接相关、存在不确定性。这个预检同时还验证了两个 vendored 里最大的不确定性：
SlimeVR→本机的 VMC 链路通不通、`viewer_fk + bvh` 骨架能不能在自包含环境跑起来。10 分钟
就能消除一个上机风险。

## 1. 装依赖（Mac）

```bash
python3 -m venv ~/slimevr_test_venv
~/slimevr_test_venv/bin/pip install numpy scipy python-osc
```

## 2. Windows 端（SlimeVR）

- 先正常校准（站直做 Full Reset）。
- **确保 tracker 里包含「头部」**（绑头上或指定一个为 Head）——没有头部 tracker 时
  FK 的 `Head` 会回退到 `Neck`，脖子不会动，测不出来。
- 打开 SlimeVR 的 **VMC** 发送（Settings 里找 OSC/VMC），目标 IP 填**本机的局域网 IP**，
  端口 `39539`（与脚本 `VMC_PORT` 一致）。
  - 本机 Mac 局域网 IP 例：`192.168.1.138`（用 `ipconfig getifaddr en0` 查当前值）。

## 3. 跑测试（Mac）

```bash
cd /path/to/GR00T-WholeBodyControl
~/slimevr_test_venv/bin/python external_dependencies/slimevr_body/test_slimevr_neck.py
```

脚本每秒打印 ~10 次 `yaw / pitch`（度），带 `← 左 / 右 → / ↑ 上 / 下 ↓` 箭头和数值范围。

## 4. 动作与预期

| 动作 | yaw / pitch |
|------|-------------|
| 单独**左右转头**（躯干不动） | **yaw 明显变** |
| 单独**上下点头**（躯干不动） | **pitch 明显变** |
| 整个人**原地转身**（头随躯干一起转） | **基本不变** |
| **前倾 / 后仰 / 走动**（头随躯干，不单独转） | **基本不变** |

关键点：脖子算的是 `Head` **相对** `Spine3` 的旋转，所以**头和躯干一起动（刚体整体
转动）不会变**，只有头**相对躯干**转/点头才会变。最后两行「不变」就是在验证
脖子-躯干解耦是否生效。

## 5. 这个预检能 / 不能证明什么

- ✅ 能证明：VMC 链路通、头部 tracker 生效、`viewer_fk+bvh` 跑得起来、**上下左右能动**、
  解耦逻辑正确，且用的是和主 server 完全相同的脖子公式。
- ⚠️ 不能证明：G1 电机最终往哪个方向转 —— Mac 上看到的 yaw/pitch 正负，到电机之间还隔着
  G1 端的 `NECK_YAW_SIGN` / `NECK_PITCH_SIGN`，那个只能上 G1 调（见
  [NECK_TELEOP_GUIDE.md](NECK_TELEOP_GUIDE.md)）。

## 6. 收不到数据排查

- Windows 和 Mac 在**同一个 WiFi/局域网**。
- SlimeVR VMC 目标 IP/端口正确（`<Mac IP>:39539`）。
- Mac 防火墙放行 UDP（系统设置 → 网络 → 防火墙，临时关掉试）。
- 脚本 3 秒没数据会打印 `[warn] 3 秒没收到有效 VMC 帧`。
- 若打印 `缺关节 Head=... Spine3=...`：检查头部 tracker 是否戴上、是否被指定为 Head。

## 配置常量（脚本文件头）

| 常量 | 默认 | 用途 |
|------|------|------|
| `VMC_IP` | `"0.0.0.0"` | 监听所有网卡 |
| `VMC_PORT` | `39539` | 与 SlimeVR VMC 发送端口一致 |
| `VMC_BVH_SCALE` | `0.01` | BVH 全局缩放 |
| `BVH_PATH` | `""` | 留空=用自带 `assets/bvh-recording.bvh` |
| `PRINT_HZ` | `10` | 打印频率 |
