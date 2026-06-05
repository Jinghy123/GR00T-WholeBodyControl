"""
test_slimevr_neck.py — 在 Mac 上独立验证 SlimeVR → 脖子 [yaw, pitch]。

不需要 G1 / GMR 包 / Manus。只用 slimevr_adapter 读 VMC，复用主 server 完全相同的
脖子公式（Head 相对 Spine3 的 YXZ 欧拉），把 yaw/pitch 实时打印出来，方便确认：
  - VMC 链路通不通、Head/Spine3 有没有数据（头部 tracker 是否生效）
  - 转头 → yaw 变；点头 → pitch 变（上下左右有没有动）
  - 身体前倾/后仰（头不转）→ yaw/pitch 基本不变（脖子-躯干解耦是否生效）

依赖（Mac）：  pip install numpy scipy python-osc
Windows 侧：   SlimeVR app 打开 VMC/OSC 发送，目标=本 Mac 的局域网 IP，端口 = VMC_PORT。

运行：         python test_slimevr_neck.py
"""

import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from slimevr_adapter import SlimevrBodyConfig, SlimevrBodyReader  # noqa: E402


# ============================================================================
# CONFIG —— 和主 server 的 VMC 段保持一致即可
# ============================================================================
VMC_IP        = "0.0.0.0"   # 监听所有网卡；Windows 那边填 Mac 的局域网 IP
VMC_PORT      = 39539       # 必须和 SlimeVR app 里 VMC 发送端口一致
VMC_BVH_SCALE = 0.01
BVH_PATH      = ""          # 留空=用自带 assets/bvh-recording.bvh
PRINT_HZ      = 10          # 打印频率
# 脖子用哪两个关节算“头相对躯干”。标准是 Head/Spine3；但 SlimeVR 无 HMD 时
# 头部 tracker 的数据落在 Neck/Spine3 段（C 段），躯干在 Spine/Spine1/Spine2 段
# （B 段），所以这里用 Neck(C) 相对 Spine1(B)。若以后用 HMD 提供真 Head，
# 改回 NECK_HEAD_JOINT="Head", NECK_BASE_JOINT="Spine3"。
NECK_HEAD_JOINT = "Neck"    # 头数据所在关节（C 段）
NECK_BASE_JOINT = "Spine1"  # 躯干参考关节（B 段）
# ============================================================================


def neck_yaw_pitch_deg(body_frame):
    """与 slimevr_manus_thread_server._human_head_to_robot_neck 完全一致的公式，
    只是输出改成角度。返回 (yaw_deg, pitch_deg) 或 None。"""
    head = body_frame.get(NECK_HEAD_JOINT)
    spine = body_frame.get(NECK_BASE_JOINT)
    if head is None or spine is None:
        return None

    def _R(joint):
        q = np.asarray(joint[1], dtype=np.float64).reshape(4)  # wxyz
        return Rotation.from_quat([q[1], q[2], q[3], q[0]])

    R_rel = _R(spine).inv() * _R(head)
    yaw, pitch, _roll = R_rel.as_euler("yxz", degrees=True)
    return float(yaw), float(pitch)


def _arrow(yaw, pitch, thr=5.0):
    lr = "← 左" if yaw > thr else ("右 →" if yaw < -thr else "  ·  ")
    ud = "↑ 上" if pitch > thr else ("下 ↓" if pitch < -thr else "  ·  ")
    return lr, ud


def main():
    bvh = BVH_PATH or os.path.join(_HERE, "assets", "bvh-recording.bvh")
    if not os.path.exists(bvh):
        print(f"[error] 找不到 BVH 骨架: {bvh}")
        sys.exit(1)

    cfg = SlimevrBodyConfig(
        vmc_ip=VMC_IP,
        vmc_port=VMC_PORT,
        vmc_timeout_s=0.5,
        vmc_rot_mode="local",
        vmc_use_fk=True,
        vmc_use_viewer_fk=True,
        vmc_fk_skeleton="bvh",
        vmc_bvh_path=bvh,
        vmc_bvh_scale=VMC_BVH_SCALE,
    )
    reader = SlimevrBodyReader(cfg)
    reader.initialize()

    print(f"监听 VMC: {VMC_IP}:{VMC_PORT}   bvh={bvh}")
    print("在 Windows 的 SlimeVR 里打开 VMC/OSC 发送，目标填本 Mac 的局域网 IP，"
          f"端口 {VMC_PORT}。")
    print("（若收不到数据：确认同一局域网、Mac 防火墙放行 UDP，端口一致。）")
    print("-" * 64)
    print("收到数据后，依次做这些动作观察：")
    print("  【会变】头相对躯干转：")
    print("    1) 单独左右转头（躯干不动）  → yaw 变")
    print("    2) 单独上下点头（躯干不动）  → pitch 变")
    print("  【不变】头和躯干一起动、没有相对旋转：")
    print("    3) 整个人原地转身（头随躯干一起转） → 基本不变")
    print("    4) 前倾/后仰、走动（头随躯干）       → 基本不变")
    print("Ctrl-C 退出。注意：这里验证的是“数据+算法对、上下左右能动”；")
    print("电机最终转向还要到 G1 端用 NECK_YAW_SIGN / NECK_PITCH_SIGN 调。")
    print("-" * 64)

    warned_missing = False
    warned_nodata = False
    warned_zero = False
    last_print = 0.0
    last_diag = 0.0
    start_t = time.time()
    last_data = time.time()
    yaw_min = pitch_min = 1e9
    yaw_max = pitch_max = -1e9

    try:
        while True:
            res = reader.read_frame()
            if not res.get("ok"):
                if time.time() - last_data > 3.0 and not warned_nodata:
                    print("[warn] 3 秒没收到有效 VMC 帧 —— 检查 SlimeVR VMC 发送 / IP / 端口 / 防火墙")
                    warned_nodata = True
                time.sleep(0.01)
                continue
            last_data = time.time()
            warned_nodata = False

            body_frame = res["body_frame"]

            # 诊断：每 2 秒打印关键关节的原始四元数（wxyz），用来区分：
            #   - RightHand 一直 [1,0,0,0] 不变 → 完全没收到 VMC
            #       （多半 UDP 39539 被正在跑的主 server 占了；测脖子要先停主 server）
            #   - Head 和 Spine3 几乎相等 / Head 不随转头变 → 没有独立头部 tracker
            #       （Head 回退到躯干，脖子自然恒 0）
            #   - Head 随转头明显变、且和 Spine3 不同，但 yaw/pitch 仍 0 → 再找我
            tnow = time.time()
            if tnow - last_diag > 2.0:
                def _q(name):
                    v = body_frame.get(name)
                    return None if v is None else np.round(
                        np.asarray(v[1], dtype=float).reshape(4), 3).tolist()
                # 沿脊柱从下到上看 quat 在哪一级开始“塌缩”成相同值 ——
                # 那一级以上就是没有 tracker、被 fallback 的部分（典型是缺头部来源）。
                print("[diag] 脊柱链 quat(wxyz)：")
                for nm in ("Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"):
                    print(f"    {nm:8s}= {_q(nm)}")
                print(f"    RightHand= {_q('RightHand')}  (参考：身体在动则它会变)")
                last_diag = tnow

            yp = neck_yaw_pitch_deg(body_frame)
            if yp is None:
                if not warned_missing:
                    has_head = "Head" in body_frame
                    has_spine = "Spine3" in body_frame
                    print(f"[warn] 缺关节 Head={has_head} Spine3={has_spine} "
                          "—— 没有头部 tracker 时 Head 会回退、脖子不会动")
                    warned_missing = True
                continue

            yaw, pitch = yp
            yaw_min, yaw_max = min(yaw_min, yaw), max(yaw_max, yaw)
            pitch_min, pitch_max = min(pitch_min, pitch), max(pitch_max, pitch)

            now = time.time()
            if now - last_print >= (1.0 / max(1, PRINT_HZ)):
                lr, ud = _arrow(yaw, pitch)
                print(
                    f"yaw={yaw:+6.1f}° [{lr}]   pitch={pitch:+6.1f}° [{ud}]   "
                    f"(range yaw[{yaw_min:+.0f},{yaw_max:+.0f}] "
                    f"pitch[{pitch_min:+.0f},{pitch_max:+.0f}])"
                )
                last_print = now

            # viewer_fk 在没收到 VMC 数据时输出零姿态 → yaw/pitch 恒为 0，
            # 此时 read_frame 仍返回 ok，所以这里用“一直为 0”来提示可能没连上。
            if (not warned_zero) and (now - start_t > 3.0) and \
               abs(yaw_min) < 1e-6 and abs(yaw_max) < 1e-6 and \
               abs(pitch_min) < 1e-6 and abs(pitch_max) < 1e-6:
                print("[提示] yaw/pitch 一直为 0 —— 多半 VMC 还没连上"
                      "（viewer_fk 收不到数据时输出零姿态）。确认 SlimeVR 的 VMC 已指向本机，"
                      "再转头看数值是否变化。")
                warned_zero = True
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        try:
            reader.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
