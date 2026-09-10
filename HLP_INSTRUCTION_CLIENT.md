# PSIX instruction client

指令从哪来，决定跑哪一种。三种都需要 VLA (8014) 和 WM (192.168.123.240:8016)。

| 模式 | 指令从哪来 | 要 HLP server |
|---|---|---|
| **A 手动** | Enter 依次走 `--next-prompt` 列表 | 否 |
| **B HLP 整句** | HLP 出整句 instruction | 是 |
| **C HLP + atomic** | 同 B，清洁/饮料任务再拆成 atomic 步骤 | 是 |

三个 launcher 只是不同的默认参数，入口都是 `psix_client.py`。命令行追加的同名参数覆盖脚本值。

---

## A 手动

不用 HLP server，不用转发。

```bash
./run_psix_hlpwm_client.sh --check-only    # 只查 VLA /info + WM /ready
./run_psix_hlpwm_client.sh --real
```

指令序列硬编码在脚本第 9–13 行（1 条 `--prompt` + 4 条 `--next-prompt`），要改直接编辑。

---

## B / C：先起 HLP server

**远端一个终端**

```bash
ssh nebula102-jionghao
cd /hfm/jwang/code/psi

ls -1dt .runs/psix_hlp_finetune/*/          # 训练 run，最新在最上
ls -1dt .runs/psix_hlp_serve/export/*/      # 已导出的 serving 目录
```

导出目录全都叫 `..._ckpt6000`，**step 号区分不了版本，只能看训练 run 的时间戳**。训练 run 比所有
export 都新，说明最新 ckpt 还没导出：

```bash
bash scripts/deploy/export_hlp_ckpt.sh <run_dir> <step>
```

然后起：

```bash
BUNDLE=<export 目录>
env PSI_VENV="$PWD/.venv-vllm" BACKEND=vllm \
    PSIX_HLP_VLLM_TENSOR_PARALLEL_SIZE=1 PSIX_HLP_VLLM_KV_CACHE_GIB=2 \
    CUDA_VISIBLE_DEVICES=4 PORT=8015 HOST=0.0.0.0 \
    bash scripts/deploy/serve_hlp.sh "$BUNDLE" <step>
```

等日志出 `READY`。`CUDA_VISIBLE_DEVICES` 挑一张空卡（`nvidia-smi` 看）。HF backend 是
`BACKEND=hf` + `PSI_VENV="$PWD/.venv-psi"`，但 A100 上 HF 一次推理 1.8s，撑不住 1 Hz 轮询。

**本机一个终端转发 + 确认**

```bash
ssh -N -o ServerAliveInterval=15 -L 8015:127.0.0.1:8015 nebula102-jionghao
curl -s http://127.0.0.1:8015/health | python -m json.tool
```

`checkpoint.path` 是确认服务的到底是哪个 ckpt 的唯一可靠依据。`hierarchical` 必须是 `true` 才能跑 C。

---

## B HLP 整句

```bash
./run_psix_hlp_instruction_client.sh --check-only
./run_psix_hlp_instruction_client.sh --real
```

## C HLP + atomic

```bash
./run_psix_hlp_atomic_client.sh --check-only
./run_psix_hlp_atomic_client.sh --real
```

分不分 atomic 只是 `/reset` 里的一个字段，**server 不用改**——同一个 server 进程，B 和 C 换个
launcher 重启客户端就切。

---

## 通用

`--check-only` 依次验 VLA `/info` 的 `rtc_mode` 是否等于 `--rtc-mode`、WM `/ready`、以及 B/C 下的
HLP `/health`，任一不符直接退出，不碰机器人。不加 `--real` 是 dry-run：全流程跑通但不发机器人命令。

**按键**

| 键 | 行为 |
|---|---|
| Enter | A：下一条 prompt；B/C：解除暂停，done 时跳过当前任务继续规划 |
| `p` | 退回上一条已执行的指令并暂停（Enter 继续）。无需回车 |
| `i` | 输入完整 instruction，Enter 应用，Esc 取消 |
| `q` | 退出并 flush recorder |
| `:sec 2.4` | 改 WM future 时长 |

**常用参数**（`--help-all` 看全部）

| 参数 | 含义 |
|---|---|
| `--prompt` | A 的初始指令，或 B/C 的 macro 文本/预设 key（`--list-prompts` 列出 4 个预设） |
| `--rtc-mode` | `train` / `test_time` / `off`（HTTP 分块）。**必须与 VLA `/info` 一致** |
| `--encoder-version` | `v1` / `v1_1`。**必须与 VLA token 和 WBC decoder 一致** |
| `--wm-period` / `--wm-seconds` | WM 请求间隔 / future 预测时长 |
| `--hlp-period` | HLP 轮询间隔，默认 1.0s |
| `--no-show-goal` | 关掉本地 goal 图窗口（三个 launcher 默认开） |

**录制**：默认写到 `~/Desktop/psi/.logs/psix_rollouts/<timestamp>/`，`--output-dir` 可改。
`requests/hlp/` 和 `requests/wm/` 存每次实际发送的 JPEG + 完整请求/回复（离线回放靠这个），
`rollout/` 存连续 states / 已发布动作 / ego video，`events.jsonl` 存指令变化和 hold。
`--no-record` 只关这两类，日志和 manifest 仍保留。

---

## 需要知道的三件事

**1. HLP 的回复只有一句话是给机器人的。** Server 拥有 instruction 序列和 memory，client 只执行。
回复里 `instruction` 是唯一送给 WM 的文本（`null` = planner 没有可执行指令，机器人保持 frozen
pose），`parent` 是 VLA 的 `Task:`（atomic 层时两者不同——VLA 训练时 `Task:` 用的是整句，atomic
步骤只出现在可选的 `Subtask:` 从句里）。`revision` 变了就重建 WM goal 和 VLA condition。

**2. 每次换指令机器人都会先停。** 切指令 → hold（发实测姿态，WBC 继续跑）→ 等新 WM goal 落地 →
恢复。约一个 `--wm-period`。atomic 模式下每一步都付这个代价。

**3. 动作放行只看一件事**：它是不是为当前生效的那条指令算出来的。被拒时打一行
`[client] dropped action version=N: <reason>`——只有一个理由要查。

## atomic 模式的已知缺口

退出 atomic 层**只有**「模型在最后一步报 done」这一条路，没有超时兜底。

两个真实 rollout 离线回放（`_v2` ckpt）都卡在同一处：`Turn right and walk three steps` →
`Turn left and walk towards the kitchen stand`。这两个连续导航动作首尾相接，画面上是一段不间断
行走、没有视觉事件标记分界，模型整段判自己还在当前步，持续报出正确的下一步却从不给 `switch`
（442 次查询里只有 2 次，卡 347s；另一个 rollout 同处卡 189s）。

**同一个 rollout、同一个 ckpt 的倒水链 7 步全部按时切换**，每步 4–20 秒，其中 `approach_cup`
也是导航步、15 秒正常切过去——因为它前面是抓取、后面是放置，边界两侧都有离散事件。

**所以修在数据里，不在 planner**：那两个清洁步骤本质是一个动作，合并即可。漏掉一次边界之后不会
自己恢复——memory 冻在 planner 以为的那一步，画面继续往前跑，planner 没有机制把 memory 重新锚回
画面。

## 代码

```text
psix_client.py         参数、契约检查、组件组装与生命周期
psix/instructions.py   唯一的指令状态；HLP 轮询与操作员命令
psix/hlp.py            HLP HTTP client 与 macro prompts
psix/wm.py             WM 请求、goal 状态、prompt epoch
psix/vla.py            共用的 _observe() tick、hold/发布、RTC 与 HTTP 传输
psix/telemetry.py      每条动作的诊断；这里任何东西都停不了机器人
psix/robot.py          相机、机器人 I/O、body encoder、wire 转换
psix/{keyboard,recording}.py   单键输入 / 请求记录、事件、manifest
rollout_recorder.py    唯一的连续记录器
```

server 端在 `nebula102-jionghao:/hfm/jwang/code/psi`：`serve_psix_hlp.py`（路由 + checkpoint +
prompt 渲染）、`hlp_hierarchy.py`（planner 状态机）、`hlp_task_registry.py`（两个可分解任务的固定
步骤序列）。结构图见 [psix_hlpwm_client_explained](https://claude.ai/code/artifact/fbf6377e-03b2-4415-8012-dca1baf12848)。

离线回归：

```bash
python -m unittest test_psix_client_launch test_psix_instructions \
    test_psix_rtc_encoder test_psix_wm_requests test_psix_recording
```

不连真实机器人或模型服务；实机行为仍需实际 rollout 验证。
