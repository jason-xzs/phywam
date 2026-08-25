nvidia-smi
conda activate worldmodel_phy
cd /data/worldmodel_xzs/phywam_v3

## i2va模式
cd /data/worldmodel_xzs/phywam_v3/
CUDA_VISIBLE_DEVICES=7 NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh

# 服务器端结果默认保存在“/data/worldmodel_xzs/phywam_v3/train_out/real、/data/worldmodel_xzs/phywam_v3/visualization”，客户端在/data/worldmodel_xzs/RoboTwin/results、/data/worldmodel_xzs/RoboTwin/eval_result
# Prompt在/data/worldmodel_xzs/phywam_v3/wan_va/configs/va_robotwin_i2va.py

## robotwin模式
# 终端1
CUDA_VISIBLE_DEVICES=3 NGPU=1 bash evaluation/robotwin/launch_server.sh

# 终端2
source /data/worldmodel_xzs/setup_vulkan.sh

CUDA_VISIBLE_DEVICES=4 bash evaluation/robotwin/launch_client.sh ./results dump_bin_bigbin


## 多GPU
# 终端1
bash evaluation/robotwin/launch_server_multigpus.sh

# 终端2
source /data/worldmodel_xzs/setup_vulkan.sh
bash evaluation/robotwin/launch_client_multigpus.sh ./results 0


## 换端口，有错误
# CUDA_VISIBLE_DEVICES=1 START_PORT=29057 MASTER_PORT=29062 NGPU=1 bash evaluation/robotwin/launch_server.sh
# CUDA_VISIBLE_DEVICES=1 PORT=29057 bash evaluation/robotwin/launch_client.sh ./results turn_switch


## 查看进程详情
#  ps -o pid,etime,cmd -p 2202423

##关于git

# git config user.name "jason-xzs"
# git config user.email "1661825351@qq.com"
# git commit -m "create base lingbot-va"
# vim .gitignore
# rm -rf WISA/.git
# git add .
# git commit -m "create base lingbot-va"
# git worktree add ../lingbot-wisa -b wisa base

# git remote rename origin upstream
# git remote add origin https://github.com/jason-xzs/lingbot-va.git
# git remote -v
# git checkout mope
# ssh -T git@github.com
# git add -A
# git commit -m "mope: initial upload"
# git push -u origin mope


# ssh-keygen -t ed25519 -C "1661825351@qq.com" -f ~/.ssh/id_ed25519_jason
# eval "$(ssh-agent -s)"
# ssh-add ~/.ssh/id_ed25519_jason
# ssh-add -l
# cat ~/.ssh/id_ed25519_jason.pub
# chmod 700 ~/.ssh
# git remote set-url origin git@github-jason:jason-xzs/lingbot-va.git
# git push -u origin base


## 关于频率与时间
# 将真实观测帧编码并写入KV cache的一次完整前向耗时，不包含去噪推理：约1.5s
# 一次推理时间：约6.5s
# va_robotwin_cfg.frame_chunk_size = 2      # 每次推理生成2帧（VAE latent帧）视频
# va_robotwin_cfg.action_per_frame = 16     # 每帧对应16步动作

# VAE时间压缩比：4
# 1 VAE latent帧 = 4 真实帧
# 保存帧率：fps=10（i2va模式）


# 动作执行：50 Hz
# 视频采样：12.5 Hz
# 客户端输出每帧视频对应 50/12.5 = 4 个动作时间步（每四步采集一次观测）
# 保存的对比视频以 15 fps 编码

## 关于cuda12.4

# 设置环境
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
# export TMPDIR=/home/nvme03/tmp
# export PIP_CACHE_DIR=/home/nvme03/pip-cache
# mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# 安装torch2.6.0+cuda12.4
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 重新生成不含四个包的依赖清单（跳过 torch 三件套 + flash_attn）
# grep -Ev "^(torch|torchvision|torchaudio|flash_attn)(==|$)" requirements.txt > /tmp/lingbot-va-req-main.txt

# 安装主依赖
# pip install -r /tmp/lingbot-va-req-main.txt

# 单独安装 flash_attn（关键参数）
# pip install flash-attn --no-build-isolation
