export HF_ENDPOINT=https://hf-mirror.com

sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.1panel.live",
    "https://dockerpull.org"
  ]
}
EOF
sudo systemctl restart docker

export KEEP_IMAGES_ABOVE_GB=30

sudo sysctl -w kernel.perf_event_paranoid=-1

export WEB_SEARCH_PROVIDER=tavily

export TASK_CONTAINER_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

sudo chmod +x scripts/setup/*.sh