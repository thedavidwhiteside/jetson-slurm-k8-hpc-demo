# jetson-slurm-k8-hpc-demo

Ansible automation for provisioning NVIDIA Jetson devices for local inference using Kubernetes (k3), vLLM, and an nginx proxy.

## Prerequisites

- Ansible installed on your control machine
- SSH access to target Jetson device(s)
- Sudo privileges on the target(s)

## Configuration

### SSH config (recommended)

Define host aliases in `~/.ssh/config` so IPs stay off the repo:

```
Host jetson1
    HostName 192.168.1.100
    User <your-user>

Host login1
    HostName 192.168.1.101
    User <your-user>
```

### Inventory

`inventory.ini` references those aliases — no IPs needed:

```ini
[jetsons]
jetson1

[logins]
login1
```

## Usage

```bash
ansible-playbook jetson.yaml -b -K
```

## What it does

- Updates the apt package index
- Upgrades all installed packages
- Installs the full NVIDIA JetPack meta-package
- Reboots the device if JetPack or system packages were updated

# Set the HF token
  kubectl create secret generic hf-token --from-literal=token=hf_xxxx

# Test

  curl -k https://jetson1/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
      "messages": [{"role": "user", "content": "What is 2+2?"}],
      "max_tokens": 100
    }'