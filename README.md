# jetson-slurm-k8-hpc-demo

Ansible automation for provisioning NVIDIA Jetson devices for local inference using Kubernetes or as SLURM nodes for model fine-tuning.

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
