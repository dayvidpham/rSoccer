# rSoccer Docker + Experiments (SAC / DDPG)

This repo provides a Docker image that installs the original **rSoccer (rsoccer-gym)** at a pinned commit and includes a starter workflow for running reinforcement learning experiments (Dribbling, etc.) using **Stable-Baselines3**.

> Important: **rSoccer does not ship a “policy zoo.”** The environments are provided, but you must train (or load) your own policies.

---

## What’s inside

- ✅ Multi-stage Docker build
  - Stage 1 builds wheels for `rc-robosim` and `rsoccer-gym` to avoid compiling in runtime
  - Stage 2 installs the wheels into a minimal micromamba Python runtime
- ✅ Python 3.10 environment (`rsoccer310`)
- ✅ `gymnasium` + `stable-baselines3`
- ✅ Suggested project structure + starter scripts for:
  - environment sanity check
  - training SAC / DDPG
  - evaluation + rendering

---

## System requirements

- Docker (recommended: Docker Desktop on macOS/Windows, Docker Engine on Linux)
- For rendering:
  - **Linux**: X11 forwarding or `--net=host` + display mounting
  - **macOS/Windows**: easiest path is to run **headless training** (no render) and evaluate later with render on a Linux machine, OR run via a VNC setup (optional)

---

## Build the image

From the directory containing the `Dockerfile`:

```bash
docker build -t rsoccer:latest .

## Run the container
Basic shell (recommended first step)
```bash
docker run -it --rm rsoccer:latest

You should drop into a shell with rsoccer310 activated automatically.

To Verify if the environment (rsoccer310) is activated, Do:

```bash
python -c "import rsoccer_gym; import gymnasium as gym; print('OK')"

