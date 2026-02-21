# ---- Stage 1: build wheels ----
FROM mambaorg/micromamba:1.5.10 AS builder

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    git \
    libode-dev \
    && rm -rf /var/lib/apt/lists/*

USER mambauser
SHELL ["/bin/bash", "-lc"]

# Conda-forge for consistent packages
RUN micromamba config append channels conda-forge && \
    micromamba config set channel_priority strict

# Build env (includes build tools like cmake/ninja)
RUN micromamba create -y -n rsoccer310 -c conda-forge \
    python=3.10 pip cmake ninja && \
    micromamba clean -a -y

# Needed to bypass old CMake policy minimum issues in dependencies
ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# Build wheels (so runtime doesn't need compilers/cmake)
RUN micromamba run -n rsoccer310 python -m pip install -U pip wheel setuptools
RUN mkdir -p /tmp/wheels
RUN mkdir -p /tmp/wheels
RUN micromamba run -n rsoccer310 python -m pip install -U pip wheel setuptools
RUN micromamba run -n rsoccer310 python -m pip wheel --wheel-dir /tmp/wheels rc-robosim
RUN micromamba run -n rsoccer310 python -m pip wheel --wheel-dir /tmp/wheels \
    "git+https://github.com/robocin/rSoccer.git@3adf7c3e89fe6d1b47431ee4099dc5ee89420415#egg=rsoccer-gym"

# ---- Stage 2: runtime ----
FROM mambaorg/micromamba:1.5.10 AS runtime

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    libode8 \
    && rm -rf /var/lib/apt/lists/*
USER mambauser
SHELL ["/bin/bash", "-lc"]

RUN micromamba config append channels conda-forge && \
    micromamba config set channel_priority strict

# Runtime env: only python + pip
RUN micromamba create -y -n rsoccer310 -c conda-forge \
    python=3.10 pip && \
    micromamba clean -a -y

# Install wheels built in builder
COPY --from=builder /tmp/wheels /wheels
RUN micromamba run -n rsoccer310 python -m pip install --no-cache-dir /wheels/*.whl

# Install your runtime Python deps
# (torch is huge; consider cpu-only indexing if you want smaller)
RUN micromamba run -n rsoccer310 python -m pip install --no-cache-dir \
    gymnasium stable-baselines3 \
    && micromamba run -n rsoccer310 python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# Convenience: interactive shells drop into activated env
SHELL ["/bin/bash", "-lc"]
RUN echo 'eval "$(micromamba shell hook -s bash)"' >> ~/.bashrc && \
    echo 'micromamba activate rsoccer310' >> ~/.bashrc

WORKDIR /project
COPY run_soccer.py .

CMD ["bash"]
