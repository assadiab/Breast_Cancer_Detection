FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /work

# Resolve and install the environment from pixi.toml (lock generated on first install)
COPY pixi.toml pixi.lock* ./
RUN pixi install

# Project code
COPY . .

# Default: regenerate the training notebook from its source script
CMD ["pixi", "run", "build-train"]
