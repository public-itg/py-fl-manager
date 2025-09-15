# Arguments to handle dependencies
ARG STAGE="dev"
ARG GROUP_DEPENDENCIES
ARG EXTRA_DEPENDENCIES
# Path arguments, installing project in `opt/project` like PyCharm
# Setting the virtual environment to /opt/.venv to avoid overwrite when mounting volume
ARG BASE_DIR="/opt"
ARG APP_HOME="$BASE_DIR/project"
ARG VIRTUAL_ENV="$BASE_DIR/.venv"
# Base images arguments
ARG DEBIAN_CODENAME="bookworm"
ARG PYTHON_VERSION=3.11
ARG BUILD_BASE="python:$PYTHON_VERSION-$DEBIAN_CODENAME"
ARG RUNTIME_BASE="python:$PYTHON_VERSION-slim-$DEBIAN_CODENAME"

# We have to create this alias stage to be able to mount uv in build stage
ARG UV_VERSION=0.8.2
ARG UV_BASE="ghcr.io/astral-sh/uv:${UV_VERSION}"
FROM $UV_BASE AS uv

FROM $BUILD_BASE AS dev-builder
ENV UV_ARGS=""

FROM $BUILD_BASE AS pro-builder
ENV UV_ARGS="--no-dev --no-editable"

FROM ${STAGE}-builder AS builder
ARG GROUP_DEPENDENCIES
ARG EXTRA_DEPENDENCIES
ARG APP_HOME
ARG VIRTUAL_ENV

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_REQUEST_TIMEOUT=1000 \
    UV_PROJECT_ENVIRONMENT=${VIRTUAL_ENV} \
    PATH=/root/.local/bin:${VIRTUAL_ENV}/bin:$PATH

WORKDIR $APP_HOME
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv venv --seed && \
    uv sync --frozen --no-install-project --no-install-workspace --no-editable ${GROUP_DEPENDENCIES} ${EXTRA_DEPENDENCIES}

RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=packages,target=packages \
    uv sync --frozen ${UV_ARGS} ${GROUP_DEPENDENCIES} ${EXTRA_DEPENDENCIES}


FROM $RUNTIME_BASE AS runtime-base
ARG PYTHON_VERSION
ARG BASE_DIR
ARG VIRTUAL_ENV

COPY --from=builder --chmod=777 $VIRTUAL_ENV $VIRTUAL_ENV
ENV HAMILTON_TELEMETRY_ENABLED=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=${VIRTUAL_ENV}/bin:${PATH} \
    _TORCH_LIB=${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/torch/lib \
    _LIB_CUDA=${BASE_DIR}/.local/lib/cuda

ENV LD_LIBRARY_PATH=${_TORCH_LIB}:${_LIB_CUDA}

RUN if [ -d ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia ]; then \
      mkdir -p $_LIB_CUDA && cd $_LIB_CUDA && \
      for file in $(find ${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia -path "*/lib/*" -name "*.so*"); do \
          ln -s $file ./; \
      done \
    fi

FROM runtime-base AS dev-runtime
ENV UNIX_PKGS="neovim"

FROM runtime-base AS pro-runtime
ENV UNIX_PKGS=""

FROM ${STAGE}-runtime AS runtime
ARG APP_HOME
WORKDIR $APP_HOME

RUN --mount=type=bind,source=devops/additional_packages.txt,target=additional_packages.txt \
  apt-get update && \
  apt-get --yes --no-install-recommends install \
    $(cat additional_packages.txt | grep -v "#") \
    ${UNIX_PKGS} && \
  apt-get clean && \
  rm -rf /tmp/* /var/cache/* /var/lib/apt/lists/*
