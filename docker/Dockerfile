FROM astral/uv:python3.12-trixie-slim

# Sometimes apt-get tries to open a dialog box, this stops it
ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities in single layer with cleanup
RUN apt-get update -y \
    && apt-get install -y build-essential git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG NAME
ENV NDIF_SERVICE=${NAME}

# Copy the entire repo
COPY . /ndif
WORKDIR /ndif

# Install service requirements and the ndif CLI
RUN uv pip install --system -r src/services/${NAME}/requirements.in \
    && pip install . \
    && find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.12 -name "*.pyo" -delete \
    && find /usr/local/lib/python3.12 -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -name "*.a" -delete \
    && rm -rf /root/.cache

# Override pypi nnsight with the local hf-serve worktree that ships
# VanillaBatchServer (cross-request batching for LM deployments). When
# the upstream nnsight release contains VanillaBatchServer, this layer
# can be removed and pypi nnsight pinned in requirements.in instead.
# See Makefile: --build-context nnsight=<host path>.
RUN --mount=type=bind,from=nnsight,target=/tmp/nnsight-src \
    cp -r /tmp/nnsight-src /tmp/nnsight && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NNSIGHT=0.6.3 \
    uv pip install --system /tmp/nnsight && \
    rm -rf /tmp/nnsight

SHELL ["/bin/bash", "-c"]
CMD if [[ -n "${RAY_ADDRESS}" ]]; then \
      ndif start "${NDIF_SERVICE}" --verbose --worker --ray-address "${RAY_ADDRESS}"; \
    else \
      ndif start "${NDIF_SERVICE}" --verbose; \
    fi
