FROM astral/uv:python3.12-trixie-slim

# Sometimes apt-get tries to open a dialog box, this stops it
ENV DEBIAN_FRONTEND=noninteractive

# Install base utilities in single layer with cleanup
RUN apt-get update -y \
    && apt-get install -y build-essential git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG NAME
COPY src/services/${NAME}/requirements.in /tmp/requirements.in
RUN uv pip install --system -r /tmp/requirements.in \
    && find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.12 -name "*.pyo" -delete \
    && find /usr/local/lib/python3.12 -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -name "*.a" -delete \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache

# Install nnsight from the local source tree bind-mounted via BuildKit
# named build context (see Makefile: --build-context nnsight=<host path>).
# Using the local tree keeps us on unreleased branches (e.g. hf-serve-work)
# without requiring the branch to be pushed to a remote.
RUN --mount=type=bind,from=nnsight,target=/tmp/nnsight-src \
    cp -r /tmp/nnsight-src /tmp/nnsight && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NNSIGHT=0.6.3 \
    uv pip install --system /tmp/nnsight && \
    rm -rf /tmp/nnsight

# --mount=type=bind` temporarily mounts content into a `RUN` command
# **without adding it as a layer** to the image.
# cp -L copies symlinks in: *dereferences* them so they are copied
RUN --mount=type=bind,source=/src,target=/tmp/src \
    cp -rL /tmp/src/services/${NAME}/src / && \
    cp /tmp/src/services/${NAME}/start.sh ./start.sh

SHELL ["/bin/bash", "-c"]
CMD bash ./start.sh
