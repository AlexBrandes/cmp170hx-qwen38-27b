# Engine image: three pins from github.com/AlexBrandes/vllm (see versions.env).
# Upstream vLLM's prebuilt wheel for the base commit, the fork's vLLM branch
# laid over it (Python files only; the branch changes no native code), and the
# vllm-cmp170hx plugin installed beside it with its EXL3 extension and the
# Qwen3.8-27B kernels compiled here. No weights and no launch settings are baked
# in; start.sh supplies them. Building needs network access and CPU/RAM, not a GPU.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

ARG UPSTREAM_COMMIT
ARG VLLM_WHEEL
ARG VLLM_WHEEL_SHA256
ARG VLLM_BRANCH_COMMIT
ARG VLLM_BRANCH_SHA256
ARG PLUGIN_COMMIT
ARG PLUGIN_SHA256

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev ca-certificates curl ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv==0.9.8

# 1. Upstream vLLM at the base commit, with its pinned dependencies.
RUN curl -fsSL -o /tmp/vllm.whl "${VLLM_WHEEL}" \
    && echo "${VLLM_WHEEL_SHA256}  /tmp/vllm.whl" | sha256sum -c - \
    && mv /tmp/vllm.whl "/tmp/$(basename "${VLLM_WHEEL}" | sed 's/%2B/+/')" \
    && uv pip install --no-cache /tmp/vllm-*.whl \
    && rm -f /tmp/vllm-*.whl

# 2. The vLLM branch (upstream + the fork's commits): its Python files over the
#    installed package. The wheel keeps its compiled extensions and _version.py.
RUN curl -fsSL -o /tmp/vllm-branch.tar.gz \
       "https://codeload.github.com/AlexBrandes/vllm/tar.gz/${VLLM_BRANCH_COMMIT}" \
    && echo "${VLLM_BRANCH_SHA256}  /tmp/vllm-branch.tar.gz" | sha256sum -c - \
    && mkdir /tmp/vllm-branch \
    && tar -xzf /tmp/vllm-branch.tar.gz --strip-components=1 -C /tmp/vllm-branch \
    && SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && cd /tmp/vllm-branch \
    && find vllm -type f -name '*.py' ! -name '_version.py' -exec cp --parents '{}' "$SP/" \; \
    && rm -rf /tmp/vllm-branch /tmp/vllm-branch.tar.gz

# 3. The plugin, with the EXL3 extension built against this image's PyTorch.
RUN curl -fsSL -o /tmp/plugin.tar.gz \
       "https://codeload.github.com/AlexBrandes/vllm/tar.gz/${PLUGIN_COMMIT}" \
    && echo "${PLUGIN_SHA256}  /tmp/plugin.tar.gz" | sha256sum -c - \
    && mkdir /opt/vllm-cmp170hx \
    && tar -xzf /tmp/plugin.tar.gz --strip-components=1 -C /opt/vllm-cmp170hx \
    && rm /tmp/plugin.tar.gz \
    && cd /opt/vllm-cmp170hx \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python csrc/exl3/build_standalone.py --out vllm_cmp170hx \
    && uv pip install --no-cache --no-deps . \
    && python -c "import importlib.metadata as m; \
assert [e.value for e in m.entry_points(group='vllm.general_plugins') if e.name == 'cmp170hx'] == ['vllm_cmp170hx.plugin:register']; \
assert [e.value for e in m.entry_points(group='vllm.platform_plugins') if e.name == 'cmp170hx'] == ['vllm_cmp170hx.plugin:platform_plugin']"

# 4. Qwen3.8-27B kernels from the plugin, built against this image's PyTorch and
#    vLLM wheel: the INT8 prefill GEMM (csrc/exl3/lab_once/cmp_i8mm.cu, CUTLASS
#    headers from the wheel) and the K3 Marlin route dispatcher
#    (csrc/marlin_routes), checked by sha256.
ARG MARLIN_ROUTES_SHA256
RUN SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && TI=$SP/torch/include && CUT=$SP/vllm/third_party/fmha_sm100/cutlass \
    && ABI=$(python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))') \
    && mkdir -p /opt/cmp170hx/q27 \
    && nvcc -O3 -std=c++20 -gencode=arch=compute_80,code=sm_80 --expt-relaxed-constexpr -DNDEBUG \
       -Xcompiler -fPIC -shared -isystem $TI -isystem $TI/torch/csrc/api/include \
       -D_GLIBCXX_USE_CXX11_ABI=$ABI -I$CUT/include -I$CUT/tools/util/include \
       /opt/vllm-cmp170hx/csrc/exl3/lab_once/cmp_i8mm.cu -o /opt/cmp170hx/q27/cmp_i8mm.so \
       -L$SP/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch \
    && bash /opt/vllm-cmp170hx/csrc/marlin_routes/build.sh /out \
    && cp /out/marlin_host.so /out/source-manifest.json /opt/cmp170hx/q27/ \
    && rm -rf /out \
    && echo "${MARLIN_ROUTES_SHA256}  /opt/cmp170hx/q27/marlin_host.so" | sha256sum -c -
ENV VLLM_CMP_MARLIN_ROUTES=/opt/cmp170hx/q27/marlin_host.so \
    VLLM_CMP_MARLIN_ROUTES_SHA256=${MARLIN_ROUTES_SHA256} \
    VLLM_CMP_I8MM_LIB=/opt/cmp170hx/q27/cmp_i8mm.so

LABEL org.opencontainers.image.source="https://github.com/AlexBrandes/cmp170hx-qwen38-27b" \
      org.opencontainers.image.description="vLLM (upstream ${UPSTREAM_COMMIT}) + AlexBrandes/vllm branch and vllm-cmp170hx plugin for Qwen3.8-27B on one CMP 170HX" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${PLUGIN_COMMIT}" \
      cmp170hx.upstream_commit="${UPSTREAM_COMMIT}" \
      cmp170hx.vllm_commit="${VLLM_BRANCH_COMMIT}" \
      cmp170hx.plugin_commit="${PLUGIN_COMMIT}"
ENTRYPOINT []
