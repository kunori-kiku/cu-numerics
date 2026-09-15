# cu-numerics: general third-party CUDA numerics and native build environment.
# Build ONLY with the text-only context made by prepare_runpod_runtime.py.
ARG RUNPOD_BASE=runpod/base:1.2.0-cuda1290-ubuntu2204
FROM ${RUNPOD_BASE}

ARG MINIFORGE_SHA256=65af53dad30b3fcbd1cb1d4ad62fd3a86221464754844544558aae3a28795189
COPY conda-linux-64.lock requirements.lock /tmp/cu-numerics-locks/
RUN curl --fail --location --retry 3 \
      https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh \
      --output /tmp/miniforge.sh && \
    echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum --check --strict - && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    CONDA_OVERRIDE_CUDA=12.9 /opt/conda/bin/conda create --yes --prefix /opt/cu-numerics --file /tmp/cu-numerics-locks/conda-linux-64.lock && \
    /opt/cu-numerics/bin/python -m pip install --index-url https://pypi.org/simple \
      --no-cache-dir --only-binary=:all: --no-deps --require-hashes -r /tmp/cu-numerics-locks/requirements.lock && \
    /opt/conda/bin/conda clean --all --yes && rm /tmp/miniforge.sh

ENV CU_NUMERICS_PREFIX=/opt/cu-numerics
ENV CU_NUMERICS_PYTHON=/opt/cu-numerics/bin/python

# Leave RunPod's PATH, entrypoint, command and SSH startup intact. Applications
# use CU_NUMERICS_PYTHON; the environment also contains nvcc, CMake and Ninja.
RUN JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/opt/cu-numerics/lib:/usr/local/cuda/lib64 \
    /opt/cu-numerics/bin/python - <<'PYTHON'
import hashlib, importlib.metadata, json, pathlib, shutil
import jax, jaxlib, numpy, scipy, optax
prefix=pathlib.Path('/opt/cu-numerics')
assert (jax.__version__,jaxlib.__version__,numpy.__version__,scipy.__version__,optax.__version__)==('0.9.2','0.9.2','2.4.4','1.17.1','0.2.3')
for name in ('nvcc','cmake','ninja','c++'):
    assert (prefix/'bin'/name).is_file(), name
assert (prefix/'include/cudss.h').is_file()
assert (prefix/'lib/libcudss.so.0').is_file()
files={}
for path in prefix.rglob('*'):
    if path.is_file() and ('.so' in path.name or path.suffix in ('.h','.hpp','.cuh') or path.name in ('python3.11','nvcc','cmake','ninja','c++','cudss.h')):
        with path.open('rb') as stream:files[str(path)]=hashlib.file_digest(stream,'sha256').hexdigest()
record=dict(schema='cu_numerics_runtime_v1',prefix=str(prefix),cuda_family='12.9',
    cudss_library=str(prefix/'lib/libcudss.so.0'),files=files,
    versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','scipy','flax','optax')},
    custom_project_code_included=False)
(prefix/'runtime-manifest.json').write_text(json.dumps(record,sort_keys=True,indent=2))
print('cu-numerics dependencies and toolchain ready; custom extensions build privately after checkout')
PYTHON
