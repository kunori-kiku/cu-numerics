# cu-numerics

A general CUDA 12.9 numerical-computing and native-extension build environment.
It inherits RunPod base 1.2.0 on Ubuntu 22.04, including its SSH startup.

The image installs public third-party packages from exact conda-forge package
URLs/checksums and hashed PyPI wheel requirements. It includes Python 3.11,
JAX/JAXLIB 0.9.2, NumPy 2.4.4, SciPy 1.17.1, NVIDIA cuDSS 0.8.0.10 with headers,
CUDA math libraries, nvcc, CMake, Ninja, C++ compilers and scientific Python tools.
The locks preserve the supplier build variants as well as package versions.

Use `/opt/cu-numerics/bin/python` or `$CU_NUMERICS_PYTHON`. Build tools and headers
are in `/opt/cu-numerics`; applications can prepend its `bin` directory to PATH
and `lib` directory to LD_LIBRARY_PATH. The image leaves RunPod's global PATH,
SSH entrypoint and command intact. A runtime dependency inventory is generated
at `/opt/cu-numerics/runtime-manifest.json`.

Applications, project-specific native extensions, private source, training data,
models and checkpoints are not included. Build your own extensions privately
against the installed libraries; NVIDIA cuDSS is installed as a prebuilt supplier
package. GPU driver compatibility remains a property of the deployment host.

The build context consists only of Dockerfile, conda-linux-64.lock,
requirements.lock and .dockerignore. No existing environment or image archive
is copied into the image. Third-party packages retain their respective licenses.

The workflow builds, checks dependency imports and the inherited SSH settings,
then publishes `ghcr.io/kunori-kiku/cu-numerics:cuda12.9`. Publication is not an
anonymous-access claim until the final unauthenticated registry check succeeds.
