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

## RunPod template and SSH

The published image retains the NVIDIA entrypoint and RunPod's `/start.sh`.
That startup script configures and starts SSH only when `PUBLIC_KEY` is nonempty.
For a personal RunPod template, use these settings:

| Field | Setting |
| --- | --- |
| Container image | `ghcr.io/kunori-kiku/cu-numerics:cuda12.9` |
| Container Start Command | Leave blank to inherit `/start.sh` when `PUBLIC_KEY` is configured below. |
| Environment variable `PUBLIC_KEY` | Complete authorized public keys, one key per line. |
| Expose TCP Ports | `22` |
| Instance networking | A public-IP-capable instance for direct SSH and file transfer. |

The image's pinned digest is
`sha256:0ce720e58149ee45938b517df406dbb0ba5486575cc5e7c8a8a2e98cd2dcec2c`.
Use `ghcr.io/kunori-kiku/cu-numerics@sha256:0ce720e58149ee45938b517df406dbb0ba5486575cc5e7c8a8a2e98cd2dcec2c`
to select those exact bytes.

RunPod offers two separate connection routes. Register the connecting key under
account **SSH Public Keys** for the `ssh.runpod.io` basic gateway. Its documented
per-Pod override is `SSH_PUBLIC_KEY`; changing the image's `PUBLIC_KEY` alone does
not register a gateway key. For direct SSH/SCP, use **SSH over exposed TCP** and
its assigned external port: `ssh root@PUBLIC_IP -p EXTERNAL_PORT -i PRIVATE_KEY`.
The external port is normally different from container port 22. The basic gateway
does not support SCP/SFTP. See the [official SSH guide](https://docs.runpod.io/pods/configuration/use-ssh).

Keep the inherited startup when customizing the command. The
[template guide](https://docs.runpod.io/pods/templates/manage-templates) explains
that the start-command field overrides the image's CMD. No SSH package
installation at Pod startup is needed for this image.

The image contains about 16.1 GB of compressed layers and expands to about
31.1 GB. A cold pull can delay SSH readiness; inspect Pod initialization and
container logs before treating an early connection refusal as authentication
failure. Attach a **network volume** at creation for data that must survive Pod
termination; an ordinary Pod volume disk does not meet that requirement.
See [RunPod storage types](https://docs.runpod.io/pods/storage/types).

The [2026-09-15 SSH smoke run](https://github.com/kunori-kiku/cu-numerics/actions/runs/34960034712)
passed real RSA4096 and Ed25519 logins and an SCP transfer on the pinned image.
It exercised the inherited entrypoint and `/start.sh`, with an override that
combined two temporary public keys before invoking `/start.sh`, through a
forwarded port on a GitHub CPU runner. It did not test RunPod's gateway, a saved
RunPod template, provider networking/storage or GPU execution.
