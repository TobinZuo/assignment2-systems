#!/usr/bin/env bash
set -euo pipefail

nsight_version=${NSIGHT_VERSION:-2025.6.3}
nsight_deb_url=${NSIGHT_DEB_URL:-https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb}
install_root=${NSIGHT_INSTALL_ROOT:-${HOME}/.local/nsight-systems-${nsight_version}}
nsight_root="${install_root}/opt/nvidia/nsight-systems/${nsight_version}"
nsys_path="${nsight_root}/bin/nsys"
importer_path="${nsight_root}/host-linux-x64/QdstrmImporter"

mkdir -p "${install_root}" "${HOME}/.local/bin"

if [[ -x "${nsys_path}" ]] && [[ -x "${importer_path}" ]] && ! ldd "${importer_path}" 2>/dev/null | grep -q "not found"; then
  ln -sf "${nsys_path}" "${HOME}/.local/bin/nsys"
  "${HOME}/.local/bin/nsys" --version
  exit 0
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

deb="${tmpdir}/nsight-systems-${nsight_version}.deb"
curl -fL --retry 3 --output "${deb}" "${nsight_deb_url}"
rm -rf "${install_root}"
mkdir -p "${install_root}"
dpkg-deb -x "${deb}" "${install_root}"

if [[ ! -x "${nsys_path}" ]]; then
  echo "Downloaded Nsight Systems, but could not find nsys at ${nsys_path}." >&2
  exit 127
fi

if [[ ! -x "${importer_path}" ]]; then
  echo "Downloaded Nsight Systems, but could not find QdstrmImporter at ${importer_path}." >&2
  exit 127
fi

if ldd "${importer_path}" | grep -q "not found"; then
  echo "Installed Nsight Systems, but QdstrmImporter still has missing dependencies:" >&2
  ldd "${importer_path}" | grep "not found" >&2
  exit 127
fi

ln -sf "${nsys_path}" "${HOME}/.local/bin/nsys"

command -v nsys
nsys --version
