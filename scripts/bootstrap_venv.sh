#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ -x "${VENV_DIR}/bin/python" && -x "${VENV_DIR}/bin/pip" ]]; then
  echo "[setup] Using existing venv: ${VENV_DIR}"
else
  echo "[setup] Creating venv (without pip): ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
  "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"

  GETPIP="${ROOT_DIR}/.get-pip.py"
  echo "[setup] Bootstrapping pip via get-pip.py (ensurepip may be missing)"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${GETPIP}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${GETPIP}" https://bootstrap.pypa.io/get-pip.py
  else
    # Last-resort downloader using system python (still not installing to system env).
    "${PYTHON_BIN}" - <<'PY'
import urllib.request, pathlib
url = "https://bootstrap.pypa.io/get-pip.py"
dst = pathlib.Path(".get-pip.py")
dst.write_bytes(urllib.request.urlopen(url).read())
print(f"Downloaded {url} -> {dst}")
PY
  fi

  "${VENV_DIR}/bin/python" "${GETPIP}"
  rm -f "${GETPIP}"
fi

echo "[setup] Upgrading packaging tools in venv"
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel

echo "[setup] Installing project requirements"
"${VENV_DIR}/bin/pip" install -r requirements.txt

echo "[setup] Installing this repo (editable)"
"${VENV_DIR}/bin/pip" install -e .

echo "[setup] Done. Activate with: source ${VENV_DIR}/bin/activate"
