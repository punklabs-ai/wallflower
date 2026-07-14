#!/usr/bin/env bash
#
# install_node.sh -- idempotent installer for a fresh wallflower perspective/recorder node.
#
#  wallflower reproduces the BFId paper (Todt, Morsbach, Strufe, CCS '25).
#  This installer sets up the CSI/BFI capture tooling on a node.
#
# What it does (idempotently -- safe to re-run):
#   1. Installs apt deps for capture + clock sync + build (skips ones present).
#   2. Creates a python venv and `pip install -e .[ml,capture]`.
#   3. Prints NEXT STEPS for things it cannot auto-install (PicoScenes, AX210
#      firmware verification, monitor mode, 6 GHz regulatory domain).
#
# Privilege handling: node1 sudo needs a password (no passwordless root). The
# script DETECTS whether non-interactive sudo works; if not, it PRINTS the exact
# apt-get command for the operator instead of running (or hanging on) it. It
# never performs destructive operations.
#
# Usage:
#   scripts/install_node.sh                 # full setup (apt may need operator)
#   scripts/install_node.sh --no-apt        # skip apt, just venv + pip
#   scripts/install_node.sh --venv .venv    # choose venv dir
#   scripts/install_node.sh -h|--help
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENV_DIR="$ROOT/.venv"
DO_APT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --no-apt) DO_APT=0; shift ;;
    --venv)   VENV_DIR="${2:?--venv needs a path}"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "install_node.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }

# apt package -> the command it provides (for "already installed?" detection).
# Order chosen to install build/tooling first.
APT_PKGS=(build-essential cmake dkms iw wireless-tools tcpdump tshark iperf3 \
          chrony python3 python3-pip python3-venv git ethtool)

echo "==================================================================="
echo " wallflower :: install_node.sh"
echo "==================================================================="
echo "Project root : $ROOT"
echo "Venv target  : $VENV_DIR"
echo

# --------------------------------------------------------------------------- #
# Detect whether we can run sudo non-interactively.
# --------------------------------------------------------------------------- #
SUDO=""
CAN_SUDO=0
if [ "$(id -u)" -eq 0 ]; then
  CAN_SUDO=1   # already root
elif have sudo && sudo -n true 2>/dev/null; then
  CAN_SUDO=1; SUDO="sudo"
fi

# --------------------------------------------------------------------------- #
# 1. apt dependencies
# --------------------------------------------------------------------------- #
if [ "$DO_APT" -eq 1 ]; then
  echo "[1/3] System packages (apt-get)"
  echo "-------------------------------------------------------------------"

  # Figure out which packages are missing using dpkg (idempotent skip).
  declare -a MISSING
  if have dpkg-query; then
    for p in "${APT_PKGS[@]}"; do
      if dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed"; then
        echo "  present : $p"
      else
        echo "  MISSING : $p"
        MISSING+=("$p")
      fi
    done
  else
    echo "  (dpkg-query unavailable; cannot pre-check -- will list all packages)"
    MISSING=("${APT_PKGS[@]}")
  fi

  if [ "${#MISSING[@]}" -eq 0 ]; then
    echo "All apt dependencies already installed -- nothing to do."
  else
    INSTALL_CMD="apt-get update && apt-get install -y ${MISSING[*]}"
    if [ "$CAN_SUDO" -eq 1 ]; then
      echo "Installing missing packages..."
      ${SUDO} apt-get update
      # shellcheck disable=SC2086
      ${SUDO} apt-get install -y "${MISSING[@]}"
      echo "apt install complete."
    else
      echo
      echo ">>> Non-interactive sudo NOT available on this node (sudo needs a"
      echo ">>> password). Run this yourself as the operator:"
      echo
      echo "    sudo $INSTALL_CMD"
      echo
      echo ">>> Then re-run: scripts/install_node.sh --no-apt"
    fi
  fi
else
  echo "[1/3] System packages: SKIPPED (--no-apt)"
fi

# --------------------------------------------------------------------------- #
# 2. Python virtualenv + editable install
# --------------------------------------------------------------------------- #
echo
echo "[2/3] Python environment"
echo "-------------------------------------------------------------------"
if ! have python3; then
  echo "ERROR: python3 not found. Install it first (see step 1) then re-run." >&2
  exit 3
fi
echo "python3 : $(python3 --version 2>&1)"

if [ -d "$VENV_DIR" ] && [ -x "$VENV_DIR/bin/python" ]; then
  echo "Reusing existing venv at $VENV_DIR (idempotent)."
else
  echo "Creating venv at $VENV_DIR ..."
  if ! python3 -m venv "$VENV_DIR" 2>/tmp/.wallflower_venv_err; then
    echo "WARNING: 'python3 -m venv' failed:"; cat /tmp/.wallflower_venv_err
    echo "  You likely need the venv package:  sudo apt-get install -y python3-venv"
    rm -f /tmp/.wallflower_venv_err
    echo "  Skipping pip install. Re-run after installing python3-venv."
    VENV_OK=0
  fi
  rm -f /tmp/.wallflower_venv_err
fi

if [ -x "$VENV_DIR/bin/python" ]; then
  VENV_OK=1
  VPIP="$VENV_DIR/bin/pip"
  echo "Upgrading pip ..."
  "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || \
    echo "  NOTE: pip upgrade skipped (no network?)."
  echo "Installing wallflower editable with [ml,capture] extras ..."
  echo "  (ml: numpy/torch/pandas/pyarrow/scikit-learn/optuna; capture: scapy)"
  if "$VPIP" install -e "$ROOT[ml,capture]"; then
    echo "Editable install OK."
  else
    echo "WARNING: editable install failed (network or build deps?)."
    echo "  Capture/orchestrator/nodes are stdlib-only and still work without it."
    echo "  Retry online with:  $VPIP install -e \"$ROOT[ml,capture]\""
  fi
else
  VENV_OK=0
fi

# --------------------------------------------------------------------------- #
# 3. Manual NEXT STEPS that cannot be auto-installed
# --------------------------------------------------------------------------- #
echo
echo "[3/3] NEXT STEPS (manual -- not auto-installable)"
echo "==================================================================="

echo
echo "A) PicoScenes (REQUIRED for CSI capture on AX210)"
echo "   PicoScenes is the CSI extraction toolkit; it is not on apt and must be"
echo "   installed from its official distribution. See:"
echo "       https://ps.zpj.io/install.html"
echo "   Follow the AX210 / iwlwifi instructions for your kernel ($(uname -r))."
echo "   Without PicoScenes the CSI modality (csi_p*.raw) cannot be recorded;"
echo "   BFI capture via tcpdump still works."

echo
echo "B) AX210 firmware check"
if have ethtool; then
  for ifc in $(ls /sys/class/net 2>/dev/null); do
    drv="$(ethtool -i "$ifc" 2>/dev/null | awk -F': ' '/^driver:/{print $2}')"
    [ "$drv" = "iwlwifi" ] || continue
    fw="$(ethtool -i "$ifc" 2>/dev/null | awk -F': ' '/^firmware-version:/{print $2}')"
    echo "   $ifc : driver=iwlwifi firmware=${fw:-unknown}"
  done
  echo "   If firmware is missing/old:  sudo apt-get install -y firmware-iwlwifi"
  echo "   (then reboot). AX210 needs recent iwlwifi firmware for 6 GHz."
else
  echo "   (ethtool not present -- install it to inspect iwlwifi firmware.)"
fi
echo "   Verify both radios are detected:  scripts/check_radios.sh"

echo
echo "C) Monitor mode (REQUIRED for CSI + the BFI recorder; needs root)"
echo "   These change radio state and need a password on node1, so run them"
echo "   yourself (the node agents PRINT, never auto-run, privileged ops):"
echo "       sudo ip link set <iface> down"
echo "       sudo iw dev <iface> set type monitor"
echo "       sudo ip link set <iface> up"
echo "       sudo iw dev <iface> set channel <ch> 160MHz   # ch37 CSI / ch85 BFI"

echo
echo "D) 6 GHz regulatory domain"
cur_reg="$(iw reg get 2>/dev/null | awk '/^country/{print $2; exit}' || echo '??')"
echo "   Current reg domain: ${cur_reg:-unknown}. 6 GHz (channels 37 & 85) is"
echo "   gated by the regulatory domain; '00' (world) disables most 6 GHz use."
echo "   Set your lab's country code so 6 GHz / 160 MHz become available:"
echo "       sudo iw reg set <CC>      # e.g. US, DE -- use YOUR jurisdiction"
echo "   Confirm afterwards:  iw reg get   and   scripts/check_radios.sh"

echo
echo "==================================================================="
echo " Done. Activate the venv with:  source $VENV_DIR/bin/activate"
echo " Then verify radios:            scripts/check_radios.sh"
echo " And clock sync:                scripts/sync_clocks.sh"
echo "==================================================================="
