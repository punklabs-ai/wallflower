#!/usr/bin/env bash
#
# check_radios.sh -- wallflower radio inventory / readiness check (MILESTONE-CRITICAL)
#
# This script is read-only inspection; it never changes radio state, never
# enables monitor mode, and never requires root.
#
# Detects the two Intel AX210 / Wi-Fi 6E radios that a wallflower perspective node
# needs, resolves each radio's netdev iface, phy, MAC, PCI id and driver, probes
# 6 GHz / 160 MHz capability, and proposes which radio takes the CSI role and
# which takes the BFI role (see contract.RADIO_ROLES, AP_CHANNELS).
#
# It CONFIRMS that exactly two AX210 radios are present and exits non-zero with a
# clear message otherwise.
#
# Usage:
#   scripts/check_radios.sh            # human-readable table
#   scripts/check_radios.sh --json     # machine-readable JSON blob
#   scripts/check_radios.sh -h|--help
#
# Works WITHOUT root on Ubuntu 26.04 (lspci, iw dev, ethtool -i, /sys reads are
# all unprivileged). sudo is only ever *suggested*, never required.
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Arg parsing
# --------------------------------------------------------------------------- #
JSON=0
for arg in "$@"; do
  case "$arg" in
    --json) JSON=1 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "check_radios.sh: unknown argument: $arg" >&2
      echo "try: check_radios.sh [--json] [--help]" >&2
      exit 2
      ;;
  esac
done

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
have() { command -v "$1" >/dev/null 2>&1; }

# Emit a non-table diagnostic only in human mode (keep JSON stdout clean).
note() { [ "$JSON" -eq 1 ] || echo "$@"; }

# Resolve the netdev iface bound to a given PCI slot (e.g. 0000:01:00.0).
# Reads /sys/class/net/*/device symlinks -- unprivileged.
iface_for_pci() {
  local pci_full="$1" n dev
  for n in /sys/class/net/*; do
    [ -e "$n/device" ] || continue
    dev="$(readlink -f "$n/device" 2>/dev/null || true)"
    if [ "${dev##*/}" = "$pci_full" ]; then
      echo "${n##*/}"
      return 0
    fi
  done
  return 1
}

# Resolve the phy name for an iface (prefer /sys, fall back to iw dev).
phy_for_iface() {
  local iface="$1" idx
  if [ -r "/sys/class/net/$iface/phy80211/name" ]; then
    cat "/sys/class/net/$iface/phy80211/name"
    return 0
  fi
  if have iw; then
    # iw dev prints "Interface <iface>" under a "phy#<N>" header.
    idx="$(iw dev 2>/dev/null | awk -v want="$iface" '
      /^phy#/ { gsub("phy#","",$1); cur=$1 }
      $1=="Interface" && $2==want { print cur; exit }')"
    [ -n "$idx" ] && { echo "phy$idx"; return 0; }
  fi
  return 1
}

# MAC for an iface from /sys (unprivileged).
mac_for_iface() {
  local iface="$1"
  [ -r "/sys/class/net/$iface/address" ] && cat "/sys/class/net/$iface/address" || echo ""
}

# Driver for an iface: prefer ethtool -i, fall back to /sys uevent.
driver_for_iface() {
  local iface="$1" drv=""
  if have ethtool; then
    drv="$(ethtool -i "$iface" 2>/dev/null | awk -F': ' '/^driver:/{print $2; exit}')"
  fi
  if [ -z "$drv" ] && [ -r "/sys/class/net/$iface/device/uevent" ]; then
    drv="$(awk -F= '/^DRIVER=/{print $2; exit}' "/sys/class/net/$iface/device/uevent")"
  fi
  echo "${drv:-unknown}"
}

# Firmware version (nice-to-have) from ethtool -i.
fw_for_iface() {
  local iface="$1"
  have ethtool || { echo ""; return 0; }
  ethtool -i "$iface" 2>/dev/null | awk -F': ' '/^firmware-version:/{print $2; exit}'
}

# Cache `iw phy <phy> info` once per phy. Reading it via a variable (rather than
# piping into grep -q) avoids a SIGPIPE race: grep -q exits on first match and
# closes the pipe, which under `set -o pipefail` can make the pipeline report
# failure non-deterministically. Grepping a captured string is stable.
declare -A PHY_INFO_CACHE
phy_info() {
  local phy="$1"
  have iw || { echo ""; return 0; }
  if [ -z "${PHY_INFO_CACHE[$phy]+x}" ]; then
    PHY_INFO_CACHE[$phy]="$(iw phy "$phy" info 2>/dev/null || true)"
  fi
  printf '%s' "${PHY_INFO_CACHE[$phy]}"
}

# Probe 6 GHz support for a phy. 6 GHz channels live in the 5955-7115 MHz range;
# presence of any such frequency line => 6 GHz capable. If the band is simply
# disabled by the regdomain the freqs won't be listed, so we also accept the
# AX210's advertised HE 6 GHz / "Band 4" capability. Returns "yes"/"unknown".
sixghz_for_phy() {
  local info; info="$(phy_info "$1")"
  [ -n "$info" ] || { echo "unknown"; return 0; }
  if grep -Eq '\b(59[5-9][0-9]|6[0-9]{3}|70[0-9][0-9]|71[01][0-9]) MHz \[' <<<"$info"; then
    echo "yes"; return 0
  fi
  if grep -Eqi 'HE.*6 ?GHz|Band 4' <<<"$info"; then
    echo "yes"; return 0
  fi
  echo "unknown"
}

# Probe 160 MHz support for a phy. AX210 advertises HE160 / 160 MHz width.
# Returns "yes" / "unknown".
width160_for_phy() {
  local info; info="$(phy_info "$1")"
  [ -n "$info" ] || { echo "unknown"; return 0; }
  if grep -Eqi '160 MHz|HE160' <<<"$info"; then
    echo "yes"; return 0
  fi
  echo "unknown"
}

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

# --------------------------------------------------------------------------- #
# 0. Preconditions (read-only tooling)
# --------------------------------------------------------------------------- #
if ! have lspci; then
  echo "ERROR: lspci not found. Install pciutils:  sudo apt-get install -y pciutils" >&2
  exit 3
fi
if ! have iw; then
  note "WARNING: 'iw' not found -- phy/capability detail will be limited."
  note "         Install with:  sudo apt-get install -y iw"
fi

note "==================================================================="
note " wallflower :: AX210 radio inventory (read-only, no root required)"
note "==================================================================="

# --------------------------------------------------------------------------- #
# 1. Find AX210 / Wi-Fi 6E radios via lspci
#    Match the human string ("AX210" or "Wi-Fi 6E") in the lspci -nn output.
# --------------------------------------------------------------------------- #
# lspci -D gives domain-qualified slots (0000:01:00.0); -nn appends [vendor:dev].
mapfile -t PCI_SLOTS < <(
  lspci -Dnn 2>/dev/null \
    | grep -iE 'AX210|Wi-Fi 6E|AX1675' \
    | awk '{print $1}'
)

COUNT=${#PCI_SLOTS[@]}

# Parallel arrays describing each discovered radio.
declare -a R_PCI R_PCISHORT R_IFACE R_PHY R_MAC R_DRIVER R_FW R_6GHZ R_160 R_ROLE R_AP R_CH

for slot in "${PCI_SLOTS[@]}"; do
  pci_short="${slot#0000:}"                 # 0000:01:00.0 -> 01:00.0
  iface="$(iface_for_pci "$slot" || true)"
  [ -z "$iface" ] && iface="(no-netdev)"
  if [ "$iface" != "(no-netdev)" ]; then
    phy="$(phy_for_iface "$iface" || echo '?')"
    mac="$(mac_for_iface "$iface")"
    drv="$(driver_for_iface "$iface")"
    fw="$(fw_for_iface "$iface")"
  else
    phy="?"; mac=""; drv="unknown"; fw=""
  fi
  six="unknown"; w160="unknown"
  if [ "$phy" != "?" ]; then
    six="$(sixghz_for_phy "$phy")"
    w160="$(width160_for_phy "$phy")"
  fi

  R_PCI+=("$slot")
  R_PCISHORT+=("$pci_short")
  R_IFACE+=("$iface")
  R_PHY+=("$phy")
  R_MAC+=("$mac")
  R_DRIVER+=("$drv")
  R_FW+=("$fw")
  R_6GHZ+=("$six")
  R_160+=("$w160")
  R_ROLE+=("")     # filled below
  R_AP+=("")
  R_CH+=("")
done

# --------------------------------------------------------------------------- #
# 2. Propose CSI vs BFI role assignment.
#    Deterministic + reproducible: sort discovered radios by PCI slot and assign
#    the lower slot to CSI, the higher to BFI. This matches contract.RADIO_ROLES
#    ("csi","bfi") and the pilot config (01:00.0->csi, 02:00.0->bfi) WITHOUT
#    hardcoding the specific addresses.
# --------------------------------------------------------------------------- #
# Build an index order sorted by PCI slot string.
mapfile -t ORDER < <(
  for i in "${!R_PCI[@]}"; do printf '%s\t%s\n' "${R_PCI[$i]}" "$i"; done \
    | sort | awk -F'\t' '{print $2}'
)

# AP_CSI_CHANNEL=37, AP_BFI_CHANNEL=85 mirror contract.py.
AP_CSI_CH=37
AP_BFI_CH=85
if [ "${#ORDER[@]}" -ge 1 ]; then
  ci="${ORDER[0]}"; R_ROLE[$ci]="csi"; R_AP[$ci]="AP-CSI"; R_CH[$ci]="$AP_CSI_CH"
fi
if [ "${#ORDER[@]}" -ge 2 ]; then
  bi="${ORDER[1]}"; R_ROLE[$bi]="bfi"; R_AP[$bi]="AP-BFI"; R_CH[$bi]="$AP_BFI_CH"
fi

# --------------------------------------------------------------------------- #
# 3a. JSON output mode
# --------------------------------------------------------------------------- #
if [ "$JSON" -eq 1 ]; then
  printf '{'
  printf '"tool":"check_radios",'
  printf '"node":"%s",' "$(json_escape "$(hostname)")"
  printf '"ax210_count":%d,' "$COUNT"
  printf '"ok":%s,' "$([ "$COUNT" -eq 2 ] && echo true || echo false)"
  printf '"expected":2,'
  printf '"radios":['
  for i in "${!R_PCI[@]}"; do
    [ "$i" -gt 0 ] && printf ','
    printf '{'
    printf '"pci":"%s",' "$(json_escape "${R_PCI[$i]}")"
    printf '"pci_short":"%s",' "$(json_escape "${R_PCISHORT[$i]}")"
    printf '"iface":"%s",' "$(json_escape "${R_IFACE[$i]}")"
    printf '"phy":"%s",' "$(json_escape "${R_PHY[$i]}")"
    printf '"mac":"%s",' "$(json_escape "${R_MAC[$i]}")"
    printf '"driver":"%s",' "$(json_escape "${R_DRIVER[$i]}")"
    printf '"firmware":"%s",' "$(json_escape "${R_FW[$i]}")"
    printf '"six_ghz":"%s",' "$(json_escape "${R_6GHZ[$i]}")"
    printf '"width_160mhz":"%s",' "$(json_escape "${R_160[$i]}")"
    printf '"proposed_role":"%s",' "$(json_escape "${R_ROLE[$i]}")"
    printf '"proposed_ap":"%s",' "$(json_escape "${R_AP[$i]}")"
    printf '"proposed_channel":%s' "${R_CH[$i]:-null}"
    printf '}'
  done
  printf ']}'
  printf '\n'
  [ "$COUNT" -eq 2 ] || exit 4
  exit 0
fi

# --------------------------------------------------------------------------- #
# 3b. Human-readable table
# --------------------------------------------------------------------------- #
echo
printf '%-9s  %-11s  %-5s  %-18s  %-8s  %-8s  %-8s\n' \
  "PCI" "IFACE" "PHY" "MAC" "DRIVER" "6GHz" "160MHz"
printf '%-9s  %-11s  %-5s  %-18s  %-8s  %-8s  %-8s\n' \
  "--------" "-----------" "-----" "------------------" "--------" "--------" "--------"
for i in "${!R_PCI[@]}"; do
  printf '%-9s  %-11s  %-5s  %-18s  %-8s  %-8s  %-8s\n' \
    "${R_PCISHORT[$i]}" "${R_IFACE[$i]}" "${R_PHY[$i]}" "${R_MAC[$i]:-?}" \
    "${R_DRIVER[$i]}" "${R_6GHZ[$i]}" "${R_160[$i]}"
done

echo
echo "Proposed role assignment (contract.RADIO_ROLES, contract.AP_CHANNELS):"
for i in "${!R_PCI[@]}"; do
  [ -n "${R_ROLE[$i]}" ] || continue
  printf '  %-3s -> %-5s  (%s, %s ch%s)\n' \
    "${R_IFACE[$i]}" "${R_ROLE[$i]}" "${R_PCISHORT[$i]}" "${R_AP[$i]}" "${R_CH[$i]}"
done

echo
echo "-------------------------------------------------------------------"
if [ "$COUNT" -eq 2 ]; then
  echo "RESULT: OK -- exactly 2 AX210/Wi-Fi 6E radios present."
  # Soft capability advisories (do not fail the milestone check on these).
  for i in "${!R_PCI[@]}"; do
    if [ "${R_6GHZ[$i]}" != "yes" ]; then
      echo "NOTE: 6 GHz not confirmed on ${R_IFACE[$i]} (${R_PHY[$i]}). The radio is"
      echo "      Wi-Fi 6E capable but the band may be disabled by the regulatory"
      echo "      domain. Operator can enable it:  sudo iw reg set <CC>   (e.g. US)"
    fi
  done
  echo "Next: enable monitor mode + set channel via scripts run by the operator"
  echo "      (see nodes/csi_agent and capture/ -- privileged ops are printed,"
  echo "      never auto-run, because node1 sudo needs a password)."
  exit 0
else
  echo "RESULT: FAIL -- expected exactly 2 AX210 radios, found ${COUNT}."
  if [ "$COUNT" -lt 2 ]; then
    echo "  A wallflower perspective node needs two AX210s (radioA=CSI, radioB=BFI)."
    echo "  Check the cards are seated and iwlwifi loaded:  lspci -nn | grep -i network"
    echo "                                                   dmesg | grep -i iwlwifi"
  else
    echo "  More than two matched. Disambiguate which pair to use in configs/nodes.yaml."
  fi
  exit 4
fi
