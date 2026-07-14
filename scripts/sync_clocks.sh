#!/usr/bin/env bash
#
# sync_clocks.sh -- inspect / enforce clock synchronisation across wallflower nodes
#                   before a recording session.
#
# Only nodes listed in configs/nodes.yaml are contacted,
# over ssh in BatchMode (no interactive password). The script is read-only: it
# never starts/stops daemons and never changes any clock. Where corrective action
# is needed it PRINTS the command for the operator (node1 sudo needs a password).
#
# WHY THIS MATTERS: BFId fuses 4 perspectives + a central BFI recorder. The
# parser aligns variable-length CSI (~285 Hz) and BFI (~10 Hz) streams by their
# appended time-delta column, so cross-node clock offset directly bounds the
# achievable temporal alignment. chrony (NTP) gets you ~ms-to-sub-ms on a LAN;
# true ms / sub-ms alignment really wants PTP (ptp4l + hardware timestamping).
#
# Behaviour:
#   1. Detect a local time-sync daemon (chrony > ntp > ptp4l) and show its status
#      / offset (chronyc tracking | ntpq -p | pmc).
#   2. If none is present, print install + setup guidance.
#   3. Regardless, compare wall clocks across the configured nodes by sampling
#      `date +%s.%N` over ssh and report the max pairwise offset vs --max-offset-ms.
#      Unreachable nodes are WARNED about and skipped, never fatal.
#
# Usage:
#   scripts/sync_clocks.sh
#   scripts/sync_clocks.sh --nodes node1,persp2.lab.local --max-offset-ms 5
#   scripts/sync_clocks.sh --config configs/nodes.yaml
#   scripts/sync_clocks.sh -h|--help
#
# Defaults: --max-offset-ms 5 (NOTE: 5 ms is only realistically met with PTP;
# plain NTP/chrony on a busy LAN may sit in the 1-50 ms range -- treat a PASS at
# 5 ms over ssh-sampled wall clocks as indicative, not authoritative).
#
set -euo pipefail

# --------------------------------------------------------------------------- #
# Defaults / args
# --------------------------------------------------------------------------- #
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="$ROOT/configs/nodes.yaml"
NODES_CSV=""
MAX_OFFSET_MS=5
SSH_USER="${USER:-ADMIN_ACCOUNT}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8)

while [ $# -gt 0 ]; do
  case "$1" in
    --nodes)          NODES_CSV="${2:?--nodes needs a value}"; shift 2 ;;
    --max-offset-ms)  MAX_OFFSET_MS="${2:?--max-offset-ms needs a value}"; shift 2 ;;
    --config)         CONFIG="${2:?--config needs a value}"; shift 2 ;;
    --user)           SSH_USER="${2:?--user needs a value}"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "sync_clocks.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
is_local() { case "$1" in localhost|127.0.0.1|"$(hostname)"|"$(hostname -s)") return 0 ;; *) return 1 ;; esac; }

# --------------------------------------------------------------------------- #
# 1. Local time-sync daemon detection + status
# --------------------------------------------------------------------------- #
echo "==================================================================="
echo " wallflower :: clock synchronisation check"
echo "==================================================================="
echo
echo "[1/3] Local time-sync daemon"
echo "-------------------------------------------------------------------"
SYNC_FOUND=0
if have chronyc; then
  echo "chrony detected (recommended for wallflower LAN sync)."
  if chronyc tracking >/tmp/.wallflower_chrony 2>&1; then
    SYNC_FOUND=1
    grep -E 'Reference ID|Stratum|System time|Last offset|RMS offset' /tmp/.wallflower_chrony || cat /tmp/.wallflower_chrony
  else
    echo "WARNING: chronyc present but 'chronyc tracking' failed (daemon not running?)."
    echo "  Operator may need:  sudo systemctl enable --now chrony"
  fi
  rm -f /tmp/.wallflower_chrony
elif have ntpq; then
  echo "ntpd detected."
  ntpq -p 2>&1 | head -n 20 || echo "WARNING: 'ntpq -p' failed."
  SYNC_FOUND=1
elif have pmc; then
  echo "ptp4l/pmc detected (PTP -- best for tight sub-ms sync)."
  # 'pmc' typically needs root for the local socket; print, don't fail.
  if pmc -u -b 0 'GET CURRENT_DATA_SET' >/tmp/.wallflower_pmc 2>&1; then
    SYNC_FOUND=1; cat /tmp/.wallflower_pmc
  else
    echo "NOTE: 'pmc' query failed (often needs root). Operator can run:"
    echo "  sudo pmc -u -b 0 'GET CURRENT_DATA_SET'"
  fi
  rm -f /tmp/.wallflower_pmc
else
  echo "No time-sync daemon found (chrony / ntpd / ptp4l all absent)."
  echo
  echo "SETUP GUIDANCE (run as operator -- node1 sudo needs a password):"
  echo "  # Recommended: chrony (simple, good ms-level LAN sync)"
  echo "    sudo apt-get install -y chrony"
  echo "    sudo systemctl enable --now chrony"
  echo "    chronyc tracking          # verify offset"
  echo
  echo "  # Tighter sync (sub-ms, for best cross-perspective BFI/CSI alignment):"
  echo "  # PTP via linuxptp -- requires NIC hardware timestamping support."
  echo "    sudo apt-get install -y linuxptp"
  echo "    sudo ptp4l -i eno1 -m      # one master, others as slaves"
fi

# --------------------------------------------------------------------------- #
# 2. Resolve node list
# --------------------------------------------------------------------------- #
echo
echo "[2/3] Resolving nodes"
echo "-------------------------------------------------------------------"
declare -a NODES
if [ -n "$NODES_CSV" ]; then
  IFS=',' read -r -a NODES <<< "$NODES_CSV"
  echo "Using --nodes: ${NODES[*]}"
elif [ -f "$CONFIG" ]; then
  # Minimal YAML scrape: pull every 'host:' value. Avoids a python dep so this
  # runs on bare nodes. (Full parsing lives in the orchestrator's python.)
  mapfile -t NODES < <(
    grep -E '^[[:space:]]*host:[[:space:]]*' "$CONFIG" \
      | sed -E 's/^[[:space:]]*host:[[:space:]]*//; s/[[:space:]]*#.*$//' \
      | tr -d '"'"'" \
      | grep -vE '\.lab\.local$' \
      | sort -u
  )
  if [ "${#NODES[@]}" -eq 0 ]; then
    echo "No concrete (non-placeholder) hosts in $CONFIG; defaulting to localhost."
    NODES=(localhost)
  else
    echo "From $CONFIG (placeholders *.lab.local skipped): ${NODES[*]}"
  fi
else
  echo "No --nodes and config $CONFIG not found; defaulting to localhost."
  NODES=(localhost)
fi

# --------------------------------------------------------------------------- #
# 3. Sample wall clocks and compute max offset
# --------------------------------------------------------------------------- #
echo
echo "[3/3] Cross-node wall-clock comparison (tolerance: ${MAX_OFFSET_MS} ms)"
echo "-------------------------------------------------------------------"

# We sample each node's epoch immediately after sampling the controller's epoch,
# then subtract the (small, measured) ssh round-trip estimate. This is coarse --
# it can't beat ssh latency -- so it is a sanity gate, not a metrology tool.
declare -a OK_NODES OK_OFFSETS
for node in "${NODES[@]}"; do
  if is_local "$node"; then
    echo "  $node : local (offset 0.000 ms by definition)"
    OK_NODES+=("$node"); OK_OFFSETS+=("0.000")
    continue
  fi
  # t0 (local) -> remote date -> t1 (local); estimate remote moment as midpoint.
  t0="$(date +%s.%N)"
  remote="$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${node}" 'date +%s.%N' 2>/dev/null || true)"
  t1="$(date +%s.%N)"
  if [ -z "$remote" ]; then
    echo "  $node : WARNING unreachable / ssh failed -- skipping (not fatal)."
    continue
  fi
  # offset_ms = (remote - midpoint(t0,t1)) * 1000, absolute value.
  offset_ms="$(awk -v r="$remote" -v a="$t0" -v b="$t1" \
    'BEGIN{ mid=(a+b)/2; d=(r-mid)*1000; if(d<0)d=-d; printf "%.3f", d }')"
  rtt_ms="$(awk -v a="$t0" -v b="$t1" 'BEGIN{ printf "%.1f", (b-a)*1000 }')"
  echo "  $node : offset ~${offset_ms} ms (ssh rtt ~${rtt_ms} ms)"
  OK_NODES+=("$node"); OK_OFFSETS+=("$offset_ms")
done

# Determine the max measured offset.
MAX="0.000"
for o in "${OK_OFFSETS[@]:-}"; do
  [ -n "$o" ] || continue
  MAX="$(awk -v m="$MAX" -v x="$o" 'BEGIN{ print (x>m)?x:m }')"
done

echo
echo "-------------------------------------------------------------------"
echo "Reachable nodes : ${#OK_NODES[@]} / ${#NODES[@]}"
echo "Max offset      : ${MAX} ms   (tolerance ${MAX_OFFSET_MS} ms)"

if [ "$SYNC_FOUND" -eq 0 ]; then
  echo "ADVISORY: no time-sync daemon -- install chrony before recording."
fi
echo "ADVISORY: ms / sub-ms alignment across perspectives needs PTP, not NTP."

PASS="$(awk -v m="$MAX" -v t="$MAX_OFFSET_MS" 'BEGIN{ print (m<=t)?1:0 }')"
if [ "$PASS" -eq 1 ]; then
  echo "RESULT: PASS -- measured offset within tolerance (indicative only)."
  exit 0
else
  echo "RESULT: OVER TOLERANCE -- max offset ${MAX} ms exceeds ${MAX_OFFSET_MS} ms."
  echo "        Tighten sync (chrony/PTP) before recording. (exit 1)"
  exit 1
fi
