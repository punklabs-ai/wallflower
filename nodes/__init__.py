"""Node-side agents for the wallflower capture stack.

Each agent runs *on a node* (a perspective machine with 2x Intel AX210 radios)
and is invoked over SSH by the orchestrator following the NODE-AGENT INVOCATION
CONTRACT:

    python3 -m nodes.<agent> <action> --participant P001 --style normal \
        --trial 001 [--perspective N] [--out-dir DIR]

    actions: detect | start | stop | status | health

Every agent prints exactly ONE structured JSON object to stdout and follows the
"degrade gracefully without root" rule: anything requiring privilege (monitor
mode, channel set, packet capture, package install) is detected; if privilege
is missing the agent prints the EXACT command an operator should run and exits
non-fatally instead of crashing.

These modules are STDLIB-ONLY so they run on a bare node with no pip install.
Capture targets a single configured AP BSSID; see bfi_recorder_agent.
"""
