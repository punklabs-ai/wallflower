"""Thin SSH/local wrapper to invoke node-agents and parse their JSON.

Honours the NODE-AGENT INVOCATION CONTRACT: every agent is invoked as

    python3 -m nodes.<agent> <action> --participant P001 --style normal \
        --trial 001 [--perspective N] [--out-dir DIR]

and prints ONE structured JSON object to stdout. This module runs that command
either locally (host == "localhost", as in --profile pilot) or over ssh, and
returns the parsed JSON plus exit metadata. Missing hosts / unreachable nodes
are tolerated gracefully (ok=False, never raises).

stdlib-only (subprocess, json, shlex) so it can run on a bare controller.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Sequence


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class AgentResult:
    """Outcome of one node-agent invocation."""
    agent: str
    action: str
    node: str
    host: str
    ok: bool
    perspective: int | None = None
    cmd: list[str] = field(default_factory=list)
    json: dict[str, Any] | None = None     # parsed structured log (if any)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""                         # transport/parse error, if any

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "node": self.node,
            "host": self.host,
            "perspective": self.perspective,
            "ok": self.ok,
            "cmd": self.cmd,
            "returncode": self.returncode,
            "json": self.json,
            "error": self.error,
            # stdout/stderr kept short in the structured report
            "stderr": self.stderr.strip()[-500:],
        }


def is_local(host: str) -> bool:
    return host in LOCAL_HOSTS


def build_agent_argv(
    agent: str,
    action: str,
    *,
    participant: str | None = None,
    style: str | None = None,
    trial: str | None = None,
    perspective: int | None = None,
    out_dir: str | None = None,
    extra: Sequence[str] | None = None,
    python: str = "python3",
) -> list[str]:
    """Construct the node-agent command per the invocation contract."""
    argv = [python, "-m", f"nodes.{agent}", action]
    if participant is not None:
        argv += ["--participant", participant]
    if style is not None:
        argv += ["--style", style]
    if trial is not None:
        argv += ["--trial", trial]
    if perspective is not None:
        argv += ["--perspective", str(perspective)]
    if out_dir is not None:
        argv += ["--out-dir", out_dir]
    if extra:
        argv += list(extra)
    return argv


def _ssh_argv(
    host: str,
    remote_cmd: Sequence[str],
    *,
    user: str | None = None,
    connect_timeout_s: int = 8,
    options: Sequence[str] | None = None,
    cwd: str | None = None,
) -> list[str]:
    target = f"{user}@{host}" if user else host
    argv = ["ssh"]
    argv += ["-o", f"ConnectTimeout={connect_timeout_s}"]
    if options:
        argv += list(options)
    argv.append(target)
    # Build a single remote shell string; cd into the project root if given so
    # 'python3 -m nodes.<agent>' resolves.
    quoted = " ".join(shlex.quote(a) for a in remote_cmd)
    if cwd:
        quoted = f"cd {shlex.quote(cwd)} && {quoted}"
    argv.append(quoted)
    return argv


def _parse_json_stdout(stdout: str) -> dict[str, Any] | None:
    """Parse the agent's structured JSON log.

    Agents emit ONE JSON object on stdout, but may also log human text. Try a
    direct parse first, then fall back to the last JSON-looking line.
    """
    s = stdout.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(s.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def run_agent(
    *,
    agent: str,
    action: str,
    node: str,
    host: str,
    participant: str | None = None,
    style: str | None = None,
    trial: str | None = None,
    perspective: int | None = None,
    out_dir: str | None = None,
    extra: Sequence[str] | None = None,
    ssh_user: str | None = None,
    ssh_options: Sequence[str] | None = None,
    connect_timeout_s: int = 8,
    timeout_s: int = 60,
    remote_cwd: str | None = None,
    python: str = "python3",
    dry_run: bool = False,
) -> AgentResult:
    """Invoke one node-agent (locally or via ssh) and parse its JSON output.

    Never raises on transport failure: returns AgentResult(ok=False, error=...).
    """
    remote_cmd = build_agent_argv(
        agent, action,
        participant=participant, style=style, trial=trial,
        perspective=perspective, out_dir=out_dir, extra=extra, python=python,
    )

    if is_local(host):
        argv = remote_cmd
    else:
        argv = _ssh_argv(
            host, remote_cmd,
            user=ssh_user, connect_timeout_s=connect_timeout_s,
            options=ssh_options, cwd=remote_cwd,
        )

    res = AgentResult(agent=agent, action=action, node=node, host=host, ok=False,
                      perspective=perspective, cmd=argv)

    if dry_run:
        res.ok = True
        res.error = ""
        res.json = {"dry_run": True}
        return res

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        # ssh or python not present on controller
        res.error = f"command not found: {exc}"
        return res
    except subprocess.TimeoutExpired:
        res.error = f"timeout after {timeout_s}s"
        return res
    except OSError as exc:
        res.error = f"os error: {exc}"
        return res

    res.returncode = proc.returncode
    res.stdout = proc.stdout
    res.stderr = proc.stderr
    res.json = _parse_json_stdout(proc.stdout)

    if res.json is not None and "ok" in res.json:
        res.ok = bool(res.json["ok"])
    else:
        res.ok = proc.returncode == 0 and res.json is not None
        if res.json is None:
            res.error = res.error or "no structured JSON on stdout"
    return res


def run_remote_raw(
    *,
    host: str,
    cmd: Sequence[str],
    ssh_user: str | None = None,
    ssh_options: Sequence[str] | None = None,
    connect_timeout_s: int = 8,
    timeout_s: int = 30,
    dry_run: bool = False,
) -> tuple[int | None, str, str, str]:
    """Run an arbitrary command locally or over ssh (used by clocks.py).

    Returns (returncode, stdout, stderr, error). Never raises.
    """
    if is_local(host):
        argv = list(cmd)
    else:
        argv = _ssh_argv(
            host, cmd, user=ssh_user,
            connect_timeout_s=connect_timeout_s, options=ssh_options,
        )
    if dry_run:
        return 0, "", "", ""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, check=False,
        )
    except FileNotFoundError as exc:
        return None, "", "", f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return None, "", "", f"timeout after {timeout_s}s"
    except OSError as exc:
        return None, "", "", f"os error: {exc}"
    return proc.returncode, proc.stdout, proc.stderr, ""
