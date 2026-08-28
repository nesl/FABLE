"""Read-only `/fable` commands for the Codex Discord bridge.

Copy this file beside ``bot.py`` and register ``fable_group`` as documented in
``docs/fablectl-trusted-deployment.md``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

import discord
from discord import app_commands


FABLECTL = "/opt/fablectl/current/venv/bin/fablectl"
FABLECTL_CONFIG = "/etc/fablectl/config.yaml"
MAX_OUTPUT_BYTES = 256 * 1024
RejectCallback = Callable[[discord.Interaction], Awaitable[bool]]
_reject: RejectCallback | None = None

fable_group = app_commands.Group(
    name="fable", description="Inspect and plan trusted FABLE evaluations"
)


def configure_fable_commands(reject_callback: RejectCallback) -> None:
    global _reject
    _reject = reject_callback


async def _authorized(interaction: discord.Interaction) -> bool:
    if _reject is None:
        await interaction.response.send_message(
            "FABLE commands are not configured.", ephemeral=True
        )
        return False
    return not await _reject(interaction)


async def _invoke(interaction: discord.Interaction, arguments: list[str]) -> None:
    if not await _authorized(interaction):
        return
    await interaction.response.defer(ephemeral=True, thinking=True)
    argv = [
        FABLECTL,
        "--config",
        FABLECTL_CONFIG,
        "--json",
        *arguments,
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=35)
    except (OSError, asyncio.TimeoutError) as exc:
        await interaction.followup.send(
            f"fablectl could not complete: `{type(exc).__name__}`", ephemeral=True
        )
        return
    if len(stdout) > MAX_OUTPUT_BYTES or len(stderr) > MAX_OUTPUT_BYTES:
        await interaction.followup.send("fablectl response exceeded the safe limit.", ephemeral=True)
        return
    try:
        response = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        await interaction.followup.send(
            f"fablectl returned invalid output (exit {process.returncode}).",
            ephemeral=True,
        )
        return
    rendered = json.dumps(response, indent=2, sort_keys=True)
    if len(rendered) > 1800:
        rendered = rendered[:1780] + "\n…truncated"
    await interaction.followup.send(f"```json\n{rendered}\n```", ephemeral=True)


@fable_group.command(name="preflight", description="Run read-only host preflight checks")
async def preflight(interaction: discord.Interaction) -> None:
    await _invoke(interaction, ["preflight"])


@fable_group.command(name="validate", description="Validate an approved run manifest")
@app_commands.describe(manifest="Manifest path beneath the configured manifest root")
async def validate(interaction: discord.Interaction, manifest: str) -> None:
    await _invoke(interaction, ["run", "validate", manifest])


@fable_group.command(name="plan", description="Validate and display a non-executable run plan")
@app_commands.describe(manifest="Manifest path beneath the configured manifest root")
async def plan(interaction: discord.Interaction, manifest: str) -> None:
    await _invoke(interaction, ["run", "plan", manifest])


@fable_group.command(name="status", description="Inspect Docker stack status for a run")
async def status(interaction: discord.Interaction, run_id: str) -> None:
    await _invoke(interaction, ["stack", "status", run_id])


@fable_group.command(name="results", description="Inspect recorded evaluation results")
async def results(interaction: discord.Interaction, run_id: str) -> None:
    await _invoke(interaction, ["results", "inspect", "--run-id", run_id])
