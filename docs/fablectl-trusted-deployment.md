# Trusted fablectl deployment

The repository checkout is a development input, not a daemon import path.
Build a reviewed release as an ordinary user:

```bash
cd /home/brianw/Documents/FABLE
python3 scripts/build_fablectl_release.py \
  --commit <FULL_REVIEWED_40_CHARACTER_COMMIT> \
  --deployment-id smoke-v1 \
  --staging-dir "$HOME/fablectl-staging/smoke-v1" \
  --compose-file iobt-minimal-ce-replay/compose.server.yaml \
  --compose-file iobt-minimal-ce-replay/compose.replay.yaml \
  --compose-file iobt-minimal-ce-replay/compose.fable.yaml \
  --compose-file iobt-minimal-ce-replay/compose.fable.phase7.yaml \
  --compose-file iobt-minimal-ce-replay/compose.fable.phase8.yaml \
  --compose-file iobt-minimal-ce-replay/compose.fable.evaluation.yaml \
  --asset iobt-minimal-ce-replay/setup/zed_settings/SN31366375.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/SN35309867.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/SN36577075.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/SN37711387.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/SN39164952.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/SN39424035.conf \
  --script scripts/compile_request.py \
  --script iobt-minimal-ce-replay/tools/replay_control.py
```

The administrator must review the generated wheel, bundle, commit, and hashes.
Example manual installation commands are intentionally not run by the build
tool:

```bash
VERSION=0.10.0
DEPLOYMENT_ID=smoke-v1
STAGING="$HOME/fablectl-staging/<COMMIT>-$DEPLOYMENT_ID"
WHEEL="$STAGING/releases/$VERSION/fable_runtime-$VERSION-py3-none-any.whl"

sudo install -d -o root -g root -m 0755 \
  "/opt/fablectl/releases/$VERSION" \
  /opt/fablectl/deployments \
  /opt/fablectl/models \
  /etc/fablectl \
  /etc/systemd/user
sudo python3 -m venv "/opt/fablectl/releases/$VERSION/venv"
sudo "/opt/fablectl/releases/$VERSION/venv/bin/pip" install "$WHEEL"
sudo cp -a "$STAGING/deployments/$DEPLOYMENT_ID" \
  "/opt/fablectl/deployments/$DEPLOYMENT_ID"
sudo chown -R root:root \
  "/opt/fablectl/releases/$VERSION" \
  "/opt/fablectl/deployments/$DEPLOYMENT_ID"
sudo chmod -R go-w \
  "/opt/fablectl/releases/$VERSION" \
  "/opt/fablectl/deployments/$DEPLOYMENT_ID"
sudo ln -sfn "/opt/fablectl/releases/$VERSION" /opt/fablectl/current
sudo install -o root -g root -m 0644 \
  config/fablectl.production.example.yaml /etc/fablectl/config.yaml
sudo install -o root -g root -m 0644 \
  systemd/fablectld.service /etc/systemd/user/fablectld.service

systemctl --user daemon-reload
systemctl --user enable --now fablectld.service
/opt/fablectl/current/venv/bin/python -c \
  'import fablectl; print(fablectl.__file__)'
systemctl --user show fablectld.service \
  --property=ExecStart --property=WorkingDirectory --no-pager
```

The administrator must adjust every executable or policy-bearing path in
`/etc/fablectl/config.yaml`, including `repository_root`, `replay_root`,
`venv_python`, `manifest_root`, `deployment_root`, and client paths, so they
refer only to reviewed locations with appropriate ownership. Manifests may live
in a separately writable submission directory because they are schema checked;
daemon code, scripts, Compose files, and deployment manifests must not.
All mutation gates remain false until a separate review.

## Discord bridge

`fablectl.discord_bridge.DiscordFableBridge` prepares the `/fable` operations
`preflight`, `validate`, `plan`, `status`, and `results`. It invokes the fixed
client `/opt/fablectl/current/bin/fablectl` using an argument array and
`shell=False`. It exposes no arbitrary command operation. Mutating Discord
commands are absent and additionally default to disabled in bridge
configuration.

For the existing bridge at `~/Documents/codex-discord-bridge`, copy
`integrations/codex-discord-bridge/fable_commands.py` beside `bot.py`. Then add:

```python
from fable_commands import configure_fable_commands, fable_group
```

After `codex_group` is created, configure the shared authorization callback:

```python
configure_fable_commands(reject_unless_authorized)
```

Finally, in `CodexDiscordBot.setup_hook`, immediately after registering
`codex_group`, register the new group:

```python
self.tree.add_command(fable_group)
```

The module uses `asyncio.create_subprocess_exec` with a fixed absolute client,
an argument array, bounded output, and no shell. It exposes no mutation command.
