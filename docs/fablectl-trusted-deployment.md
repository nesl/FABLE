# Trusted fablectl deployment

The repository checkout is a development input, not a daemon import path.
Build a reviewed release as an ordinary user:

```bash
export FABLE_ROOT="$(git rev-parse --show-toplevel)"
cd "$FABLE_ROOT"
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
  --asset iobt-minimal-ce-replay/setup/zed_settings/camera-a.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/camera-b.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/camera-c.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/camera-d.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/camera-e.conf \
  --asset iobt-minimal-ce-replay/setup/zed_settings/camera-f.conf \
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
