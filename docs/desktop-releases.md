# Desktop releases

Infernet publishes two desktop installers:

- Apple Silicon macOS: signed and notarized `Infernet_<version>_aarch64.dmg`.
- Windows x64: unsigned current-user NSIS `Infernet_<version>_x64-setup.exe`.

There is intentionally no MSI build and no Windows Authenticode certificate. Windows users must choose **More info → Run anyway** the first time they install Infernet. Updater artifacts are still signed with Infernet's separate Tauri updater key.

## One-time GitHub secrets

The release workflow requires these repository secrets:

- `TAURI_SIGNING_PRIVATE_KEY`: contents of `~/.tauri/infernet-updater.key`.
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`: empty for the current unencrypted updater key.
- `APPLE_CERTIFICATE`: base64-encoded Developer ID Application `.p12` export.
- `APPLE_CERTIFICATE_PASSWORD`: password used when exporting the `.p12`.
- `KEYCHAIN_PASSWORD`: a temporary password used for the CI build keychain.
- `APPLE_ID`: Apple ID used for notarization.
- `APPLE_PASSWORD`: Apple app-specific password used for notarization.

The Apple team ID is already pinned to `4PDUNTF69S` in the workflow. The local `AC_NOTARY` keychain profile described in `/Users/christopher/HOW-TO-SIGN-MAC-APPS.md` remains useful for manual notarization, but GitHub-hosted runners need the secrets above because they cannot access this Mac's login keychain.

The updater private key must never be committed. Losing it prevents installed copies from accepting future updates. Its public key is embedded in `infernet-ui/src-tauri/tauri.conf.json`.

## Publish a release

1. Update `version` in `infernet-ui/src-tauri/tauri.conf.json`.
2. Commit the release.
3. Create and push a matching tag, for example `app-v0.2.0`.
4. The `Release desktop apps` workflow builds both native installers and publishes one GitHub Release.
5. Verify the macOS DMG with `spctl` and `stapler`, then install both artifacts on clean machines.

The in-app updater checks:

`https://github.com/GnosysLabs/Infernet/releases/latest/download/latest.json`

The release must be published—not left as a draft—for installed apps to discover it.

## Local macOS release check

The certificate and `AC_NOTARY` profile are already configured on this Mac. Tauri itself expects Apple ID notarization variables when it performs the full build, while `xcrun notarytool` can continue using `AC_NOTARY` for manual verification.

Use `security find-identity -v -p codesigning` to copy the exact Developer ID Application identity. A complete local Tauri notarization build also needs `APPLE_ID`, `APPLE_PASSWORD`, and `APPLE_TEAM_ID`; the Electron-specific `APPLE_KEYCHAIN_PROFILE=AC_NOTARY` variable is not consumed by Tauri. The GitHub workflow is therefore the canonical production release path.

## Verification

```bash
codesign -dv --verbose=4 "target/aarch64-apple-darwin/release/bundle/macos/Infernet.app"
spctl -a -vvv "target/aarch64-apple-darwin/release/bundle/macos/Infernet.app"
xcrun stapler validate "target/aarch64-apple-darwin/release/bundle/dmg/Infernet_*.dmg"
```

On Windows, confirm that only an NSIS setup executable and its updater signature are published; no `.msi` should exist.
