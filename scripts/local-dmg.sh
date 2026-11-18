#!/usr/bin/env bash
# Package a local (unsigned or ad-hoc signed) VoiceInk.app into a DMG with an
# /Applications shortcut. Never signs or notarizes.
#
# Uses create-dmg with the release layout in release/dmg/ when it is installed
# (brew install create-dmg); otherwise falls back to a plain hdiutil image.
#
# Usage: scripts/local-dmg.sh <VoiceInk.app> <output.dmg>

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DMG_ASSET_DIR="$REPO_ROOT/release/dmg"

fail() {
    echo "Error: $*" >&2
    exit 1
}

[[ $# -eq 2 ]] || fail "usage: $0 <VoiceInk.app> <output.dmg>"
APP_PATH="${1%/}"
DMG_PATH="$2"

[[ -d "$APP_PATH" && -f "$APP_PATH/Contents/Info.plist" ]] \
    || fail "$APP_PATH is not an app bundle. Build it first with 'make local'."
[[ "$DMG_PATH" == *.dmg ]] || fail "output path must end in .dmg: $DMG_PATH"
APP_NAME="$(basename "$APP_PATH")"

mkdir -p "$(dirname "$DMG_PATH")"
DMG_PATH="$(cd "$(dirname "$DMG_PATH")" && pwd)/$(basename "$DMG_PATH")"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/voiceink-local-dmg.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

SOURCE_DIR="$WORK_DIR/source"
mkdir -p "$SOURCE_DIR"
ditto "$APP_PATH" "$SOURCE_DIR/$APP_NAME"

# create-dmg refuses to overwrite, so replace any previous local DMG.
rm -f "$DMG_PATH"

if command -v create-dmg >/dev/null 2>&1 && [[ -f "$DMG_ASSET_DIR/layout.conf" ]]; then
    # shellcheck source=../release/dmg/layout.conf
    source "$DMG_ASSET_DIR/layout.conf"
    echo "Building DMG with create-dmg and the release layout..."
    create-dmg \
        --volname "$DMG_VOLUME_NAME" \
        --volicon "$DMG_ASSET_DIR/volume-icon.icns" \
        --background "$DMG_ASSET_DIR/background.tiff" \
        --window-pos "$DMG_WINDOW_X" "$DMG_WINDOW_Y" \
        --window-size "$DMG_WINDOW_WIDTH" "$DMG_WINDOW_HEIGHT" \
        --text-size "$DMG_TEXT_SIZE" \
        --icon-size "$DMG_ICON_SIZE" \
        --icon "$APP_NAME" "$DMG_APP_X" "$DMG_APP_Y" \
        --hide-extension "$APP_NAME" \
        --app-drop-link "$DMG_APPLICATIONS_X" "$DMG_APPLICATIONS_Y" \
        --filesystem APFS \
        --format ULFO \
        --no-internet-enable \
        "$DMG_PATH" \
        "$SOURCE_DIR"
else
    echo "create-dmg not found; building a plain DMG with hdiutil (brew install create-dmg for the styled layout)..."
    ln -s /Applications "$SOURCE_DIR/Applications"
    hdiutil create \
        -volname "VoiceInk" \
        -srcfolder "$SOURCE_DIR" \
        -fs APFS \
        -format ULFO \
        -ov \
        "$DMG_PATH"
fi

hdiutil verify "$DMG_PATH" >/dev/null
echo ""
echo "DMG created: $DMG_PATH"
