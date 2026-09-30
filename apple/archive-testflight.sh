#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
PROJECT_PATH="$SCRIPT_DIR/neTV.xcodeproj"
SCHEME="neTV-tvOS"
BUILD_NUMBER="${NETV_BUILD_NUMBER:-$(date -u +'%y%m.%d%H.%M%S')}"
ARCHIVE_DAY=$(date +'%Y-%m-%d')
ARCHIVE_PATH="${NETV_ARCHIVE_PATH:-$HOME/Library/Developer/Xcode/Archives/$ARCHIVE_DAY/$SCHEME-$BUILD_NUMBER.xcarchive}"

case "$BUILD_NUMBER" in
    ""|*[!0-9.]*)
        echo "Build number must contain only digits and periods: $BUILD_NUMBER" >&2
        exit 1
        ;;
esac

if [ ! -d "$PROJECT_PATH" ]; then
    echo "Xcode project not found: $PROJECT_PATH" >&2
    exit 1
fi

if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app/Contents/Developer ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

if ! command -v xcodebuild >/dev/null 2>&1; then
    echo "xcodebuild is required to create a TestFlight archive." >&2
    exit 1
fi

mkdir -p "$(dirname -- "$ARCHIVE_PATH")"

echo "Archiving $SCHEME with build number $BUILD_NUMBER"
xcodebuild \
    -project "$PROJECT_PATH" \
    -scheme "$SCHEME" \
    -configuration Release \
    -destination "generic/platform=tvOS" \
    -archivePath "$ARCHIVE_PATH" \
    -allowProvisioningUpdates \
    CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
    archive

echo "Archive created at $ARCHIVE_PATH"
if ! open "$ARCHIVE_PATH"; then
    echo "Could not open the archive automatically; open it from Xcode Organizer." >&2
fi
