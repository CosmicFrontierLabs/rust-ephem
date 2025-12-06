#!/bin/bash
# Build manylinux/musllinux wheels locally using Docker
#
# Usage:
#   ./scripts/build_wheels.sh              # Build all platforms
#   ./scripts/build_wheels.sh linux        # Build only manylinux wheels
#   ./scripts/build_wheels.sh musllinux    # Build only musllinux wheels (Alpine)
#   ./scripts/build_wheels.sh macos        # Build only macOS wheels (native only)
#
# Caching:
#   Build artifacts are cached in .cache/cargo-* directories for faster rebuilds.
#   To clear cache: rm -rf .cache/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Setup cache directories for Cargo registry and build artifacts
CACHE_DIR="$PROJECT_DIR/.cache"
mkdir -p "$CACHE_DIR/cargo-registry"
mkdir -p "$CACHE_DIR/cargo-git"
mkdir -p "$CACHE_DIR/cargo-target-manylinux-x86_64"
mkdir -p "$CACHE_DIR/cargo-target-manylinux-aarch64"
mkdir -p "$CACHE_DIR/cargo-target-musllinux-x86_64"
mkdir -p "$CACHE_DIR/cargo-target-musllinux-aarch64"

# Common cache volume mounts
CACHE_MOUNTS="-v $CACHE_DIR/cargo-registry:/root/.cargo/registry -v $CACHE_DIR/cargo-git:/root/.cargo/git"

# Get version from git describe --tags and convert to PEP 440 format
# git describe outputs: v0.1.13 or v0.1.13-2-g13fb01b
# PEP 440 wants: 0.1.13 or 0.1.13.dev2
# Rust semver wants: 0.1.13 or 0.1.13-dev.2
RAW_VERSION=$(git describe --tags 2>/dev/null || echo "v0.1.0")
# Strip leading 'v'
RAW_VERSION=${RAW_VERSION#v}

# Convert git describe format to both formats
# v0.1.13-2-g13fb01b -> rust: 0.1.13-dev.2, python: 0.1.13.dev2
if [[ "$RAW_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+)-([0-9]+)-g[a-f0-9]+$ ]]; then
    BASE_VERSION="${BASH_REMATCH[1]}"
    DEV_NUM="${BASH_REMATCH[2]}"
    RUST_VERSION="${BASE_VERSION}-dev.${DEV_NUM}"
    PYTHON_VERSION="${BASE_VERSION}.dev${DEV_NUM}"
else
    RUST_VERSION="$RAW_VERSION"
    PYTHON_VERSION="$RAW_VERSION"
fi
echo "Building version: Rust=$RUST_VERSION, Python=$PYTHON_VERSION (from git: $RAW_VERSION)"

# Update Cargo.toml with the Rust-compatible version
sed -i.bak "s/^version = .*/version = \"$RUST_VERSION\"/" Cargo.toml

# Create dist directory
mkdir -p dist

build_linux_x86_64() {
    echo "=== Building Linux x86_64 wheels ==="
    docker run --rm \
        --entrypoint /bin/bash \
        -v "$PROJECT_DIR":/io \
        $CACHE_MOUNTS \
        -v "$CACHE_DIR/cargo-target-manylinux-x86_64":/io/target \
        -w /io \
        ghcr.io/pyo3/maturin \
        -c "yum install -y openssl-devel perl-IPC-Cmd perl-Time-Piece && maturin build --release --out dist --interpreter 3.10 3.11 3.12 3.13 3.14"
}

build_linux_aarch64() {
    echo "=== Building Linux aarch64 wheels ==="
    # Requires Docker with QEMU configured for ARM emulation
    # On macOS with Docker Desktop, this should work out of the box
    docker run --rm --platform linux/arm64 \
        --entrypoint /bin/bash \
        -v "$PROJECT_DIR":/io \
        $CACHE_MOUNTS \
        -v "$CACHE_DIR/cargo-target-manylinux-aarch64":/io/target \
        -w /io \
        ghcr.io/pyo3/maturin \
        -c "yum install -y openssl-devel perl-IPC-Cmd perl-Time-Piece && maturin build --release --out dist --interpreter 3.10 3.11 3.12 3.13 3.14"
}

build_musllinux_x86_64() {
    echo "=== Building musllinux x86_64 wheels ==="
    # Use PyPA musllinux image, install Rust and maturin
    docker run --rm \
        -v "$PROJECT_DIR":/io \
        $CACHE_MOUNTS \
        -v "$CACHE_DIR/cargo-target-musllinux-x86_64":/io/target \
        -w /io \
        -e CARGO_HOME=/root/.cargo \
        -e RUSTUP_HOME=/root/.rustup \
        quay.io/pypa/musllinux_1_2_x86_64 \
        /bin/sh -c "apk add --no-cache openssl-dev perl curl && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source /root/.cargo/env && /opt/python/cp312-cp312/bin/pip install maturin && /opt/python/cp312-cp312/bin/maturin build --release --out dist --interpreter 3.10 3.11 3.12 3.13 3.14"
}

build_musllinux_aarch64() {
    echo "=== Building musllinux aarch64 wheels ==="
    docker run --rm --platform linux/arm64 \
        -v "$PROJECT_DIR":/io \
        $CACHE_MOUNTS \
        -v "$CACHE_DIR/cargo-target-musllinux-aarch64":/io/target \
        -w /io \
        -e CARGO_HOME=/root/.cargo \
        -e RUSTUP_HOME=/root/.rustup \
        quay.io/pypa/musllinux_1_2_aarch64 \
        /bin/sh -c "apk add --no-cache openssl-dev perl curl && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source /root/.cargo/env && /opt/python/cp312-cp312/bin/pip install maturin && /opt/python/cp312-cp312/bin/maturin build --release --out dist --interpreter 3.10 3.11 3.12 3.13 3.14"
}

build_macos_native() {
    echo "=== Building macOS wheels (native architecture) ==="
    # Requires maturin installed: pip install maturin
    if command -v maturin &> /dev/null; then
        maturin build --release --out dist --interpreter 3.10 3.11 3.12 3.13 3.14
    else
        echo "maturin not found. Install with: pip install maturin"
        exit 1
    fi
}

build_macos_universal() {
    echo "=== Building macOS universal2 wheels ==="
    # Builds fat binary supporting both Intel and Apple Silicon
    if command -v maturin &> /dev/null; then
        rustup target add x86_64-apple-darwin aarch64-apple-darwin 2>/dev/null || true
        maturin build --release --out dist --target universal2-apple-darwin --interpreter 3.10 3.11 3.12 3.13 3.14
    else
        echo "maturin not found. Install with: pip install maturin"
        exit 1
    fi
}

case "${1:-all}" in
    linux)
        build_linux_x86_64
        build_linux_aarch64
        ;;
    linux-x86)
        build_linux_x86_64
        ;;
    linux-arm)
        build_linux_aarch64
        ;;
    musllinux)
        build_musllinux_x86_64
        build_musllinux_aarch64
        ;;
    musllinux-x86)
        build_musllinux_x86_64
        ;;
    musllinux-arm)
        build_musllinux_aarch64
        ;;
    macos)
        build_macos_native
        ;;
    macos-universal)
        build_macos_universal
        ;;
    all)
        build_linux_x86_64
        build_linux_aarch64
        build_musllinux_x86_64
        build_musllinux_aarch64
        build_macos_native
        ;;
    *)
        echo "Usage: $0 [linux|linux-x86|linux-arm|musllinux|musllinux-x86|musllinux-arm|macos|macos-universal|all]"
        # Restore original Cargo.toml
        mv Cargo.toml.bak Cargo.toml 2>/dev/null || true
        exit 1
        ;;
esac

# Restore original Cargo.toml
mv Cargo.toml.bak Cargo.toml 2>/dev/null || true

echo ""
echo "=== Built wheels (version $PYTHON_VERSION) ==="
ls -la dist/*.whl 2>/dev/null || echo "No wheels found in dist/"
