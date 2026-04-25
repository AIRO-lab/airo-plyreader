apt_packages=(
    libgomp1       # Open3D / OpenMP runtime
)

echo "Installing system dependencies..."
apt update -qq
apt install -y --no-install-recommends "${apt_packages[@]}"
rm -rf /var/lib/apt/lists/*
echo "Done."
