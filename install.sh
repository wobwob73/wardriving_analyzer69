#!/bin/bash
#
# Silver Streak Analyzer (SSA) - Single-Shot Installer
# Run: ./install.sh
# Then: ./run.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║    ⚡ SILVER STREAK ANALYZER - SINGLE-SHOT INSTALLER     ║"
echo "║                      Version 4.6.1                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo -e "${YELLOW}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python 3 not found. Please install Python 3.10 or later.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  Found Python $PYTHON_VERSION"

# Check if version is >= 3.10
MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo -e "${RED}ERROR: Python 3.10 or later required. Found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Python $PYTHON_VERSION OK${NC}"

# Create virtual environment
echo -e "${YELLOW}[2/5] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "  Removing existing venv..."
    rm -rf venv
fi

$PYTHON_CMD -m venv venv
echo -e "${GREEN}  ✓ Virtual environment created${NC}"

# Activate venv and upgrade pip
echo -e "${YELLOW}[3/5] Upgrading pip...${NC}"
source venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}  ✓ pip upgraded${NC}"

# Install dependencies
echo -e "${YELLOW}[4/5] Installing dependencies (this may take 1-2 minutes)...${NC}"
pip install -r backend/requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ All dependencies installed${NC}"
else
    echo -e "${RED}ERROR: Failed to install dependencies${NC}"
    echo -e "${YELLOW}Trying with --only-binary flag...${NC}"
    pip install --only-binary :all: -r backend/requirements.txt
fi

# Create data directories
echo -e "${YELLOW}[5/5] Creating directories...${NC}"
mkdir -p data/uploads data/exports logs
echo -e "${GREEN}  ✓ Directories created${NC}"

# Create run script
cat > run.sh << 'RUNSCRIPT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source venv/bin/activate
cd backend
python app.py "$@"
RUNSCRIPT
chmod +x run.sh

# Create environment template
cat > .env << 'ENVFILE'
# Silver Streak Analyzer Configuration

# Flask Server
FLASK_PORT=5000
FLASK_DEBUG=False

# MQTT Broker (optional - for live data ingestion)
MQTT_HOST=
MQTT_PORT=1883
MQTT_USER=
MQTT_PASS=
ENVFILE

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗"
echo -e "║     ⚡ SILVER STREAK ANALYZER - INSTALLATION COMPLETE!    ║"
echo -e "╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To start the analyzer:"
echo -e "  ${BLUE}./run.sh${NC}"
echo ""
echo -e "Then open in browser:"
echo -e "  ${BLUE}http://localhost:5000${NC}"
echo ""
echo -e "Features:"
echo -e "  • RF Analysis (Wardriving) - Map/Table view"
echo -e "  • GNSS Link - Multi-receiver variance analysis"
echo ""
echo -e "Optional MQTT configuration:"
echo -e "  Edit ${YELLOW}.env${NC} to set MQTT_HOST for live data"
echo ""
