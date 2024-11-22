echo "Updating system package list..."
sudo apt-get update

echo "Installing Tesseract OCR and Poppler utilities..."
sudo apt-get install -y tesseract-ocr tesseract-ocr-guj poppler-utils

if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Installing Python3..."
    sudo apt-get install -y python3 python3-pip
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Setup complete!"