# ---------------------------------------------
# 🛠️ Project Initialization Steps
# Run this script after cloning the repository.
# It sets up the environment and downloads necessary resources.
# ---------------------------------------------

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy English model
python3 -m spacy download en_core_web_sm

# Download NLTK tokenizer data
python3 -m nltk.downloader punkt
