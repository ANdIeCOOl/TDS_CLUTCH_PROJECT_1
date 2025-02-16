# /// script
# dependencies = [
#   "uv",
# ]
# ///
import subprocess

# Step 1: Set the URL to download the Python script
url = 'https://raw.githubusercontent.com/ANdIeCOOl/TDS-Project1-Ollama_FastAPI-/refs/heads/main/datagen.py'

# Step 2: Download the script using curl command
subprocess.run(['curl', '-O', url])

# Step 3: Define the email argument
email_argument = '23f1002382@ds.study.iitm.ac.in'

# Step 4: Run the downloaded script using uv
subprocess.run(['uv', 'run', 'datagen.py', email_argument])
