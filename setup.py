#!/usr/bin/env python3
"""
Setup script for Myanmar Food LLM Evaluator
"""

import os
import subprocess
import sys

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has API key"""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
            if "GOOGLE_API_KEY" in content:
                print("✓ .env file exists with API key")
                return True
            else:
                print("⚠️  .env file exists but no API key found")
                return False
    else:
        print("⚠️  .env file not found")
        return False

def create_env_template():
    """Create a template .env file"""
    env_content = """# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_api_key_here
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    print("✓ Created .env template file")
    print("⚠️  Please add your actual API key to the .env file")

def main():
    """Main setup function"""
    print("Myanmar Food LLM Evaluator Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check environment file
    if not check_env_file():
        create_env_template()
        print("\n📝 Next steps:")
        print("1. Get a Google Gemini API key from: https://makersuite.google.com/app/apikey")
        print("2. Edit the .env file and replace 'your_gemini_api_key_here' with your actual API key")
        print("3. Run: python myanmar_food_evaluator.py")
    else:
        print("\n🎉 Setup complete! You can now run:")
        print("python myanmar_food_evaluator.py")
    
    return True

if __name__ == "__main__":
    main() 