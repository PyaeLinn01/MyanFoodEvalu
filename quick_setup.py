#!/usr/bin/env python3
"""
Quick Setup Script for Myanmar Food LLM Evaluator
Helps users set up their environment quickly
"""

import os
import shutil

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = ".env"
    template_file = "env_template.txt"
    
    if os.path.exists(env_file):
        print("✓ .env file already exists")
        return True
    
    if not os.path.exists(template_file):
        print("❌ env_template.txt not found")
        return False
    
    try:
        shutil.copy(template_file, env_file)
        print("✓ Created .env file from template")
        print("⚠️  Please edit .env and add your actual API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_gitignore():
    """Check if .gitignore exists and contains .env"""
    gitignore_file = ".gitignore"
    
    if not os.path.exists(gitignore_file):
        print("⚠️  .gitignore file not found")
        return False
    
    try:
        with open(gitignore_file, 'r') as f:
            content = f.read()
            if '.env' in content:
                print("✓ .env is properly ignored in .gitignore")
                return True
            else:
                print("⚠️  .env is not in .gitignore")
                return False
    except Exception as e:
        print(f"❌ Error reading .gitignore: {e}")
        return False

def main():
    """Main setup function"""
    print("Myanmar Food LLM Evaluator - Quick Setup")
    print("=" * 50)
    
    # Check if .env exists or create it
    if create_env_file():
        print("\n📝 Next steps:")
        print("1. Edit .env file and add your API keys:")
        print("   - GOOGLE_API_KEY=your_actual_gemini_api_key")
        print("   - OPENROUTER_API_KEY=your_actual_openrouter_api_key")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Test APIs: python test_both_apis.py")
        print("4. Run evaluations: python myanmar_food_evaluator.py")
    else:
        print("\n❌ Setup failed")
    
    # Check .gitignore
    print("\n" + "-" * 30)
    check_gitignore()
    
    print("\n🔒 Security Note:")
    print("Your .env file contains API keys and is private.")
    print("It's automatically ignored by git to keep your keys safe.")

if __name__ == "__main__":
    main() 