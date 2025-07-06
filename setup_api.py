#!/usr/bin/env python3
"""
API Setup Helper for Myanmar Food LLM Evaluator
Helps users set up their Google Gemini API key
"""

import os
import sys

def create_env_file():
    """Create or update the .env file"""
    env_file = ".env"
    
    print("Setting up Google Gemini API key...")
    print("=" * 50)
    
    # Check if .env already exists
    if os.path.exists(env_file):
        print("⚠️  .env file already exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if "GOOGLE_API_KEY" in content and "your_gemini_api_key_here" not in content:
                print("✓ API key already configured")
                return True
    
    # Get API key from user
    print("\nTo get your Google Gemini API key:")
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated API key")
    print("\n" + "-" * 50)
    
    api_key = input("Enter your Google Gemini API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    if api_key == "your_gemini_api_key_here":
        print("❌ Please enter your actual API key, not the placeholder")
        return False
    
    # Create or update .env file
    env_content = f"GOOGLE_API_KEY={api_key}\n"
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✓ API key saved to .env file")
    return True

def test_setup():
    """Test if the setup is working"""
    print("\nTesting API setup...")
    
    try:
        # Import required modules
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        # Load environment
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ API key not found in .env file")
            return False
        
        # Test API connection
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Simple test
        response = model.generate_content("Say 'Hello' if you can hear me.")
        
        if response.text:
            print("✓ API connection successful!")
            return True
        else:
            print("❌ No response from API")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Myanmar Food LLM Evaluator - API Setup")
    print("=" * 50)
    
    # Create .env file
    if create_env_file():
        print("\nTesting the setup...")
        if test_setup():
            print("\n🎉 Setup complete!")
            print("You can now run:")
            print("  python apitest.py     # Test API connection")
            print("  python myanmar_food_evaluator.py  # Run full evaluation")
        else:
            print("\n❌ Setup failed. Please check your API key and try again.")
    else:
        print("\n❌ Failed to create .env file")

if __name__ == "__main__":
    main() 