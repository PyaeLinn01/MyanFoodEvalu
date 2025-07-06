#!/usr/bin/env python3
"""
API Test Script for Myanmar Food LLM Evaluator
Tests if the Google Gemini API key is working correctly
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

def test_api_connection():
    """Test the API connection with a simple question"""
    print("Testing Google Gemini API Connection...")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please create a .env file with your API key:")
        print("GOOGLE_API_KEY=your_gemini_api_key_here")
        return False
    
    if api_key == "your_gemini_api_key_here":
        print("❌ Please replace 'your_gemini_api_key_here' with your actual API key")
        return False
    
    print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✓ API configured successfully")
        
        # Test with a simple question
        test_prompt = """
        Please answer this simple question about Myanmar food:
        
        What is the most common noodle type used in Mohinga (မုန့်ဟင်းခါး)?
        a) Wheat noodles
        b) Rice noodles
        c) Egg noodles
        d) Bean noodles
        
        Please respond with only the letter (a, b, c, or d).
        """
        
        print("Testing API with a simple question...")
        response = model.generate_content(test_prompt)
        
        print("✓ API response received successfully")
        print(f"Response: {response.text.strip()}")
        
        # Test Myanmar language support
        myanmar_prompt = """
        မုန့်ဟင်းခါးမှာ အသုံးများသည့် မုန့်အမျိုးအစားက ဘာလဲ။
        a) ဂျုံမုန့်
        b) ဆန်မုန်
        c) ဝက်သားမုန့်
        d) တိုဖူးမုန့်
        
        Please answer with only the letter (a, b, c, or d).
        """
        
        print("\nTesting Myanmar language support...")
        myanmar_response = model.generate_content(myanmar_prompt)
        
        print("✓ Myanmar language test successful")
        print(f"Myanmar Response: {myanmar_response.text.strip()}")
        
        print("\n" + "=" * 50)
        print("✅ API TEST PASSED")
        print("=" * 50)
        print("Your API key is working correctly!")
        print("You can now run the full evaluation:")
        print("python myanmar_food_evaluator.py")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. Network connectivity issues")
        print("3. API rate limits exceeded")
        print("4. API service temporarily unavailable")
        return False

def test_api_with_simple_question():
    """Test API with a very simple question to check basic functionality"""
    print("\nRunning basic API functionality test...")
    
    try:
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key or api_key == "your_gemini_api_key_here":
            print("❌ No valid API key found")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Very simple test
        simple_prompt = "What is 2 + 2? Answer with just the number."
        response = model.generate_content(simple_prompt)
        
        print(f"✓ Basic test successful. Response: {response.text.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main function to run API tests"""
    print("Myanmar Food LLM Evaluator - API Test")
    print("=" * 50)
    
    # First test basic functionality
    if test_api_with_simple_question():
        print("✓ Basic API functionality confirmed")
    else:
        print("❌ Basic API test failed")
        return
    
    # Then test with Myanmar food question
    if test_api_connection():
        print("\n🎉 All tests passed! Your API is ready to use.")
    else:
        print("\n❌ API test failed. Please check your configuration.")

if __name__ == "__main__":
    main() 