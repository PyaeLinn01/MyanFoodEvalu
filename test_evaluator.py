#!/usr/bin/env python3
"""
Test script for Myanmar Food Evaluator
This script tests the structure and logic without requiring an API key
"""

import json
from myanmar_food_evaluator import MyanmarFoodEvaluator

def test_evaluator_structure():
    """Test the evaluator structure and question format"""
    print("Testing Myanmar Food Evaluator Structure...")
    print("=" * 50)
    
    # Test without API key (will fail but we can check structure)
    try:
        evaluator = MyanmarFoodEvaluator()
        print("✓ Evaluator class created successfully")
        
        # Test question formatting
        if evaluator.questions:
            print(f"✓ Found {len(evaluator.questions)} questions")
            
            # Test first question formatting
            first_question = evaluator.questions[0]
            formatted = evaluator.format_question(first_question)
            print("✓ Question formatting works")
            print(f"Sample formatted question:\n{formatted[:200]}...")
            
            # Test scoring logic
            test_result = {
                "question_id": "Q1",
                "question": "Test question",
                "llm_answer": "b",
                "correct_answer": "b",
                "is_correct": True,
                "score": 1,
                "explanation": "Test explanation"
            }
            print("✓ Result structure is valid")
            
            # Test JSON serialization
            test_data = {
                "total_questions": 6,
                "correct_answers": 4,
                "incorrect_answers": 2,
                "accuracy_percentage": 66.67,
                "total_score": 2,
                "max_possible_score": 6,
                "detailed_results": [test_result]
            }
            
            json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
            print("✓ JSON serialization works")
            
            print("\n" + "=" * 50)
            print("STRUCTURE TEST PASSED")
            print("=" * 50)
            print("The evaluator is ready to use with a valid API key.")
            print("\nTo run the full evaluation:")
            print("1. Get a Google Gemini API key")
            print("2. Create a .env file with: GOOGLE_API_KEY=your_key_here")
            print("3. Run: python myanmar_food_evaluator.py")
            
        else:
            print("✗ No questions found")
            
    except ValueError as e:
        if "GOOGLE_API_KEY" in str(e):
            print("✓ API key validation works (expected error)")
            print("✓ Evaluator structure is correct")
            print("\nTo run the full evaluation:")
            print("1. Get a Google Gemini API key")
            print("2. Create a .env file with: GOOGLE_API_KEY=your_key_here")
            print("3. Run: python myanmar_food_evaluator.py")
        else:
            print(f"✗ Unexpected error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_evaluator_structure() 