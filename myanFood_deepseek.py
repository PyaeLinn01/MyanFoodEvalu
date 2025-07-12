import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class MyanmarFoodDeepSeekEvaluator:
    def __init__(self):
        """Initialize the evaluator with OpenRouter API for DeepSeek"""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
        
        # Load questions from JSON file
        try:
            with open('mote_hin_khar.json', 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("mote_hin_khar.json file not found. Please ensure the file exists in the same directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing mote_hin_khar.json: {e}")
        except Exception as e:
            raise Exception(f"Error loading questions from mote_hin_khar.json: {e}")

    def format_question(self, question_data: Dict[str, Any]) -> str:
        """Format a question for the LLM"""
        question_text = f"{question_data['id']}: {question_data['question']}\n"
        options_text = ""
        for key, value in question_data['options'].items():
            options_text += f"{key}) {value}\n"
        
        return question_text + options_text + "\nPlease answer with only the letter (a, b, c, or d)."

    def call_deepseek_api(self, prompt: str) -> str:
        """Call DeepSeek API via OpenRouter and return response"""
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://myanfood-evaluator.com",  # Optional
                    "X-Title": "Myanmar Food Evaluator",  # Optional
                },
                extra_body={},
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            return completion.choices[0].message.content.strip()
                
        except Exception as e:
            raise Exception(f"API request failed: {e}")

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with the LLM"""
        formatted_question = self.format_question(question_data)
        
        prompt = f"""
You are being tested on your knowledge of Myanmar food. Please answer the following multiple choice question about Mohinga (မုန့်ဟင်းခါး):

{formatted_question}

Please respond with only the letter of your answer (a, b, c, or d).
"""

        try:
            llm_response = self.call_deepseek_api(prompt)
            llm_answer = llm_response.lower()
            
            # Extract just the letter from the response
            if llm_answer.startswith('answer:'):
                llm_answer = llm_answer.replace('answer:', '').strip()
            if llm_answer.startswith('the answer is'):
                llm_answer = llm_answer.replace('the answer is', '').strip()
            
            # Clean up the answer to get just the letter
            llm_answer = llm_answer.replace(')', '').replace('(', '').strip()
            
            # Check if answer is correct
            is_correct = llm_answer == question_data['correct_answer']
            score = 1 if is_correct else -1
            
            return {
                "question_id": question_data['id'],
                "question": question_data['question'],
                "llm_answer": llm_answer,
                "correct_answer": question_data['correct_answer'],
                "is_correct": is_correct,
                "score": score,
                "explanation": question_data['explanation']
            }
            
        except Exception as e:
            return {
                "question_id": question_data['id'],
                "question": question_data['question'],
                "llm_answer": "ERROR",
                "correct_answer": question_data['correct_answer'],
                "is_correct": False,
                "score": -1,
                "explanation": question_data['explanation'],
                "error": str(e)
            }

    def evaluate_all_questions(self) -> Dict[str, Any]:
        """Evaluate all questions and return comprehensive results"""
        results = []
        total_score = 0
        correct_count = 0
        
        print("Starting evaluation of Myanmar food knowledge with DeepSeek via OpenRouter...")
        print("=" * 50)
        
        for i, question in enumerate(self.questions, 1):
            print(f"Evaluating question {i}/{len(self.questions)}...")
            result = self.evaluate_single_question(question)
            results.append(result)
            
            total_score += result['score']
            if result['is_correct']:
                correct_count += 1
            
            print(f"Question {result['question_id']}: {'✓' if result['is_correct'] else '✗'}")
            print(f"LLM Answer: {result['llm_answer']}, Correct: {result['correct_answer']}")
            print("-" * 30)
        
        # Calculate statistics
        accuracy = (correct_count / len(self.questions)) * 100
        
        evaluation_summary = {
            "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "api_provider": "OpenRouter",
            "total_questions": len(self.questions),
            "correct_answers": correct_count,
            "incorrect_answers": len(self.questions) - correct_count,
            "accuracy_percentage": round(accuracy, 2),
            "total_score": total_score,
            "max_possible_score": len(self.questions),
            "detailed_results": results
        }
        
        return evaluation_summary

def main():
    """Main function to run the evaluation"""
    try:
        evaluator = MyanmarFoodDeepSeekEvaluator()
        results = evaluator.evaluate_all_questions()
        
        # Print summary
        print("\n" + "=" * 50)
        print("DEEPSEEK EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Model: {results['model']}")
        print(f"API Provider: {results['api_provider']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Incorrect Answers: {results['incorrect_answers']}")
        print(f"Accuracy: {results['accuracy_percentage']}%")
        print(f"Total Score: {results['total_score']}/{results['max_possible_score']}")
        
        # Save results to JSON file
        with open('deepseek_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to 'deepseek_evaluation_results.json'")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    main()
