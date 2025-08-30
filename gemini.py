import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

class MyanmarFoodEvaluator:
    def __init__(self):
        """Initialize the evaluator with Gemini API"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load questions from JSON file
        try:
            with open('Coconut noodle VQA_text.json', 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Coconut noodle VQA_text.json file not found. Please ensure the file exists in the same directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing mote_hin_khar.json: {e}")
        except Exception as e:
            raise Exception(f"Error loading questions from mote_hin_khar.json: {e}")

    def format_question(self, question_data: Dict[str, Any]) -> str:
        """Format a question for the LLM"""
        question_text = f"{question_data['id']}: {question_data['question']}\n"
        options_text = ""
        for i, choice in enumerate(question_data['choices']):
            options_text += f"{i}) {choice}\n"

        return (
            question_text
            + options_text
            + "\nPlease answer with only the index number(s). If more than one is correct, reply with comma-separated indices, e.g., 0 or 1,2."
        )

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with the LLM"""
        formatted_question = self.format_question(question_data)
        
        prompt = f"""
You are being tested on your knowledge of Myanmar food. Please answer the following multiple choice question:

{formatted_question}

Please respond with only the index number(s). If multiple are correct, separate with commas (e.g., 0 or 1,2).
"""

        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip() if hasattr(response, 'text') else str(response)

            # Extract all integers from response as selected indices
            import re as _re
            parsed_tokens = [_re.sub(r"[^0-9]", "", tok) for tok in _re.split(r"[\s,]+", raw_text) if tok]
            parsed_indices = [int(x) for x in parsed_tokens if x != ""]

            chosen_set = set(parsed_indices)
            correct_set = set(int(x) for x in question_data.get('answer', []))

            # Scoring rules:
            # 1.0 if chosen exactly matches all correct answers
            # 0.5 if there is any overlap but not full match
            # 0.0 otherwise
            if not chosen_set:
                score = 0.0
                is_correct = False
                partial = False
            elif chosen_set == correct_set:
                score = 1.0
                is_correct = True
                partial = False
            elif chosen_set & correct_set:
                score = 0.5
                is_correct = False
                partial = True
            else:
                score = 0.0
                is_correct = False
                partial = False

            llm_answer = ",".join(str(i) for i in sorted(chosen_set)) if chosen_set else ""
            correct_answer_str = ",".join(str(i) for i in sorted(correct_set))

            return {
                "question_id": question_data['id'],
                "question": question_data['question'],
                "llm_answer": llm_answer,
                "correct_answer": correct_answer_str,
                "is_correct": is_correct,
                "partial_correct": partial,
                "score": score,
                "choices": question_data['choices']
            }
            
        except Exception as e:
            return {
                "question_id": question_data['id'],
                "question": question_data['question'],
                "llm_answer": "ERROR",
                "correct_answer": ",".join(str(i) for i in question_data.get('answer', [])),
                "is_correct": False,
                "partial_correct": False,
                "score": 0.0,
                "choices": question_data['choices'],
                "error": str(e)
            }

    def evaluate_all_questions(self) -> Dict[str, Any]:
        """Evaluate all questions and return comprehensive results"""
        results = []
        total_score = 0.0
        correct_count = 0
        
        print("Starting evaluation of Myanmar food knowledge...")
        print("=" * 50)
        
        for i, question in enumerate(self.questions, 1):
            print(f"Evaluating question {i}/{len(self.questions)}...")
            result = self.evaluate_single_question(question)
            results.append(result)
            
            total_score += float(result['score'])
            if result['is_correct']:
                correct_count += 1
            
            print(f"Question {result['question_id']}: {'✓' if result['is_correct'] else '✗'}")
            print(f"LLM Answer: {result['llm_answer']}, Correct: {result['correct_answer']}")
            print("-" * 30)
        
        # Calculate statistics
        accuracy = (correct_count / len(self.questions)) * 100 if self.questions else 0.0
        
        evaluation_summary = {
            "total_questions": len(self.questions),
            "correct_answers": correct_count,
            "incorrect_answers": len(self.questions) - correct_count,
            "accuracy_percentage": round(accuracy, 2),
            "total_score": round(total_score, 2),
            "max_possible_score": float(len(self.questions)),
            "detailed_results": results,
            "model": "gemini-2.5-flash",
            "api_provider": "Google"
        }
        
        return evaluation_summary

def main():
    """Main function to run the evaluation"""
    try:
        evaluator = MyanmarFoodEvaluator()
        results = evaluator.evaluate_all_questions()
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Incorrect Answers: {results['incorrect_answers']}")
        print(f"Accuracy: {results['accuracy_percentage']}%")
        print(f"Total Score: {results['total_score']}/{results['max_possible_score']}")
        
        # Save results to JSON file
        with open('mc_results_json/evaluation_results_gemini.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to 'mc_results_json/evaluation_results_gemini.json'")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    main() 