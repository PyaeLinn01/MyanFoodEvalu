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
        api_key = os.getenv('Groq')
        if not api_key:
            raise ValueError("Groq API not found in environment variables")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = "x-ai/grok-4"
        
        # Define the questions and correct answers
        self.questions = [
            {
                "id": "Q1",
                "question": "မုန့်ဟင်းခါးမှာ အသုံးများသည့် မုန့်အမျိုးအစားက ဘာလဲ။",
                "options": {
                    "a": "ဂျုံမုန့်",
                    "b": "ဆန်မုန်",
                    "c": "ဝက်သားမုန့်",
                    "d": "တိုဖူးမုန့်"
                },
                "correct_answer": "b",
                "explanation": "မုန့်ဟင်းခါးတွင် ဆန်မုန့်ကို အသုံးများသည်။"
            },
            {
                "id": "Q2",
                "question": "မုန့်ဟင်းခါးရဲ့ အဓိက အသားပါဝင်ပစ္စည်းက ဘာလဲ။",
                "options": {
                    "a": "ဝက်သား",
                    "b": "ကြက်သား",
                    "c": "ငါး",
                    "d": "ငှက်သား"
                },
                "correct_answer": "c",
                "explanation": "မုန့်ဟင်းခါးတွင် ငါးကို အဓိကအသားအဖြစ် သုံးသည်။"
            },
            {
                "id": "Q3",
                "question": "မုန့်ဟင်းခါးကို ယေဘူယျအားဖြင့် ဘယ်အချိန်အတွက် စားသုံးလေ့ရှိသလဲ။",
                "options": {
                    "a": "နေ့လယ်စာ",
                    "b": "ညစာ",
                    "c": "နံနက်စာ",
                    "d": "နေ့လယ်ဝန်းစာ"
                },
                "correct_answer": "c",
                "explanation": "မုန့်ဟင်းခါးကို နံနက်စာအဖြစ် စားသုံးလေ့ရှိသည်။"
            },
            {
                "id": "Q4",
                "question": "မုန့်ဟင်းခါးရဲ့ အရသာကို အကြမ်းဖျင်းဖော်ပြလျှင် ဘယ်လိုနည်း။",
                "options": {
                    "a": "ချဉ်ချဉ်",
                    "b": "ချိုချို",
                    "c": "စပ်စပ်",
                    "d": "ခပ်ဆိမ့်ဆိမ့်"
                },
                "correct_answer": "d",
                "explanation": "မုန့်ဟင်းခါးသည် ခပ်ဆိမ့်ဆိမ့်အရသာရှိသည်။"
            },
            {
                "id": "Q5",
                "question": "မုန့်ဟင်းခါးတွင် ပုံမှန်အားဖြင့် မပါဝင်သည့် အကြော်ပစ္စည်းက ဘာလဲ။",
                "options": {
                    "a": "ဘူးသီးကြော်",
                    "b": "ဘယာကြော်",
                    "c": "ကြက်သားကြော်",
                    "d": "ကြက်သွန်ကြော်"
                },
                "correct_answer": "c",
                "explanation": "မုန့်ဟင်းခါးတွင် ကြက်သားကြော်ကို မသုံးလေ့မရှိ။"
            },
            {
                "id": "Q6",
                "question": "မုန့်ဟင်းခါးရဲ့ အဓိကချက်ပြုတ်နည်းက ဘာလဲ။",
                "options": {
                    "a": "ကင်ခြောက်ခြင်း",
                    "b": "ပြုတ်ခြင်း",
                    "c": "ဆားငန်ခြင်း",
                    "d": "ထန်းသွေးထောင်းခြင်း"
                },
                "correct_answer": "b",
                "explanation": "မုန့်ဟင်းခါးကို ပြုတ်ခြင်းနည်းဖြင့် ချက်ပြုတ်သည်။"
            }
        ]

    def format_question(self, question_data: Dict[str, Any]) -> str:
        """Format a question for the LLM"""
        question_text = f"{question_data['id']}: {question_data['question']}\n"
        options_text = ""
        for key, value in question_data['options'].items():
            options_text += f"{key}) {value}\n"
        
        return question_text + options_text + "\nPlease answer with only the letter (a, b, c, or d)."

    def call_deepseek_api(self, prompt: str) -> str:
        """Call x-ai/grok-4 API via OpenRouter and return response"""
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
            "model": "x-ai/grok-4",
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
        print("x-ai/grok-4 SUMMARY")
        print("=" * 50)
        print(f"Model: {results['model']}")
        print(f"API Provider: {results['api_provider']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Incorrect Answers: {results['incorrect_answers']}")
        print(f"Accuracy: {results['accuracy_percentage']}%")
        print(f"Total Score: {results['total_score']}/{results['max_possible_score']}")
        
        # Save results to JSON file
        with open('x-ai_grok-4.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to 'x-ai_grok-4.json'")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

if __name__ == "__main__":
    main()
