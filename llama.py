import os
import json
import re
import requests
from typing import List, Dict, Any

# Ollama settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


class MyanmarFoodLlamaEvaluator:
    def __init__(self):
        """Initialize the evaluator for local Ollama Llama model"""
        # Load questions from JSON file
        try:
            with open('Coconut noodle VQA_text.json', 'r', encoding='utf-8') as f:
                self.questions = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Coconut noodle VQA_text.json file not found. Please ensure the file exists in the same directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing Coconut noodle VQA_text.json: {e}")
        except Exception as e:
            raise Exception(f"Error loading questions: {e}")

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

    def call_ollama(self, prompt: str) -> str:
        """Call local Ollama chat API and return the assistant message content"""
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are being evaluated. Reply ONLY with the index number(s). If multiple indices are correct, reply as comma-separated numbers like 0 or 1,2. Do not include any words or explanations."},
                {"role": "user", "content": prompt}
            ],
            # Keep responses short and deterministic
            "options": {
                "temperature": 0.0,
                "num_predict": 32  # small cap; indices are short
            },
            # Ensure JSON (non-streaming) response
            "stream": False,
        }
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/chat returns the last message in data["message"]["content"]
            # or in some setups, an array of messages. Handle both.
            if isinstance(data, dict) and "message" in data and "content" in data["message"]:
                return str(data["message"]["content"]).strip()
            # Fallback: some servers return { "messages": [ ... ] }
            if "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                # get last assistant message
                for m in reversed(data["messages"]):
                    if m.get("role") == "assistant" and "content" in m:
                        return str(m["content"]).strip()
            # If chat payload did not yield assistant content, try /api/generate as fallback
            gen_payload = {
                "model": OLLAMA_MODEL,
                "prompt": (
                    "You are being evaluated. Reply ONLY with the index number(s). "
                    "If multiple indices are correct, reply as comma-separated numbers like 0 or 1,2. "
                    "Do not include any words or explanations.\n\n" + prompt
                ),
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 32
                }
            }
            gen_url = OLLAMA_URL.replace("/api/chat", "/api/generate")
            gen_resp = requests.post(gen_url, json=gen_payload, timeout=120)
            gen_resp.raise_for_status()
            gen_data = gen_resp.json()
            if isinstance(gen_data, dict) and "response" in gen_data:
                return str(gen_data["response"]).strip()
            # As a last resort, stringify the response
            return str(data)
        except requests.RequestException as e:
            raise Exception(f"Ollama request failed: {e}")
        except ValueError as e:
            # Include raw text (truncated) to help diagnose streaming/malformed JSON
            raw = ''
            try:
                raw = resp.text[:400]
            except Exception:
                pass
            raise Exception(f"Ollama response parse error: {e}; raw=<{raw}>")
        except Exception as e:
            raise Exception(f"Ollama call error: {e}")

    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question with the LLM"""
        formatted_question = self.format_question(question_data)

        prompt = f"""
You are being tested on your knowledge of Myanmar food. Please answer the following multiple choice question:

{formatted_question}

Please respond with only the index number(s). If multiple are correct, separate with commas (e.g., 0 or 1,2).
"""

        try:
            llm_response = self.call_ollama(prompt)

            # Extract all integers from response as selected indices
            parsed_tokens = [re.sub(r"[^0-9]", "", tok) for tok in re.split(r"[\s,]+", llm_response.strip()) if tok]
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

        print("Starting evaluation of Myanmar food knowledge with Ollama (llama3.2:latest)...")
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
            "model": OLLAMA_MODEL,
            "api_provider": "Ollama",
            "total_questions": len(self.questions),
            "correct_answers": correct_count,
            "incorrect_answers": len(self.questions) - correct_count,
            "accuracy_percentage": round(accuracy, 2),
            "total_score": round(total_score, 2),
            "max_possible_score": float(len(self.questions)),
            "detailed_results": results
        }

        return evaluation_summary


def main():
    """Main function to run the evaluation"""
    try:
        evaluator = MyanmarFoodLlamaEvaluator()
        results = evaluator.evaluate_all_questions()

        # Print summary
        print("\n" + "=" * 50)
        print("LLAMA (Ollama) EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Model: {results['model']}")
        print(f"API Provider: {results['api_provider']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Incorrect Answers: {results['incorrect_answers']}")
        print(f"Accuracy: {results['accuracy_percentage']}%")
        print(f"Total Score: {results['total_score']}/{results['max_possible_score']}")

        # Save results to JSON file
        os.makedirs('mc_results_json', exist_ok=True)
        with open('mc_results_json/llama.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to 'mc_results_json/llama.json'")

        return results

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None


if __name__ == "__main__":
    main()