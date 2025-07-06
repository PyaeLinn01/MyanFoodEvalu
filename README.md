# Myanmar Food LLM Evaluator

This system evaluates Large Language Models (LLMs) like Google's Gemini and DeepSeek on their knowledge of Myanmar food using multiple choice questions about Mohinga (မုန့်ဟင်းခါး).

## Features

- Tests LLM knowledge on Myanmar food culture
- Multiple choice questions in Myanmar language
- Scoring system: +1 for correct answers, -1 for incorrect answers
- Detailed JSON output with comprehensive results
- Support for Google Gemini API and DeepSeek API
- Compare performance between different LLM models

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get API Keys**
   - **Google Gemini**: Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - **DeepSeek via OpenRouter**: Go to [OpenRouter](https://openrouter.ai/keys)
   - Create API keys and copy them

3. **Configure Environment**
   - Copy `env_template.txt` to `.env`:
     ```bash
     cp env_template.txt .env
     ```
   - Edit `.env` and add your actual API keys:
     ```
     GOOGLE_API_KEY=your_actual_gemini_api_key
     OPENROUTER_API_KEY=your_actual_openrouter_api_key
     ```
   - **Important**: The `.env` file is in `.gitignore` to keep your API keys private

## Usage

### Quick Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the evaluations:**
   ```bash
   python myanmar_food_evaluator.py # Gemini evaluation
   python myanFood_deepseek.py      # DeepSeek evaluation
   ```

## Output

The system will:
1. Ask each question to the LLM
2. Compare answers with correct responses
3. Calculate scores and accuracy
4. Save detailed results to `evaluation_results.json`

### Sample Output JSON Structure

```json
{
  "total_questions": 6,
  "correct_answers": 4,
  "incorrect_answers": 2,
  "accuracy_percentage": 66.67,
  "total_score": 2,
  "max_possible_score": 6,
  "detailed_results": [
    {
      "question_id": "Q1",
      "question": "မုန့်ဟင်းခါးမှာ အသုံးများသည့် မုန့်အမျိုးအစားက ဘာလဲ။",
      "llm_answer": "b",
      "correct_answer": "b",
      "is_correct": true,
      "score": 1,
      "explanation": "မုန့်ဟင်းခါးတွင် ဆန်မုန့်ကို အသုံးများသည်။"
    }
  ]
}
```

## Questions Included

The system tests knowledge on:
1. **Rice noodles** - Most common noodle type in Mohinga
2. **Fish** - Primary protein ingredient
3. **Breakfast** - Typical meal time for Mohinga
4. **Savory taste** - Characteristic flavor profile
5. **Fried chicken** - Ingredient typically not included
6. **Boiling** - Primary cooking method

## Scoring System

- **Correct Answer**: +1 point
- **Incorrect Answer**: -1 point
- **Maximum Score**: 6 points (all correct)
- **Minimum Score**: -6 points (all incorrect)

## Customization

To add more questions or modify existing ones, edit the `questions` list in the `MyanmarFoodEvaluator` class in `myanmar_food_evaluator.py`.

## Requirements

- Python 3.7+
- Google Gemini API access (optional)
- DeepSeek API access via OpenRouter (optional)
- Internet connection for API calls

### Testing Your APIs

Before running the full evaluation, test your APIs:

```bash
python test_both_apis.py
```

This will:
- Test both Gemini and DeepSeek APIs
- Verify your API keys are valid
- Test basic API functionality
- Test Myanmar language support
- Provide detailed error messages if something goes wrong

## Error Handling

The system includes error handling for:
- Missing API key
- API rate limits
- Network connectivity issues
- Invalid responses from LLM

## License

This project is open source and available under the MIT License. 