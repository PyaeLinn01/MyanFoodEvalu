# Myanmar Food LLM Evaluator

This system evaluates Large Language Models (LLMs) like Google's Gemini on their knowledge of Myanmar food using multiple choice questions about Mohinga (မုန့်ဟင်းခါး).

## Features

- Tests LLM knowledge on Myanmar food culture
- Multiple choice questions in Myanmar language
- Scoring system: +1 for correct answers, -1 for incorrect answers
- Detailed JSON output with comprehensive results
- Support for Google Gemini API

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Google Gemini API Key**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the API key

3. **Configure Environment**
   - Create a `.env` file in the project root
   - Add your API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

### Quick Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key:**
   ```bash
   python setup_api.py
   ```

3. **Test API connection:**
   ```bash
   python apitest.py
   ```

4. **Run the evaluation:**
   ```bash
   python myanmar_food_evaluator.py
   ```

### Manual Setup
If you prefer to set up manually:

1. Create a `.env` file with your API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

2. Test the API:
   ```bash
   python apitest.py
   ```

3. Run the evaluation:
   ```bash
   python myanmar_food_evaluator.py
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
- Google Gemini API access
- Internet connection for API calls

## API Testing

The system includes several testing scripts:

- **`apitest.py`** - Tests if your API key works correctly
- **`test_evaluator.py`** - Tests the evaluator structure without API calls
- **`setup_api.py`** - Interactive setup for API key configuration

### Testing Your API Key

Before running the full evaluation, test your API key:

```bash
python apitest.py
```

This will:
- Verify your API key is valid
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