# Myanmar Food LLM Evaluator & Image Preprocessing

This repository provides tools to evaluate Large Language Models (LLMs) on Myanmar food knowledge using multiple choice questions, and to preprocess Myanmar food images for machine learning.

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MyanFoodEvalu.git
cd MyanFoodEvalu
```

### 2. Create and Activate a Python Environment
- **Using Conda:**
  ```bash
  conda create -n myanfood python=3.9
  conda activate myanfood
  ```
- **Or using venv:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get Your API Keys
- Sign up at [OpenRouter](https://openrouter.ai/keys) to get your API key.
- (Optional) Get Google Gemini API key if you want to use Gemini models.

### 5. Configure API Keys
- Copy `env_template.txt` to `.env`:
  ```bash
  cp env_template.txt .env
  ```
- Edit `.env` and paste your API keys:
  ```
  OPENROUTER_API_KEY=your_openrouter_api_key
  GOOGLE_API_KEY=your_gemini_api_key  # (optional)
  ```

## Evaluating LLMs on Myanmar Food Knowledge
- Each model file (e.g. `gemini.py`, `claude-3.7-sonnet.py`, etc.) evaluates the questions in `mote_hin_khar.json`.
- To run an evaluation, simply execute the desired model script:
  ```bash
  python gemini.py
  # or
  python claude-3.7-sonnet.py
  ```
- The evaluation results will be saved in the `mc_results_json` folder.

## Image Preprocessing
- To preprocess Myanmar food images, run:
  ```bash
  python image_preprocessing/preprocess.py
  ```
- Processed images will be saved under the `processed_images` directory.

## Notes
- **API keys are required** for LLM evaluation. Put them in your `.env` file.
- Make sure to get your own API keys from OpenRouter (and optionally Google Gemini).
- This repo is for evaluating LLMs on Myanmar food knowledge and preparing food image datasets for machine learning. 