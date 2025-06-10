# minellm

`minellm` demonstrates a privacy-aware approach to prompt tuning. The goal
is to show how sensitive text can be sanitized and then used in a training
pipeline. The project is organized around two main scripts:

- **advanced_prompt_tuning.py** – utility classes and helpers for advanced prompt tuning experiments.
- **privacy_ai_pipeline.py** – an example pipeline that combines secure data
  processing with prompt tuning.

The scripts are not fully featured but should run once the required
packages are installed. Use the provided `requirements.txt` to install the
dependencies:

```bash
pip install -r requirements.txt
```

Both scripts perform dependency checks and will print a helpful error
message if something important is missing.

## Quick demo

You can run `purpose_demo.py` without any optional libraries to see how the
project sanitizes text:

```bash
python purpose_demo.py
```
