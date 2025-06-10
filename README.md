diff --git a/README.md b/README.md
index 665172ff517d4b1c9eb014ea45d28a20acf3f668..5e074045ab212c2d79063f9f265946754c0eb0d5 100644
--- a/README.md
+++ b/README.md
@@ -1,5 +1,39 @@
 # minellm
 
-This repository contains experimental code snippets for advanced prompt tuning
-and a privacy-first AI pipeline using Ollama. The code is currently incomplete
-and provided for reference only.
+`minellm` demonstrates a privacy-aware approach to prompt tuning. The goal
+is to show how sensitive text can be sanitized and then used in a training
+pipeline. The project is organized around two main scripts:
+
+- **advanced_prompt_tuning.py** – utility classes and helpers for advanced prompt tuning experiments.
+- **privacy_ai_pipeline.py** – an example pipeline that combines secure data
+  processing with prompt tuning.
+
+The scripts are not fully featured but should run once the required
+packages are installed. Use the provided `requirements.txt` to install the
+dependencies:
+
+```bash
+pip install -r requirements.txt
+```
+
+Both scripts perform dependency checks and will print a helpful error
+message if something important is missing.
+
+## Quick demo
+
+You can run `purpose_demo.py` without any optional libraries to see how the
+project sanitizes text:
+
+```bash
+python purpose_demo.py
+```
+
+## Streamlit interface
+
+A simple Streamlit app lets you experiment with the text sanitization step in a browser. Once `streamlit` is installed run:
+
+```bash
+streamlit run streamlit_app.py
+```
+
+Enter text in the text area and the sanitized result will appear below.
