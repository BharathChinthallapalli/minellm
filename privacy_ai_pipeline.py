# -*- coding: utf-8 -*-
"""
Unified Enterprise AI Pipeline: Secure Training and Inference
================================================================

This script combines the concepts from advanced_prompt_tuning.py and
privacy_ai_pipeline.py into a single, runnable implementation.

Workflow:
1.  Raw data is analyzed by the OllamaSecurityGateway for PII and sensitivity.
2.  A SecureDataProcessor filters and sanitizes the data, creating a clean dataset.
3.  The AdvancedTrainer uses cutting-edge techniques (EWC, SAM) to train a
    prompt-tuned model on this secure dataset.
4.  The final, securely-trained model is saved and tested.
"""

import os
import gc
import json
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re
import logging
import time
import asyncio
import sys

missing_deps = []
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None
    missing_deps.append("numpy")

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None
    missing_deps.append("pandas")

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None
    missing_deps.append("requests")

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    missing_deps.append("torch")

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover - optional dependency
    Dataset = None
    missing_deps.append("datasets")

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
    )
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    Trainer = None
    DataCollatorForLanguageModeling = None
    EarlyStoppingCallback = None
    missing_deps.append("transformers")

try:
    from peft import (
        get_peft_model,
        PromptTuningConfig,
        PromptTuningInit,
        TaskType,
        PeftModel,
    )
except ImportError:  # pragma: no cover - optional dependency
    get_peft_model = None
    PromptTuningConfig = None
    PromptTuningInit = None
    TaskType = None
    PeftModel = None
    missing_deps.append("peft")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if missing_deps:
    if __name__ == "__main__":
        print(
            "This script requires additional dependencies: "
            + ", ".join(missing_deps)
        )
        print("Install them with `pip install -r requirements.txt`.")
        sys.exit(0)
    else:
        raise ImportError(
            "Missing dependencies: " + ", ".join(missing_deps)
        )

# =============================================================================
# 1. UNIFIED CONFIGURATION SYSTEM
# =============================================================================

@dataclass
class PrivacyConfig:
    """Configuration for the security and privacy gateway."""
    ollama_base_url: str = "http://localhost:11434"
    ollama_classification_model: str = "llama3.1:8b"
    ollama_sanitization_model: str = "phi3:3.8b"
    ollama_timeout: int = 60
    ollama_max_retries: int = 2
    force_local_processing_categories: List[str] = field(default_factory=lambda: ["HEALTH", "FINANCIAL", "PERSONAL"])

@dataclass
class TrainingConfig:
    """Configuration for the advanced training process."""
    # Core Model & PEFT Settings
    base_model_name: str = "bigscience/bloomz-560m"
    max_seq_length: int = 256  # Reduced for faster example
    peft_num_virtual_tokens: int = 10
    peft_prompt_tuning_init: str = "RANDOM"
    output_dir: str = "./unified_securely_tuned_model"

    # Training Hyperparameters
    learning_rate: float = 3e-4
    num_epochs: int = 1  # Reduced for speed
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    label_smoothing: float = 0.1

    # Advanced Training Features
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_ewc_regularization: bool = True
    ewc_lambda: float = 0.4
    use_sam_optimizer: bool = True
    use_early_stopping: bool = True
    patience: int = 3

@dataclass
class UnifiedConfig:
    """Main system configuration that combines privacy and training."""
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            # Override to disable GPU-specific features if on CPU
            self.training.use_mixed_precision = False

# =============================================================================
# 2. OLLAMA SECURITY GATEWAY
# =============================================================================

class OllamaSecurityGateway:
    """Local Ollama-based gateway for PII detection and data classification."""
    
    def __init__(self, config: PrivacyConfig):
        self.config = config
        self.session = requests.Session()
        self.pii_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'CREDIT_CARD': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'SSN': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        }

    def _ollama_generate(self, prompt: str, model: str) -> str:
        """Sends a prompt to the Ollama API and returns the response."""
        url = f"{self.config.ollama_base_url}/api/generate"
        for attempt in range(self.config.ollama_max_retries):
            try:
                data = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}}
                response = self.session.post(url, json=data, timeout=self.config.ollama_timeout)
                response.raise_for_status()
                result = response.json()
                response_text = result.get('response', '{}').strip()
                json_part = response_text[response_text.find('{'):response_text.rfind('}')+1]
                return json_part if json_part else response_text
            except requests.exceptions.RequestException as e:
                if attempt == self.config.ollama_max_retries - 1:
                    print(f"Ollama API error after {self.config.ollama_max_retries} retries: {e}")
                    raise
                time.sleep(2 ** attempt)
        return ""

    def classify_and_sanitize(self, text: str) -> Dict[str, Any]:
        """
        Analyzes text for sensitivity and PII, then sanitizes it.
        Returns a dictionary with the analysis and the sanitized text.
        """
        # Step 1: Classify the data sensitivity
        classification_prompt = f"""
Analyze the text below. Classify its category.
Respond ONLY with a valid JSON object with one key: "category" (one of: PERSONAL, FINANCIAL, HEALTH, TECHNICAL, BUSINESS, PUBLIC).

Text: "{text[:1000]}"

JSON Response:
"""
        try:
            classification_response = self._ollama_generate(classification_prompt, self.config.ollama_classification_model)
            classification = json.loads(classification_response)
        except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
            print(f"Warning: Data classification failed: {e}. Defaulting to BUSINESS.")
            classification = {"category": "BUSINESS"}
        
        category = classification.get("category", "BUSINESS").upper()
        
        # Step 2: Check if the category forces local processing (i.e., is too sensitive for training)
        is_too_sensitive = category in self.config.force_local_processing_categories
        
        # Step 3: Sanitize text using regex for known PII patterns
        sanitized_text = text
        pii_found = False
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, sanitized_text):
                pii_found = True
                sanitized_text = re.sub(pattern, f'[{pii_type}]', sanitized_text, flags=re.IGNORECASE)

        return {
            "original_text": text,
            "sanitized_text": sanitized_text,
            "category": category,
            "is_too_sensitive": is_too_sensitive,
            "contains_pii": pii_found,
        }

# =============================================================================
# 3. SECURE DATA PROCESSOR
# =============================================================================

class SecureDataProcessor:
    """
    Processes a raw dataset using the OllamaSecurityGateway to produce a clean,
    secure dataset suitable for model training.
    """
    def __init__(self, privacy_config: PrivacyConfig):
        self.gateway = OllamaSecurityGateway(privacy_config)

    def create_secure_dataset(self, raw_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filters and sanitizes a list of data entries.
        - Rejects entries that are classified as too sensitive.
        - Sanitizes entries containing PII before adding them to the clean dataset.
        """
        print("\n🔒 Starting secure dataset processing...")
        secure_training_data = []
        rejected_count = 0

        for item in raw_data:
            text = item.get("text", "")
            if not text:
                continue

            analysis = self.gateway.classify_and_sanitize(text)
            
            if analysis["is_too_sensitive"]:
                rejected_count += 1
                print(f"  - REJECTED data classified as '{analysis['category']}'.")
            else:
                if analysis["contains_pii"]:
                    print(f"  - SANITIZED data classified as '{analysis['category']}'.")
                secure_training_data.append({"text": analysis["sanitized_text"]})
        
        print(f"✅ Secure dataset processing complete. Kept {len(secure_training_data)} records, rejected {rejected_count}.")
        return secure_training_data

# =============================================================================
# 4. ADVANCED TRAINING SYSTEM
# =============================================================================
# Note: The GradientSurgery and SharpnessAwareMinimizer classes from the original
# file are included here for completeness. For brevity, they are collapsed.
class GradientSurgery:
    def __init__(self, model: nn.Module, ewc_lambda: float, device: str): self.model, self.ewc_lambda, self.device, self.fisher_information, self.optimal_params = model, ewc_lambda, device, {}, {}
    def compute_fisher_information(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval(); fisher_dict = {name: torch.zeros_like(p, device=self.device) for name, p in self.model.named_parameters() if p.requires_grad}
        for batch in dataloader:
            self.model.zero_grad(); batch = {k: v.to(self.device) for k, v in batch.items()}; outputs = self.model(**batch); loss = outputs.loss; loss.backward()
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None: fisher_dict[name] += p.grad.pow(2)
        num_samples = sum(len(b['input_ids']) for b in dataloader); [v.div_(num_samples) for v in fisher_dict.values()]
        self.fisher_information = fisher_dict; self.optimal_params = {name: p.clone().detach() for name, p in self.model.named_parameters() if p.requires_grad}
    def ewc_loss(self) -> torch.Tensor:
        if not self.fisher_information: return 0.0
        return self.ewc_lambda * sum((self.fisher_information[name] * (p - self.optimal_params[name]).pow(2)).sum() for name, p in self.model.named_parameters() if name in self.fisher_information)
class SharpnessAwareMinimizer:
    def __init__(self, optimizer, model, rho=0.05): self.optimizer, self.model, self.rho = optimizer, model, rho
    def step(self, loss_fn):
        loss = loss_fn(); loss.backward(); self._ascent_step(); self.optimizer.zero_grad(); loss_fn().backward(); self._descent_step(); self.optimizer.zero_grad(); return loss
    def _grad_norm(self): return torch.norm(torch.stack([p.grad.norm(p=2) for g in self.optimizer.param_groups for p in g['params'] if p.grad is not None]), p=2)
    def _ascent_step(self):
        grad_norm = self._grad_norm(); scale = self.rho / (grad_norm + 1e-12)
        for g in self.optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None: p.adv = p.grad * scale.to(p); p.data.add_(p.adv)
    def _descent_step(self):
        for g in self.optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None: p.data.sub_(p.adv)
        self.optimizer.step()

class AdvancedTrainer:
    """Enterprise-grade trainer adapted for the unified pipeline."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.training_config = config.training
        self.model = None
        self.tokenizer = None
    
    def setup_model_and_tokenizer(self):
        """Initializes the model and tokenizer for training."""
        print("\n🔧 Setting up model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.training_config.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {'torch_dtype': torch.float16 if self.training_config.use_mixed_precision else torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(self.training_config.base_model_name, **model_kwargs).to(self.config.device)
        if self.training_config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=self.training_config.peft_prompt_tuning_init,
            num_virtual_tokens=self.training_config.peft_num_virtual_tokens,
            tokenizer_name_or_path=self.training_config.base_model_name,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def create_optimized_dataset(self, data: List[Dict]):
        """Creates a tokenized Hugging Face Dataset."""
        dataset = Dataset.from_pandas(pd.DataFrame(data))
        return dataset.map(
            lambda ex: self.tokenizer(ex['text'], truncation=True, padding='max_length', max_length=self.training_config.max_seq_length),
            batched=True, remove_columns=dataset.column_names
        )

    class CustomTrainer(Trainer):
        """Custom trainer integrating SAM and EWC."""
        def __init__(self, config: TrainingConfig, gradient_surgery=None, **kwargs):
            super().__init__(**kwargs)
            self.training_config, self.gradient_surgery, self.sam_optimizer = config, gradient_surgery, None
            if self.training_config.use_sam_optimizer:
                base_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
                self.optimizer = base_optimizer
                self.sam_optimizer = SharpnessAwareMinimizer(base_optimizer, self.model)
        
        def training_step(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
            model.train(); inputs = self._prepare_inputs(inputs)
            def loss_fn():
                with self.compute_loss_context_manager(): loss = self.compute_loss(model, inputs)
                if self.gradient_surgery: loss += self.gradient_surgery.ewc_loss()
                return loss
            loss = self.sam_optimizer.step(loss_fn) if self.sam_optimizer else loss_fn()
            if not self.sam_optimizer: (loss / self.args.gradient_accumulation_steps).backward()
            return loss.detach()

    def train(self, secure_dataset: List[Dict]):
        """Executes the full training pipeline on the secure dataset."""
        self.setup_model_and_tokenizer()
        
        # Split data for training and validation
        train_data = secure_dataset[:-2] if len(secure_dataset) > 2 else secure_dataset
        val_data = secure_dataset[-2:] if len(secure_dataset) > 2 else secure_dataset
        train_dataset = self.create_optimized_dataset(train_data)
        val_dataset = self.create_optimized_dataset(val_data)
        
        # Setup Training Arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            num_train_epochs=self.training_config.num_epochs,
            fp16=self.training_config.use_mixed_precision,
            logging_steps=5, evaluation_strategy="steps", eval_steps=10, save_steps=20,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            report_to="none", remove_unused_columns=False
        )

        # Setup EWC
        gradient_surgery = None
        if self.training_config.use_ewc_regularization:
            gradient_surgery = GradientSurgery(self.model, self.training_config.ewc_lambda, self.config.device)
            print("🧠 Computing Fisher Information Matrix for EWC...")
            fisher_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, collate_fn=DataCollatorForLanguageModeling(self.tokenizer, mlm=False))
            gradient_surgery.compute_fisher_information(list(iter(fisher_loader)))

        # Initialize and run Trainer
        trainer = self.CustomTrainer(
            model=self.model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False), tokenizer=self.tokenizer,
            config=self.training_config, gradient_surgery=gradient_surgery,
            callbacks=[EarlyStoppingCallback(self.training_config.patience)]
        )
        
        print("\n🔥 Starting model training on secure dataset...")
        trainer.train()
        print("✅ Training complete.")

        print("\n💾 Saving the final trained model...")
        trainer.save_model(self.training_config.output_dir)
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        print(f"Model saved to {self.training_config.output_dir}")

    def test_inference(self, prompt: str):
        """Demonstrates inference with the final tuned model."""
        print("\n🧪 Running a test inference...")
        output_dir = self.training_config.output_dir
        if not os.path.exists(output_dir):
            print(f"Model directory not found at {output_dir}. Please train the model first.")
            return

        base_model = AutoModelForCausalLM.from_pretrained(self.training_config.base_model_name).to(self.config.device)
        model = PeftModel.from_pretrained(base_model, output_dir).to(self.config.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        inputs = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, eos_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)


# =============================================================================
# 5. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to run the entire unified pipeline."""
    # Check for Ollama connection first
    try:
        requests.get("http://localhost:11434", timeout=5).raise_for_status()
        print("✅ Ollama connection successful.")
    except requests.exceptions.RequestException:
        print("\n❌ ERROR: Ollama is not running or not reachable at http://localhost:11434.")
        print("Please install and run Ollama to use this script: https://ollama.com")
        return

    # 1. Initialize Configuration
    config = UnifiedConfig()

    # 2. Define Raw, Unfiltered Data
    # This dataset contains PII, sensitive business data, and public data.
    raw_training_data = [
        {"text": "Instruction: What is the capital of Germany? Response: The capital is Berlin."},
        {"text": "Instruction: Who wrote 'Faust'? Response: Johann Wolfgang von Goethe."},
        {"text": "Instruction: My account is locked, email me at test@example.com. Response: We will look into it."},
        {"text": "Instruction: The patient, John Doe, complains of chest pain. Response: Recommend an ECG."},
        {"text": "Instruction: What were the Q3 revenue projections? Response: Q3 revenue is projected to be $1.5M."},
        {"text": "Instruction: Write a short story about a dragon. Response: The dragon soared over the mountains."},
    ]
    
    # 3. Secure the Dataset
    # The SecureDataProcessor uses the gateway to clean the raw data.
    secure_processor = SecureDataProcessor(config.privacy)
    secure_dataset = secure_processor.create_secure_dataset(raw_training_data)

    if not secure_dataset:
        print("\n❌ No secure data was available for training after filtering. Exiting.")
        return

    # 4. Train the Model on the Secure Dataset
    # The AdvancedTrainer will now only see the sanitized, approved data.
    trainer = AdvancedTrainer(config)
    trainer.train(secure_dataset)

    # 5. Test the Final Model
    trainer.test_inference("What is the main advantage of prompt tuning?")
    
    # Clean up memory
    del trainer, secure_processor, config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    if missing_deps:
        print(
            "This script requires additional dependencies: "
            + ", ".join(missing_deps)
        )
        print("Install them with `pip install -r requirements.txt`.")
    else:
        main()
