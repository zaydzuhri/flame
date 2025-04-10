# test_load.py
import transformers
import fla # Make sure this import happens FIRST to trigger registration

model_path = "evaluation_lm_eval/models/vanilla-340M-4096-batch16-steps100000-20250409-210858"
print(f"Attempting to load model from: {model_path}")

try:
    # Trust remote code is often needed for custom models
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    print(model.config)
except Exception as e:
    print(f"Error loading model: {e}")