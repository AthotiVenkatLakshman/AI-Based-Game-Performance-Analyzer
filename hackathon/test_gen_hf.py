import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HF_API_KEY")

# Models that only support text-generation (not chat)
NON_CHAT_MODELS = set()  # extend if needed

def test_generative_hf(model_id, providers=("auto",)):
    """
    Test a HF model. Tries each provider in order:
      - chat_completion first
      - if the model reports "not a chat model" → text_generation fallback
      - if the model reports a provider-routing error → retry with next provider
    """
    print(f"\n--- Testing: {model_id} ---")
    prompt = "Context: The capital of France is Paris. Question: What is the capital of France? Answer concisely:"

    for provider in providers:
        client = InferenceClient(api_key=api_key, provider=provider)
        try:
            resp = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model_id,
                max_tokens=50
            )
            print(f"✅ [chat_completion | {provider}] {resp.choices[0].message.content.strip()}")
            return
        except Exception as e:
            err_str = str(e)
            non_chat_signals = [
                "not a chat model",
                "doesn't support task 'conversational'",
                "task_not_supported",
            ]
            if any(s in err_str for s in non_chat_signals):
                # Genuinely not a chat model — fall back to text_generation
                print(f"⚠️  [{provider}] Not a chat model — falling back to text_generation...")
                try:
                    resp = client.text_generation(prompt=prompt, model=model_id, max_new_tokens=60)
                    print(f"✅ [text_generation | {provider}] {resp.strip()}")
                except Exception as e2:
                    print(f"❌ [text_generation | {provider}] {str(e2)[:200]}")
                return
            else:
                print(f"  ⚠️  [{provider}] {err_str[:150]}")

    print(f"❌ All providers failed for {model_id}")


if __name__ == "__main__":
    # ✅ Confirmed working
    test_generative_hf("meta-llama/Meta-Llama-3-8B-Instruct")

    # ✅ Confirmed working
    test_generative_hf("Qwen/Qwen2.5-7B-Instruct")

    # Mistral: novita routes it wrong — try hf-inference as fallback provider
    test_generative_hf(
        "mistralai/Mistral-7B-Instruct-v0.3",
        providers=("nebius", "sambanova", "together", "auto")
    )
