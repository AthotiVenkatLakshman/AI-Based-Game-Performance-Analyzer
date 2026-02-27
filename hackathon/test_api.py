import os
from dotenv import load_dotenv
from rag_pipeline import TrainingBot

def test_backend():
    load_dotenv()
    hf_api_key = os.getenv("HF_API_KEY")

    if not hf_api_key:
        print("❌ Error: HF_API_KEY not found in .env")
        return

    print(f"Testing Backend with HF Key: {hf_api_key[:10]}...")

    try:
        # Initialize Bot
        bot = TrainingBot()
        
        # Test 1: Embedding Model (Cloud-based)
        print("Testing Cloud Embedding Connection...")
        test_text = "This is a connection test."
        if bot.embeddings:
            vector = bot.embeddings.embed_query(test_text)
            print(f"✅ Embedding API Working: Vector size {len(vector)}")
        else:
            print("❌ Embeddings not initialized")

        # Test 2: Hugging Face API Connection (via TrainingBot)
        print("Testing Hugging Face Inference Client...")
        if bot.client:
            try:
                # Test Q&A logic
                print("Testing Q&A (Hugging Face / deepset/roberta-base-squad2)...")
                qa_result = bot.client.question_answering(
                    question="What is AI?", 
                    context="Artificial Intelligence is the simulation of human intelligence.",
                    model="deepset/roberta-base-squad2"
                )
                print(f"✅ Hugging Face QA Working: {qa_result.answer}")
            except Exception as e:
                print(f"❌ Hugging Face QA Error: {e}")
        else:
            print("❌ HF Client not initialized")
        
        print("\n🚀 BACKEND VERIFICATION COMPLETE.")

    except Exception as e:
        print(f"\n❌ Error Detected: {e}")

if __name__ == "__main__":
    test_backend()
