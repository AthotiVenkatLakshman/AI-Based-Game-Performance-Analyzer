import os
import json
import time
from dotenv import load_dotenv
import pypdf
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()

class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

class TrainingBot:
    def __init__(self):
        print("DEBUG: TrainingBot.__init__ started")
        # API Keys
        self.hf_api_key = os.getenv("HF_API_KEY")
        print(f"DEBUG: HF_API_KEY present: {bool(self.hf_api_key)}")
        
        # Using Cloud-based Hugging Face embeddings to avoid local load hangs
        if self.hf_api_key:
            print("DEBUG: Initializing HuggingFaceEndpointEmbeddings...")
            self.embeddings = HuggingFaceEndpointEmbeddings(
                huggingfacehub_api_token=self.hf_api_key,
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("DEBUG: Embeddings initialized")
            print("DEBUG: Initializing InferenceClient...")
            self.client = InferenceClient(api_key=self.hf_api_key)
            print("DEBUG: InferenceClient initialized")
        else:
            self.embeddings = None
            self.client = None

        # LLM model priority: primary → fallback
        self.primary_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.fallback_model = "Qwen/Qwen2.5-7B-Instruct"
            
        self.vector_db = None
        self.cache_file = "llm_cache.json"
        print("DEBUG: Loading cache...")
        self.cache = self._load_cache()
        print("DEBUG: TrainingBot initialized successfully")

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    def _call_llm(self, messages, max_tokens=250, temperature=0.1):
        """Call chat_completion with primary model, fall back to secondary on failure."""
        for model in (self.primary_model, self.fallback_model):
            try:
                resp = self.client.chat_completion(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                print(f"DEBUG: LLM response from {model}")
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"DEBUG: {model} failed — {str(e)[:120]}")
        raise RuntimeError("All LLM models failed. Please check your HF_API_KEY and model availability.")

    def ingest_pdf(self, file_path):
        """Loads PDF, splits text, and creates FAISS index."""
        print(f"DEBUG: Loading PDF: {file_path}")
        full_text = ""
        with open(file_path, "rb") as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        print("DEBUG: Splitting text...")
        text_splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(full_text)
        
        print(f"DEBUG: Creating FAISS index from {len(chunks)} chunks...")
        self.vector_db = FAISS.from_texts(chunks, self.embeddings)
        print("DEBUG: FAISS index created")
        return "Knowledge base updated successfully."

    def get_answer(self, query, language):
        """Retrieves context and generates a response using a generative Hugging Face model."""
        if not self.vector_db:
            return "Please upload a PDF first."

        try:
            # Check cache first
            cache_key = f"{query}_{language}"
            if cache_key in self.cache:
                print(f"DEBUG: Cache hit for {query}")
                return self.cache[cache_key]

            # Retrieve context from vector store
            docs = self.vector_db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            if not self.client:
                return "❌ Hugging Face API Key is missing."

            # Generative RAG Prompt — document-grounded + explanation
            system_prompt = f"""You are an expert corporate training assistant.
Your job is to answer the employee's question using the provided document context.

Follow this exact two-part format in your response:

**📄 From the Document:**
Give a direct, accurate answer based strictly on the provided context.
If the topic is not covered, say: "This specific topic is not mentioned in the current document."

**💡 Explanation:**
Expand on the answer with a clear, simple explanation to help the employee fully understand the concept, rule, or policy. You may use examples or analogies where helpful.

Your entire response MUST be in {language}."""

            user_prompt = f"Document Context:\n{context}\n\nEmployee Question: {query}\n\nRespond in {language} using the two-part format:"

            # Use InferenceClient for Generative Chat Completion (primary: Llama, fallback: Qwen)
            result = self._call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )

            # Save to cache
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
        except Exception as e:
            if "503" in str(e):
                return "⚠️ Hugging Face model is currently loading. Please try again in 30-60 seconds."
            return f"❌ Error generating answer: {str(e)}"

    def generate_summary(self, language):
        """Generates a professional summary using a generative Hugging Face model."""
        if not self.vector_db:
            return "No document uploaded."

        try:
            # Check cache for summary
            cache_key = f"summary_{language}"
            if cache_key in self.cache:
                print(f"DEBUG: Cache hit for summary in {language}")
                return self.cache[cache_key]

            # Get representative content for summary
            docs = self.vector_db.similarity_search("main policy highlights and overview", k=5)
            context = "\n".join([d.page_content for d in docs])
            
            if not self.client:
                return "❌ Hugging Face API Key is missing."

            # Generative Summary Prompt
            system_prompt = f"""You are a professional assistant. 
Summarize the following document context for an employee. 
Highlight the most important rules or guidelines.
Your summary MUST be entirely in {language}."""

            user_prompt = f"Document Context:\n{context}\n\nProvide a concise and professional summary in {language}:"

            # Use InferenceClient for Summary (primary: Llama, fallback: Qwen)
            result = self._call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )

            # Save to cache
            self.cache[cache_key] = result
            self._save_cache()

            return result
        except Exception as e:
            if "503" in str(e):
                return "⚠️ Hugging Face model is currently loading. Please try again in 30-60 seconds."
            return f"❌ Error generating summary: {str(e)}"
