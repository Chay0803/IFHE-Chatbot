import os
import sys
import transformers
import torch # Added: Required for tensor operations in CustomStoppingCriteria

# --------- Environment Setup for HF Cache ---------
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/transformers"

# Ensure directories exist
for env_var in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"]:
    path = os.environ[env_var]
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"✅ Created cache directory: {path}", flush=True)
        except OSError as e:
            print(f"❌ Error creating cache directory {path}: {e}", file=sys.stderr, flush=True)

print("🚀 Starting chatbot script...", flush=True)

# --------- Imports ---------
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# --------- Custom Stopping Criteria Class (NEW) ---------
class CustomStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self, stop_token_ids: list[list[int]]):
        self.stop_token_ids = []
        for ids in stop_token_ids:
            # Convert token IDs to tensors and move them to the correct device (CPU or CUDA)
            self.stop_token_ids.append(
                torch.tensor(ids, dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the generated sequence ends with any of the stop_token_ids
        for stop_ids in self.stop_token_ids:
            # Ensure the current generated sequence is long enough to contain the stop sequence
            if input_ids.shape[-1] >= stop_ids.shape[-1]:
                # Check if the last generated tokens match the stop sequence
                if torch.all(input_ids[0][-stop_ids.shape[-1]:] == stop_ids):
                    return True
        return False

# --------- Custom course matcher import ---------
try:
    from course_matcher import match_courses
    print("✅ Successfully imported course_matcher.", flush=True)
except ImportError:
    print("❌ ERROR: 'course_matcher.py' not found or invalid.", file=sys.stderr, flush=True)
    sys.exit(1)

# --------- Load PDF documents ---------
def load_documents_from_folder(folder_path):
    print(f"📂 Loading documents from: {folder_path}", flush=True)
    all_docs = []
    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder '{folder_path}' not found.", file=sys.stderr, flush=True)
        return []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            print(f"📄 Found PDF: {file_name}", flush=True)
            try:
                loader = PyPDFLoader(os.path.join(folder_path, file_name))
                docs = loader.load_and_split()
                all_docs.extend(docs)
                print(f"   ↳ Loaded {len(docs)} pages.", flush=True)
            except Exception as e:
                print(f"❌ Failed to load {file_name}: {e}", file=sys.stderr, flush=True)

    print(f"📘 Total documents loaded: {len(all_docs)}", flush=True)
    return all_docs

docs = load_documents_from_folder("data/")
if not docs:
    print("⚠️ WARNING: No documents found. QA may not work.", flush=True)

# --------- Build FAISS Vector Store ---------
print("🔧 Creating vector database...", flush=True)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    db = FAISS.from_documents(docs, embedding_model)
    retriever = db.as_retriever(search_kwargs={'k': 2}) # Limiting retriever to 2 documents
    print("✅ FAISS retriever is ready.", flush=True)
except Exception as e:
    print(f"❌ ERROR: Failed to create FAISS DB: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# --------- Load TinyLlama LLM ---------
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print("⬇️ Loading TinyLlama model...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("✅ Model and tokenizer loaded.", flush=True)
except Exception as e:
    print(f"❌ ERROR: Could not load model: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# Define custom stopping criteria list
stopping_criteria_list = transformers.StoppingCriteriaList([
    CustomStoppingCriteria([
        tokenizer.encode("\nQuestion:", add_special_tokens=False), # Stop if model generates a newline then "Question:"
        tokenizer.encode("Question:", add_special_tokens=False)    # Stop if model generates "Question:" directly
    ])
])

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # You can adjust this value
    do_sample=True,
    return_full_text=False, # Essential for correct output with LangChain
    eos_token_id=tokenizer.eos_token_id, # Ensure end-of-sentence token is respected
    pad_token_id=tokenizer.pad_token_id, # Ensure pad token is respected
    stopping_criteria=stopping_criteria_list # Use the custom stopping criteria
)
llm = HuggingFacePipeline(pipeline=pipe)
print("✅ Text generation pipeline is ready.", flush=True)

# --------- Custom QA Prompt Setup ---------
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question concisely and accurately.
If the answer is not in the context, just say you don't know.

Context:
{context}

Question: {question}
Answer:"""
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# --------- Ask Bot Function (Improved) ---------
def ask_bot(question):
    print("\n🔍 Searching documents and generating response...", flush=True)
    try:
        full_response = qa.invoke({"query": question})

        if "source_documents" in full_response:
            print("\n--- Retrieved Source Documents ---", flush=True)
            for i, doc in enumerate(full_response["source_documents"]):
                source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                page_number = doc.metadata.get('page', 'N/A')
                print(f"Document {i+1} (Page {page_number}, Source: {source_file}):", flush=True)
                print(doc.page_content[:500] + "...", flush=True)
                print("-" * 30, flush=True)
            print("--- End of Source Documents ---", flush=True)
        else:
            print("No source documents found in the response.", flush=True)

        response = full_response["result"].strip()
        
        if response.lower().startswith("answer:"):
            response = response[7:].strip()
        
        print("✅ Response generated.", flush=True)
        return response
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr, flush=True)
        return "Sorry, I encountered an error while answering that."

# --------- Candidate Profile Function ---------
def candidate_profile():
    print("\n🎯 Let's find suitable courses for you.", flush=True)
    stream = input("What stream did you take in 12th (Science/Commerce/Arts)? ")
    interest = input("What is your area of interest (Tech/Management/Law/etc.)? ")
    english = input("Are you fluent in English? (yes/no): ")

    profile = {
        "stream": stream.strip(),
        "interest": interest.strip(),
        "english": english.strip()
    }

    try:
        suggestions = match_courses(profile)
        print("\n🎓 Recommended Courses Based on Your Profile:", flush=True)
        for course in suggestions:
            print("✅", course, flush=True)
    except Exception as e:
        print(f"❌ ERROR: {e}", file=sys.stderr, flush=True)
        print("Could not provide course suggestions at this time.", flush=True)

# --------- Main Loop ---------
if __name__ == "__main__":
    print("🎓 Welcome to IFHE College Admission Chatbot 🎓", flush=True)
    while True:
        print("\nChoose an option:", flush=True)
        print("1. Ask a question about programs or admissions", flush=True)
        print("2. Get course suggestions based on your background", flush=True)
        print("3. Exit", flush=True)

        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == "1":
            query = input("Type your question: ")
            response = ask_bot(query)
            print("\n🤖 Bot Response:\n", response, flush=True)
        elif choice == "2":
            candidate_profile()
        elif choice == "3":
            print("👋 Thank you for using the chatbot. Goodbye!", flush=True)
            break
        else:
            print("❌ Invalid input. Please enter 1, 2, or 3.", flush=True)