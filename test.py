import os
import sys
import logging
from typing import List, Tuple
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain.document_loaders import Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    from langchain_community.llms import GPT4All
    from langchain.prompts import PromptTemplate
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    sys.exit(1)

# FAISS import với xử lý lỗi
try:
    from langchain.vectorstores import FAISS
except ImportError:
    logger.error(
        "Could not import faiss.\n"
        "Install with `pip install faiss-cpu` (for CPU) or `pip install faiss-gpu` (if you have GPU)."
    )
    sys.exit(1)

# Import thêm cho đọc file DOC
try:
    import docx2txt
    import mammoth
except ImportError as e:
    logger.error(f"Missing package for DOC reading: {e}")
    logger.error("Install with: pip install docx2txt mammoth")
    sys.exit(1)

# Configuration
DOC_FILES = [
    r"C:\project\data\tieuchuan2017.docx",
    
    
]
INDEX_DIR = "faiss_index_doc"
MODEL_PATH = r"C:/project/model/vinallama-7b-chat_q5_0.gguf"
PROMPT_CONFIG_FILE = "C:\project\prompt.json"

# Chunking parameters
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DocLoader:
    """Custom Document Loader cho file DOC và DOCX"""
    
    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """Load file DOCX sử dụng docx2txt"""
        try:
            text = docx2txt.process(file_path)
            if text.strip():
                return [Document(page_content=text, metadata={"source": file_path})]
            else:
                logger.warning(f"No text found in {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            return []
    
    @staticmethod
    def load_doc_with_mammoth(file_path: str) -> List[Document]:
        """Load file DOC/DOCX sử dụng mammoth"""
        try:
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                if text.strip():
                    return [Document(page_content=text, metadata={"source": file_path})]
                else:
                    logger.warning(f"No text found in {file_path}")
                    return []
        except Exception as e:
            logger.error(f"Error loading DOC with mammoth {file_path}: {e}")
            return []
    
    @staticmethod
    def load_doc_file(file_path: str) -> List[Document]:
        """Load file DOC/DOCX với fallback methods"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.docx':
            # Thử docx2txt trước
            docs = DocLoader.load_docx(file_path)
            if docs:
                return docs
            # Fallback sang mammoth
            return DocLoader.load_doc_with_mammoth(file_path)
        
        elif file_ext == '.doc':
            # Sử dụng mammoth cho file .doc
            return DocLoader.load_doc_with_mammoth(file_path)
        
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            return []


def load_docs(files: List[str]) -> List[Document]:
    """Load tất cả file DOC/DOCX"""
    docs: List[Document] = []
    for file_path in files:
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            continue
        
        logger.info(f"Loading DOC file: {file_path}")
        file_docs = DocLoader.load_doc_file(file_path)
        docs.extend(file_docs)
        logger.info(f"Loaded {len(file_docs)} documents from {file_path}")
    
    return docs


def split_into_chunks(docs: List[Document]) -> List[Document]:
    """Chia documents thành các chunks nhỏ"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def build_vectorstore(files: List[str], index_dir: str) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    """Xây dựng vectorstore từ các file DOC"""
    os.makedirs(index_dir, exist_ok=True)
    docs = load_docs(files)
    
    if not docs:
        raise RuntimeError("No DOC content to process. Check file paths and content.")
    
    chunks = split_into_chunks(docs)
    logger.info(f"Created {len(chunks)} chunks from DOC content")
    
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embed_model)
    vectorstore.save_local(index_dir)
    logger.info(f"Vectorstore saved to {index_dir}")
    
    return vectorstore, embed_model


def load_or_build_vectorstore(files: List[str], index_dir: str) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    """Load vectorstore hiện có hoặc tạo mới"""
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    
    try:
        vs = FAISS.load_local(index_dir, embed_model, allow_dangerous_deserialization=True)
        logger.info("Loaded existing vectorstore.")
        return vs, embed_model
    except Exception as e:
        logger.info(f"Could not load existing vectorstore: {e}")
        logger.info("Building new vectorstore...")
        return build_vectorstore(files, index_dir)


def load_prompt_config(config_file: str) -> dict:
    """Load prompt configuration từ file JSON"""
    default_config = {
        "system_prompt": """Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời câu hỏi dựa trên thông tin từ tài liệu được cung cấp.

Quy tắc trả lời:
1. Chỉ trả lời dựa trên thông tin có trong tài liệu
2. Nếu không tìm thấy thông tin, hãy nói rõ ràng
3. Trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu
4. Đưa ra câu trả lời chi tiết và có cấu trúc

Thông tin từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời:""",
        
        "question_prompt": """Dựa trên thông tin sau đây:

{context}

Hãy trả lời câu hỏi: {question}

Trả lời chi tiết và chính xác:""",
        
        "no_answer_response": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi này trong tài liệu.",
        
        "greeting_message": "Xin chào! Tôi là trợ lý AI có thể giúp bạn tìm hiểu thông tin từ tài liệu. Hãy đặt câu hỏi cho tôi!",
        
        "model_settings": {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9
        }
    }
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge với default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            # Tạo file config mặc định
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            logger.info(f"Created default prompt config at {config_file}")
            return default_config
    except Exception as e:
        logger.error(f"Error loading prompt config: {e}")
        return default_config


def init_llm(model_path: str, config: dict) -> GPT4All:
    """Khởi tạo LLM với config"""
    if not os.path.isfile(model_path):
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    model_settings = config.get("model_settings", {})
    
    return GPT4All(
        model=model_path,
        temp=model_settings.get("temperature", 0.7),
        max_tokens=model_settings.get("max_tokens", 512),
        top_p=model_settings.get("top_p", 0.9)
    )


def create_qa_chain(vectorstore: FAISS, llm: GPT4All, config: dict) -> RetrievalQA:
    """Tạo QA chain với custom prompt"""
    
    # Tạo custom prompt template
    prompt_template = config.get("system_prompt", config["question_prompt"])
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Tạo retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Lấy 3 documents liên quan nhất
        ),
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def main():
    """Main function"""
    print("🚀 Starting DOC-based RAG Chatbot...")
    
    # Load prompt configuration
    config = load_prompt_config(PROMPT_CONFIG_FILE)
    
    # Load hoặc build vectorstore
    try:
        vectorstore, _ = load_or_build_vectorstore(DOC_FILES, INDEX_DIR)
    except Exception as e:
        logger.error(f"Failed to create vectorstore: {e}")
        sys.exit(1)
    
    # Khởi tạo LLM
    llm = init_llm(MODEL_PATH, config)
    
    # Tạo QA chain
    qa_chain = create_qa_chain(vectorstore, llm, config)
    
    # Greeting
    print(f"\n{config['greeting_message']}")
    print("Gõ 'exit' để thoát, 'reload' để tải lại prompt config.\n")
    
    while True:
        try:
            query = input("Bạn: ").strip()
            
            if query.lower() == 'exit':
                print("Tạm biệt!")
                break
            
            if query.lower() == 'reload':
                config = load_prompt_config(PROMPT_CONFIG_FILE)
                qa_chain = create_qa_chain(vectorstore, llm, config)
                print("✅ Đã tải lại prompt config!")
                continue
            
            if not query:
                continue
            
            # Thực hiện query
            print("🤔 Đang suy nghĩ...")
            
            try:
                response = qa_chain.invoke({"query": query})
                answer = response.get("result", config["no_answer_response"])
                print(f"Bot: {answer}\n")
            except Exception as e:
                logger.error(f"Error during query: {e}")
                print(f"Bot: Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.\n")
                
        except KeyboardInterrupt:
            print("\n\nTạm biệt!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("Đã có lỗi xảy ra. Vui lòng thử lại.")


if __name__ == '__main__':
    main()