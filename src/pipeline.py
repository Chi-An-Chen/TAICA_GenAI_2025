import os
from typing import Optional
from src.RAG import RAGChatbot
from src.inference import GradioRAGInterface


class RAGPipeline:
    """
    RAG 系統完整流程管理類別
    提供簡潔的 API 來設定參數並執行整個流程
    """
    
    def __init__(
        self,
        # PDF 處理參數
        pdf_paths: str,
        
        # Embedding 參數
        embedding_provider: str = "gemini",
        api_keys_path: str = "./API_KEY.jsonl",
        
        # 文本分割參數
        splitter: str = "semantic",
        
        # LLM 參數
        llm_provider: str = "ollama",
        ollama_model: str = "gemma3n:e4b",
        ollama_base_url: str = "http://localhost:11434",
        
        # 向量資料庫參數
        persist_directory: str = "./chroma_db_ollama",
        
        # 輸出檔案參數
        summary_path: str = "./summary.txt",
        quiz_path: str = "./quiz.txt",
        quiz_num_questions: int = 10,
        
        # RAG 檢索參數
        retrieval_k: int = 4,
        
        # Gradio 介面參數
        gradio_share: bool = True,
        gradio_server_name: str = "0.0.0.0",
        gradio_server_port: int = 7860,
        gradio_inbrowser: bool = True
    ):
        """
        初始化 RAG Pipeline
        
        Args:
            pdf_paths: PDF 資料夾路徑
            embedding_provider: Embedding 提供者 ("gemini", "openai", "huggingface")
            api_keys_path: API 金鑰檔案路徑
            splitter: 文本分割方式 ("semantic" 或 "recursive")
            llm_provider: LLM 提供者 ("ollama", "gemini", "openai")
            ollama_model: Ollama 模型名稱
            ollama_base_url: Ollama 服務器 URL
            persist_directory: 向量資料庫儲存路徑
            summary_path: 總結輸出檔案路徑
            quiz_path: 測驗輸出檔案路徑
            quiz_num_questions: 測驗題目數量
            retrieval_k: RAG 檢索的文件數量
            gradio_share: 是否生成分享連結
            gradio_server_name: Gradio 服務器名稱
            gradio_server_port: Gradio 服務器端口
            gradio_inbrowser: 是否自動開啟瀏覽器
        """
        # PDF 處理參數
        self.pdf_paths = pdf_paths
        
        # RAG Chatbot 參數
        self.embedding_provider = embedding_provider
        self.api_keys_path = api_keys_path
        self.splitter = splitter
        self.llm_provider = llm_provider
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.persist_directory = persist_directory
        
        # 輸出檔案參數
        self.summary_path = summary_path
        self.quiz_path = quiz_path
        self.quiz_num_questions = quiz_num_questions
        
        # RAG 檢索參數
        self.retrieval_k = retrieval_k
        
        # Gradio 參數
        self.gradio_share = gradio_share
        self.gradio_server_name = gradio_server_name
        self.gradio_server_port = gradio_server_port
        self.gradio_inbrowser = gradio_inbrowser
        
        # 內部狀態
        self.chatbot: Optional[RAGChatbot] = None
        self.interface: Optional[GradioRAGInterface] = None
        self.documents = None
        self.summary = None
        self.quiz = None
    
    def initialize_chatbot(self):
        """步驟 1: 初始化 RAG Chatbot"""
        print("="*70)
        print("[STEP 1] 初始化 RAG Chatbot")
        print("="*70)
        
        self.chatbot = RAGChatbot(
            api_keys_path=self.api_keys_path,
            embedding_provider=self.embedding_provider,
            splitter=self.splitter,
            llm_provider=self.llm_provider,
            ollama_model=self.ollama_model,
            ollama_base_url=self.ollama_base_url,
            persist_directory=self.persist_directory
        )
        
        print("[INFO] ✅ RAG Chatbot 初始化完成\n")
        return self.chatbot
    
    def process_documents(self):
        """步驟 2: 處理 PDF 文件"""
        if self.chatbot is None:
            raise ValueError("請先執行 initialize_chatbot()")
        
        print("="*70)
        print("[STEP 2] 處理 PDF 文件")
        print("="*70)
        
        self.documents = self.chatbot.process_documents(self.pdf_paths)
        
        print(f"[INFO] ✅ 已處理 {len(self.documents)} 個文件塊\n")
        return self.documents
    
    def generate_summary(self, force_regenerate: bool = False):
        """步驟 3: 生成文件總結"""
        if self.chatbot is None:
            raise ValueError("請先執行 initialize_chatbot()")
        
        print("="*70)
        print("[STEP 3] 生成文件總結")
        print("="*70)
        
        # 檢查檔案是否存在
        if os.path.exists(self.summary_path) and not force_regenerate:
            choice = input(
                f"[INFO] 檔案 '{self.summary_path}' 已存在，是否要重新生成並覆蓋？ (y/n): "
            ).lower().strip()
            
            if choice != "y":
                print(f"[INFO] 跳過總結生成，將使用現有檔案\n")
                # 讀取現有檔案
                with open(self.summary_path, 'r', encoding='utf-8') as f:
                    self.summary = f.read()
                return self.summary
        
        self.summary = self.chatbot.summarize_documents(output_path=self.summary_path)
        
        print(f"[INFO] ✅ 總結已儲存至 {self.summary_path}\n")
        return self.summary
    
    def generate_quiz(self, force_regenerate: bool = False):
        """步驟 4: 生成學習測驗"""
        if self.chatbot is None:
            raise ValueError("請先執行 initialize_chatbot()")
        
        if not self.summary:
            print("[WARNING] 尚未生成總結，將先執行 generate_summary()")
            self.generate_summary(force_regenerate)
        
        print("="*70)
        print("[STEP 4] 生成學習測驗")
        print("="*70)
        
        # 檢查檔案是否存在
        if os.path.exists(self.quiz_path) and not force_regenerate:
            choice = input(
                f"[INFO] 檔案 '{self.quiz_path}' 已存在，是否要重新生成並覆蓋？ (y/n): "
            ).lower().strip()
            
            if choice != "y":
                print(f"[INFO] 跳過測驗生成，將使用現有檔案\n")
                # 讀取現有檔案
                with open(self.quiz_path, 'r', encoding='utf-8') as f:
                    self.quiz = f.read()
                return self.quiz
        
        self.quiz = self.chatbot.generate_quiz(
            num_questions=self.quiz_num_questions,
            output_path=self.quiz_path
        )
        
        print(f"[INFO] ✅ 測驗已儲存至 {self.quiz_path}\n")
        return self.quiz
    
    def build_vectorstore(self):
        """步驟 5: 建立向量資料庫"""
        if self.chatbot is None:
            raise ValueError("請先執行 initialize_chatbot()")
        
        if self.documents is None:
            raise ValueError("請先執行 process_documents()")
        
        print("="*70)
        print("[STEP 5] 建立向量資料庫")
        print("="*70)
        
        self.chatbot.build_vectorstore(self.documents)
        
        print(f"[INFO] ✅ 向量資料庫已建立\n")
    
    def setup_qa_system(self):
        """步驟 6: 設定 RAG 問答系統"""
        if self.chatbot is None:
            raise ValueError("請先執行 initialize_chatbot()")
        
        print("="*70)
        print("[STEP 6] 設定 RAG 問答系統")
        print("="*70)
        
        self.chatbot.setup_qa_chain(k=self.retrieval_k)
        
        print(f"[INFO] ✅ 問答系統已就緒（檢索 top-{self.retrieval_k} 文件）\n")
    
    def launch_interface(self):
        """步驟 7: 啟動 Gradio 介面"""
        if self.chatbot is None:
            raise ValueError("請先執行 initialize_chatbot()")
        
        print("="*70)
        print("[STEP 7] 啟動 Gradio 介面")
        print("="*70)
        
        self.interface = GradioRAGInterface(self.chatbot)
        
        # 傳遞總結和測驗內容（如果已生成）
        if self.summary:
            self.interface.summary_text = self.summary
        if self.quiz:
            self.interface.quiz_text = self.quiz
        
        self.interface.launch(
            share=self.gradio_share,
            server_name=self.gradio_server_name,
            server_port=self.gradio_server_port,
            inbrowser=self.gradio_inbrowser
        )
    
    def run(self, skip_summary: bool = False, skip_quiz: bool = False):
        """
        執行完整流程（一鍵運行）
        
        Args:
            skip_summary: 是否跳過總結生成
            skip_quiz: 是否跳過測驗生成
        """
        print("\n" + "="*35)
        print("RAG 系統啟動 - 完整流程執行")
        print("="*35 + "\n")
        
        try:
            # 步驟 1: 初始化
            self.initialize_chatbot()
            
            # 步驟 2: 處理文件
            self.process_documents()
            
            # 步驟 3: 生成總結（可選）
            if not skip_summary:
                self.generate_summary()
            
            # 步驟 4: 生成測驗（可選）
            if not skip_quiz:
                self.generate_quiz()
            
            # 步驟 5: 建立向量資料庫
            self.build_vectorstore()
            
            # 步驟 6: 設定問答系統
            self.setup_qa_system()
            
            # 步驟 7: 啟動介面
            self.launch_interface()
            
        except KeyboardInterrupt:
            print("\n[INFO] 使用者中斷程式執行")
        except Exception as e:
            print(f"\n[ERROR] ❌ 執行過程中發生錯誤: {e}")
            raise