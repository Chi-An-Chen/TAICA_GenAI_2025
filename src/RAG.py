"""
Author: Chi-An Chen
Date: 2025-11-21
Description: RAG.py 包含 RAG Chatbot 的實作
"""
import os
from datetime import datetime
from typing import List, Dict, Any
# LangChain
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
# LangChain Community
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader

from src.utils import get_embedding_model, test_ollama_connection

class RAGChatbot:
    """
    RAG (Retrieval-Augmented Generation) Chatbot
    支援多種 Embedding 模型和 LLM (包含 Ollama)
    整合文件總結和 Gradio 介面
    """
    
    def __init__(
        self,
        api_keys_path: str = "./API_KEY.jsonl",
        embedding_provider: str = "gemini",
        splitter: str = "traditional",
        llm_provider: str = "ollama",
        ollama_model: str = "gemma3n:e4b",
        ollama_base_url: str = "http://localhost:11434",
        persist_directory: str = "./chroma_db"
    ):
        """
        初始化 RAG Chatbot
        
        Args:
            api_key: API 金鑰 (Gemini 或 OpenAI API Key，Ollama 不需要)
            embedding_provider: Embedding 模型提供者 ("gemini", "openai", "huggingface")
            llm_provider: LLM 提供者 ("ollama", "gemini", "openai")
            ollama_model: Ollama 模型名稱
            ollama_base_url: Ollama 服務器 URL
            persist_directory: 向量資料庫儲存路徑
        """
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.splitter = splitter
        
        if llm_provider == "ollama":
            test_ollama_connection(ollama_base_url)
        
        # 初始化 Embedding 模型
        self.embeddings = get_embedding_model(
            api_keys_path=api_keys_path,
            provider=embedding_provider,
            test_connection=True
        )
        
        if self.splitter == "traditional":
            print(f"\n[INFO] Using RecursiveCharacterTextSplitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", "。", "!", "?", ",", " ", ""]
            )
        elif self.splitter == "semantic":
            print(f"\n[INFO] Using SemanticChunker with {self.embedding_provider} embeddings")
            self.text_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.9
            )
        
        # 初始化向量資料庫
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        # 初始化 LLM
        print(f"[INFO] 初始化 {llm_provider} LLM...")
        if llm_provider == "ollama":
            self.llm = ChatOllama(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.7,
                num_predict=4096  # 最大輸出長度
            )
            print(f"[INFO] 使用 Ollama 模型: {ollama_model}")
            print(f"[INFO] Ollama 服務器: {ollama_base_url}")
        elif llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.7,
                convert_system_message_to_human=True
            )
        elif llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.7
            )
        else:
            raise ValueError(f"不支援的 LLM 提供者: {llm_provider}")
        
        # 初始化 QA Chain
        self.qa_chain = None
        
        # 儲存當前處理的文件
        self.current_documents = []
        self.summary_text = ""
    
    def load_pdf_with_pymupdf(self, pdf_path: str) -> List[Document]:
        """
        使用 PyMuPDFLoader 載入 PDF 文件
        
        Args:
            pdf_path: PDF 文件路徑
            
        Returns:
            Document 列表 (每頁一個 Document)
        """
        try:
            print(f"[INFO] 使用 PyMuPDFLoader 載入: {pdf_path}")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            print(f"[INFO] 成功載入 {len(documents)} 頁")
            return documents
        except Exception as e:
            print(f"[ERROR] PyMuPDFLoader 載入失敗: {e}")
            return []
        
    def process_documents(self, folder_path: str) -> List[Document]:
        """
        處理資料夾內所有 PDF 文件並進行文本分割
        
        Args:
            folder_path: 包含 PDF 的資料夾路徑
                
        Returns:
            分割後的文件塊列表
        """

        if not os.path.isdir(folder_path):
            raise ValueError(f"[ERROR] 資料夾不存在: {folder_path}")

        # 找出資料夾內所有 PDF
        pdf_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_paths:
            print(f"[WARNING] {folder_path} 資料夾內沒有 PDF 文件")
            return []

        all_documents = []
        
        for pdf_path in pdf_paths:
            print(f"\n[INFO] 正在處理: {pdf_path}")
            
            # 使用 PyMuPDFLoader 載入文件
            documents = self.load_pdf_with_pymupdf(pdf_path)
            
            if not documents:
                print(f"[WARNING] {pdf_path} 沒有載入任何內容")
                continue
            
            # 更新元數據, 添加文件名
            for doc in documents:
                doc.metadata["source"] = os.path.basename(pdf_path)
            
            all_documents.extend(documents)
        
        # 儲存原始文件
        self.current_documents = all_documents
        print(f"\n[INFO] 總共載入 {len(all_documents)} 頁文件")

        print(f"\n[INFO] 使用 {self.splitter} 分割處理文件...")
        
        if self.splitter == "traditional":
            splits = self.text_splitter.split_documents(all_documents)
            
        elif self.splitter == "semantic":
            splits = []
            
            # ✨ 逐個文件處理
            for i, doc in enumerate(all_documents, 1):
                try:
                    print(f"[INFO] 語義分割進度: {i}/{len(all_documents)}", end="\r")
                    
                    # 使用 split_text 處理單個文件的內容
                    text_chunks = self.text_splitter.split_text(doc.page_content)
                    
                    # 將分割結果轉為 Document 物件
                    for chunk_text in text_chunks:
                        splits.append(Document(
                            page_content=chunk_text,
                            metadata=doc.metadata.copy()
                        ))
                        
                except Exception as e:
                    print(f"\n[WARNING] 文件 {doc.metadata.get('source', 'unknown')} 語義分割失敗: {e}")
                    # 回退到包含完整文件
                    splits.append(doc)
            
            print()  # 換行
        
        else:
            raise ValueError(f"[ERROR] 未知的分割器: {self.splitter}")
        
        print(f"[INFO] 已分割為 {len(splits)} 個文本塊")
        
        return splits
    
    def summarize_documents(self, output_path: str = None) -> str:
        """
        使用 load_summarize_chain 總結所有文件
        
        Args:
            output_path: 輸出 txt 文件路徑,若為 None 則自動生成
            
        Returns:
            總結文本
        """
        if not self.current_documents:
            raise ValueError("沒有可總結的文件,請先處理 PDF 文件")
        
        print(f"\n[INFO] 正在總結 {len(self.current_documents)} 頁文件...")
        
        # 如果文件太多,先限制數量避免過長
        docs_to_summarize = self.current_documents[:50]  # 限制最多50頁
        if len(self.current_documents) > 50:
            print(f"[INFO] 文件過多,僅總結前 50 頁")
        
        # 儲存總結到 txt 文件
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"summary_{timestamp}.txt"
        
        # 自定義總結提示模板 - 簡化版本
        prompt_template = """你是一個專業的文件分析與報告撰寫助手。請根據以下文件內容生成一份 **繁體中文報告**，要求如下：

【報告要求】
1. **繁體中文**撰寫，語言正式、自然流暢。
2. **保留細節**  
   - 不可遺漏重要數據、例子、技術流程或專有名詞  
   - 避免過度簡化或隨意增添未提及資訊
3. **邏輯清晰**  
   - 段落條理分明，可使用標題或編號  
   - 方便讀者快速理解並查找資訊
4. **報告長度可完整呈現內容**  
   - 如果文件內容多，報告可以較長  
   - 保證所有關鍵點都有覆蓋

---

以下是文件全文，請依照以上要求生成完整報告:
<document>
{text}
</document>

請開始生成報告:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        
        # 使用 stuff 方式進行總結 (更適合 Ollama)
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="stuff",  # 改用 stuff，更簡單直接
            prompt=PROMPT,
            verbose=False
        )
        
        try:
            # 執行總結
            result = chain.invoke({"input_documents": docs_to_summarize})
            summary = result.get("output_text", "報告生成失敗")
            self.summary_text = summary
        except Exception as e:
            print(f"[ERROR]: {e}")
            summary = "由於文件過長或其他原因，自動總結失敗。請直接使用問答功能查詢文件內容。"
            self.summary_text = summary
        
        # 收集所有來源文件名
        sources = list(set([doc.metadata.get('source', '未知') for doc in self.current_documents]))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("文件總結\n")
            f.write("="*50 + "\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"文件數量: {len(sources)}\n")
            f.write(f"總頁數: {len(self.current_documents)}\n")
            f.write(f"文件來源: {', '.join(sources)}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "="*50 + "\n")
        
        print(f"[INFO] 總結已儲存至: {output_path}")
        
        return summary
    
    def generate_quiz(self, num_questions: int = 10, output_path: str = None) -> str:
        """
        根據文件總結生成測驗題目
        
        Args:
            num_questions: 要生成的題目數量 (預設10題)
            output_path: 輸出 txt 文件路徑,若為 None 則自動生成
            
        Returns:
            測驗文本
        """
        if not self.summary_text:
            raise ValueError("請先執行 summarize_documents() 生成總結")
        
        print(f"\n[INFO] 正在根據總結生成 {num_questions} 題測驗...")
        
        # 自定義測驗生成提示模板
        quiz_prompt = f"""請根據以下文件總結,生成 {num_questions} 道測驗題目。

要求:
1. 題型包含: 單選題(60%)、是非題(40%)
2. 難度分布: 簡單(30%)、中等(50%)、困難(20%)
3. 每題必須包含:
   - 題目描述
   - 選項 (單選題4個選項A/B/C/D, 是非題2個選項True/False)
   - 正確答案
   - 詳細解析 (必須引用原文內容,說明為何該答案正確,其他選項為何錯誤)

4. 輸出格式範例:

【題目 1】(單選題 - 難度:中等)
問題: [題目內容]
(A) [選項A]
(B) [選項B]
(C) [選項C]
(D) [選項D]

正確答案: B

詳細解析:
根據文件內容,正確答案是B,因為[引用原文說明]。
選項A錯誤是因為[說明]。
選項C錯誤是因為[說明]。
選項D錯誤是因為[說明]。

---

【題目 2】(是非題 - 難度:簡單)
問題: [題目內容]
(True) 正確
(False) 錯誤

正確答案: True

詳細解析:
此敘述正確,根據文件提到[引用原文],因此[說明]。

---

文件總結內容:
{self.summary_text}

請開始生成測驗:"""

        try:
            # 使用 LLM 生成測驗
            response = self.llm.invoke(quiz_prompt)
            
            # 處理回應內容
            if hasattr(response, 'content'):
                quiz_text = response.content
            else:
                quiz_text = str(response)
            
            print(f"[INFO] 測驗生成成功!")
            
        except Exception as e:
            print(f"[ERROR] 測驗生成失敗: {e}")
            quiz_text = "測驗生成失敗，請檢查 LLM 連線或稍後再試。"
        
        # 儲存測驗到 txt 文件
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"quiz_{timestamp}.txt"
        
        # 收集所有來源文件名
        sources = list(set([doc.metadata.get('source', '未知') for doc in self.current_documents]))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("自動生成測驗 - 基於文件內容\n")
            f.write("="*70 + "\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"題目數量: {num_questions} 題\n")
            f.write(f"文件來源: {', '.join(sources)}\n")
            f.write(f"總頁數: {len(self.current_documents)} 頁\n")
            f.write("\n" + "="*70 + "\n")
            f.write("使用說明:\n")
            f.write("- 本測驗由 AI 根據文件內容自動生成\n")
            f.write("- 每題皆附有詳細解析,幫助理解與學習\n")
            f.write("- 建議先自行作答,再查看解析\n")
            f.write("="*70 + "\n\n")
            f.write(quiz_text)
            f.write("\n\n" + "="*70 + "\n")
            f.write("測驗結束 - 記得檢討錯誤並重新閱讀相關章節!\n")
            f.write("="*70 + "\n")
        
        print(f"[INFO] 測驗已儲存至: {output_path}")
        
        return quiz_text
    
    def build_vectorstore(self, documents: List[Document]):
        """
        建立向量資料庫
        
        Args:
            documents: 文件列表
        """
        print(f"\n[INFO] 正在建立向量資料庫...")
        print(f"[INFO] 使用 {self.embedding_provider} 進行向量化...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        self.vectorstore.persist()
        print(f"[INFO] 向量資料庫已建立,共 {len(documents)} 個文本塊")
    
    def setup_qa_chain(self, k: int = 4):
        """
        設定問答鏈
        
        Args:
            k: 檢索的文件數量
        """
        if self.vectorstore is None:
            raise ValueError("請先建立或載入向量資料庫")
        
        # 自定義提示模板
        template = """你是一位親切地學習助手，請根據以下提供的背景資訊回答問題。如果你不知道答案,請誠實地說你不知道,不要嘗試編造答案。
請用繁體中文回答,並盡可能詳細、準確且有條理。

背景資訊:
{context}

問題: {question}

詳細回答:"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 建立 RetrievalQA 鏈
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print(f"[INFO] 問答系統已就緒 (使用 {self.llm_provider} LLM, 檢索 top-{k} 文件)")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        向系統提問
        
        Args:
            question: 問題
            
        Returns:
            包含答案和來源文件的字典
        """
        if self.qa_chain is None:
            raise ValueError("請先設定問答鏈")
        
        print(f"\n[INFO] 正在處理問題: {question}")
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def query_stream(self, question: str):
        """
        串流版本的查詢
        """
        if self.qa_chain is None:
            raise ValueError("請先設定問答鏈")
        
        print(f"\n[INFO] 正在處理問題: {question}")
        
        # 檢索相關文件
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        
        # 準備 context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 準備提示
        template = """請根據以下提供的背景資訊回答問題。如果你不知道答案,請誠實地說你不知道,不要嘗試編造答案。
    請用繁體中文回答,並盡可能詳細、準確且有條理。

    背景資訊:
    {context}

    問題: {question}

    詳細回答:"""
        
        prompt = template.format(context=context, question=question)
        
        # 串流生成
        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content, docs
            else:
                yield str(chunk), docs