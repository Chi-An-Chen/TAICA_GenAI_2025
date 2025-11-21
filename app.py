import os
from src.pipeline import RAGPipeline


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    pipeline = RAGPipeline(
        # PDF 參數
        pdf_paths=os.path.join(BASE_DIR, "pdf_files"),
        
        # Embedding 參數
        embedding_provider="gemini",
        api_keys_path=os.path.join(BASE_DIR, "API_KEY.json"),
        
        # 分割器
        splitter="semantic",
        
        # LLM 參數
        llm_provider="ollama",
        ollama_model="gemma3n:e4b",
        ollama_base_url="http://localhost:11434",
        
        # 輸出檔案
        summary_path=os.path.join(BASE_DIR, "gen_results/summary.txt"),
        quiz_path=os.path.join(BASE_DIR, "gen_results/quiz.txt"),
        quiz_num_questions=15,
        
        # RAG 參數
        retrieval_k=5,
        
        # Gradio 參數
        gradio_share=False,
        gradio_inbrowser=True,
        gradio_server_port=7860
    )
    pipeline.run()