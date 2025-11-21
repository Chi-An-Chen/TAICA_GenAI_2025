"""
Author: Chi-An Chen
Date: 2025-11-21
Description: utils.py 包含載入 API 金鑰和測試 Embedding 模型的功能
"""
import os
import json
import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def test_embedding_model(embeddings, provider: str) -> bool:
    """
    測試 Embedding 模型是否正常運作
    
    Args:
        embeddings: Embedding 模型實例
        provider: 提供者名稱
        
    Returns:
        測試是否成功
    """
    test_text = "這是一個測試句子，用於驗證 Embedding 模型是否正常運作。"
    
    try:
        print(f"\n[TEST] 測試 {provider} Embedding 模型...")
        print(f"[TEST] 測試文本: '{test_text}'")
        
        # 生成 embedding
        result = embeddings.embed_query(test_text)
        
        # 檢查結果
        if isinstance(result, list) and len(result) > 0:
            print(f"[TEST] ✅ {provider} Embedding 模型測試成功!")
            print(f"[TEST] 向量維度: {len(result)}")
            print(f"[TEST] 向量前5個值: {result[:5]}")
            return True
        else:
            print(f"[TEST] ❌ {provider} Embedding 模型測試失敗: 返回格式不正確")
            return False
            
    except Exception as e:
        print(f"[TEST] ❌ {provider} Embedding 模型測試失敗: {e}")
        return False


def test_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """
    測試 Ollama 服務是否正常運作
    
    Args:
        base_url: Ollama 服務器 URL
        
    Returns:
        測試是否成功
    """
    try:
        import requests
        
        print(f"\n[TEST] 測試 Ollama 連線...")
        print(f"[TEST] 服務器 URL: {base_url}")
        
        # 測試 Ollama API
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"[TEST] Ollama 連線成功!")
            print(f"[TEST] 可用模型數量: {len(models)}")
            if models:
                print(f"[TEST] 可用模型:")
                for model in models[:5]:  # 只顯示前5個
                    print(f"  - {model.get('name', 'unknown')}")
            return True
        else:
            print(f"[TEST] Ollama 連線失敗: HTTP {response.status_code}")
            return False
            
    except ImportError:
        print(f"[TEST] 請安裝 requests 套件: pip install requests")
        return False
    except requests.exceptions.ConnectionError:
        print(f"[TEST] 無法連線到 Ollama 服務器")
        print(f"[TEST] 請確認 Ollama 是否已啟動: ollama serve")
        return False
    except Exception as e:
        print(f"[TEST] Ollama 連線測試失敗: {e}")
        return False
    
def get_embedding_model(provider: str, api_keys_path: str = "./API_KEY.json", test_connection: bool = True):
    """
    根據提供者選擇 Embedding 模型（整合 API 金鑰載入和測試）
    
    Args:
        provider: "huggingface", "openai", 或 "gemini"
        api_keys_path: API 金鑰檔案路徑
        test_connection: 是否測試連線
        
    Returns:
        對應的 Embedding 模型
    """
    provider = provider.lower()
    
    # ========== STEP 1: 載入 API 金鑰 ==========
    if provider in ["openai", "gemini"]:
        print(f"\n[INFO] Loading API Key...")
        if not os.path.isabs(api_keys_path):
            # 如果是相對路徑，從當前工作目錄（執行 app.py 的位置）解析
            api_keys_path = os.path.abspath(api_keys_path)
            print(f"[INFO] API 金鑰檔案路徑: {api_keys_path}")
        try:
            with open(api_keys_path, 'r', encoding='utf-8') as f:
                api_keys = json.load(f)
            
            print(f"[INFO] ✅ 成功載入 API 金鑰檔案")
            
            # 設定環境變數
            if "OPENAI_API_KEY" in api_keys:
                os.environ["OPENAI_API_KEY"] = api_keys["OPENAI_API_KEY"]
            if "GOOGLE_API_KEY" in api_keys:
                os.environ["GOOGLE_API_KEY"] = api_keys["GOOGLE_API_KEY"]
                
        except FileNotFoundError:
            print(f"[ERROR] 找不到 API 金鑰檔案: {api_keys_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"[ERROR] API 金鑰檔案格式錯誤: {e}")
            raise
        except Exception as e:
            print(f"[ERROR] 載入 API 金鑰時發生錯誤: {e}")
            raise
    
    # ========== STEP 2: 初始化 Embedding 模型 ==========
    print(f"\n[INFO] 初始化 {provider} Embedding 模型...")
    embeddings = None
    
    if provider == "huggingface":
        model_name = "BAAI/bge-m3"
        device = ("cuda" if torch.cuda.is_available() 
                 else "mps" if torch.backends.mps.is_available() 
                 else "cpu")
        model_kwargs = {"device": device}
        print(f"[INFO] 使用 Huggingface 於 {device}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    elif provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("[ERROR] 未設定 OPENAI_API_KEY")
        model_name = "text-embedding-3-large"
        embeddings = OpenAIEmbeddings(model=model_name)
    
    elif provider == "gemini":
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("[ERROR] 未設定 GOOGLE_API_KEY")
        model_name = "text-embedding-004"
        embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type="retrieval_document"
        )
    
    else:
        raise ValueError(f"[ERROR] 未知的 embedding 提供者: {provider}")
    
    # ========== STEP 3: 測試 Embedding 模型 ==========
    if test_connection and embeddings:
        test_text = "這是一個測試句子，用於驗證 Embedding 模型是否正常運作。"
        
        try:
            print(f"\n[STEP 3] 測試 {provider} Embedding 模型...")
            print(f"[TEST] 測試文本: '{test_text}'")
            
            # 生成 embedding
            result = embeddings.embed_query(test_text)
            
            # 檢查結果
            if isinstance(result, list) and len(result) > 0:
                print(f"[TEST] {provider} Embedding 模型測試成功!")
                print(f"[TEST] 向量維度: {len(result)}")
                print(f"[TEST] 向量前5個值: {result[:5]}")
            else:
                print(f"[TEST] {provider} Embedding 模型測試失敗: 返回格式不正確")
                raise ValueError("Embedding 模型測試失敗")
                
        except Exception as e:
            print(f"[TEST] {provider} Embedding 模型測試失敗: {e}")
            raise
    
    print(f"[INFO] {provider} Embedding 模型初始化完成\n")
    return embeddings