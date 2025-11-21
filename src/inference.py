"""
Author: Chi-An Chen
Date: 2025-11-21
Description: inference.py 包含 Gradio 介面封裝
"""
import gradio as gr
from typing import List

from src.RAG import RAGChatbot

class GradioRAGInterface:
    """
    Gradio 介面封裝
    """
    
    def __init__(self, chatbot: RAGChatbot):
        self.chatbot = chatbot
        
    def chat(self, message: str, history: List):
        """
        串流版本的聊天
        """
        if not message.strip():
            yield history, ""
            return
        
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        
        try:
            accumulated_answer = ""
            sources = None
            
            # 使用串流查詢
            for chunk, docs in self.chatbot.query_stream(message):
                accumulated_answer += chunk
                history[-1]["content"] = accumulated_answer
                sources = docs
                yield history, ""
            
            # 添加來源資訊
            if sources:
                source_info = "\n\n📚 參考來源:\n"
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get('source', '未知')
                    page = doc.metadata.get('page', '未知')
                    preview = doc.page_content[:100].replace('\n', ' ')
                    source_info += f"\n[{i}] {source} (頁碼: {page})\n    內容預覽: {preview}..."
                
                history[-1]["content"] += source_info
                yield history, ""
                
        except Exception as e:
            history[-1]["content"] = f"錯誤: {str(e)}"
            yield history, ""
    
    def clear_chat(self):
        """清除聊天記錄"""
        return []
    
    def launch(self, share: bool = True, server_name: str = "0.0.0.0", server_port: int = 7860, inbrowser: bool = True):
        """啟動 Gradio 介面"""
        
        with gr.Blocks(title="RAG Chatbot - PDF 文件問答系統", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🤖 RAG Chatbot - PDF 文件問答系統
                ### 會自動生成文件總結和相關測驗題目
                """
            )
            
            # 顯示總結
            gr.Markdown("## 文件總結")
            summary_display = gr.Textbox(
                label="文件總結內容",
                value=self.chatbot.summary_text,
                lines=20,
                interactive=False,
                show_copy_button=True
            )
            
            gr.Markdown("---")
            gr.Markdown("## RAG 問答")
            
            chatbot_ui = gr.Chatbot(
                type="messages",
                label="對話記錄",
                height=500,
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="輸入您的問題",
                    placeholder="例如: 這份文件的主要內容是什麼?",
                    scale=4
                )
                submit_btn = gr.Button("發送", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("清除對話記錄")
            
            gr.Markdown(
                """
                ---
                **提示:**
                - 問答時系統會顯示參考來源和頁碼
                - LLM 使用 Ollama 本地模型進行回答，無需 API 配額
                - 使用 Gemini 進行文件向量化
                - 總結及測驗題目已自動儲存為 txt 文件
                """
            )
            
            # 事件綁定
            # ✅ 正確的串流綁定方式
            submit_btn.click(
                fn=self.chat,
                inputs=[msg_input, chatbot_ui],
                outputs=[chatbot_ui, msg_input],
                api_name="chat"
            ).then(
                fn=lambda: gr.update(value=""),
                outputs=[msg_input]
            )

            msg_input.submit(
                fn=self.chat,
                inputs=[msg_input, chatbot_ui],
                outputs=[chatbot_ui, msg_input],
                api_name="chat_submit"
            ).then(
                fn=lambda: gr.update(value=""),
                outputs=[msg_input]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot_ui]
            )
        
        demo.launch(share=share, server_name=server_name, server_port=server_port, inbrowser=inbrowser)