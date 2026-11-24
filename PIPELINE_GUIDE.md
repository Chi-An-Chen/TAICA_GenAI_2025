# Following the PDF chatbot pipeline

To inspect the course project without starting its services, read the [saved summary](gen_results/summary.txt) and [quiz](gen_results/quiz.txt). To trace their source, start with [app.py](app.py) in the repository root and follow [RAGPipeline.run](src/pipeline.py).

1. The application reads PDFs from [pdf_files/](pdf_files/). The shipped configuration selects Gemini for embeddings and a local Ollama server with `gemma3n:e4b` for the language model. Both providers must be available for that configuration.
2. The pipeline generates a summary and quiz before setting up question answering. If the output files already exist, it prompts whether to regenerate or reuse them. When supplying a different set of PDFs, check that any reused material belongs to that input.
3. After building the vector store and setting up retrieval, the application opens the Gradio interface. The configured port is `7860`. New summary and quiz files use the same `gen_results/` paths linked above.

The saved text is an example of the output format. The files alone do not show whether the configured providers are available today.
