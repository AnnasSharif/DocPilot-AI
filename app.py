import gradio as gr
import os
import requests
import re
from collections import Counter
from PyPDF2 import PdfReader
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are DocPilotAI, an intelligent assistant that answers questions strictly based on the uploaded PDF documents.
Rules:
- Use ONLY the provided document context to answer.
- If the answer is not found in the document, clearly say that it is not available in the provided PDFs.
- Explain answers in a clear, student-friendly way.
Explanation Style:
- If the topic is theoretical, conceptual, language-related, or a computer science concept, explain it with simple and clear examples.
- If the topic is programming or code-related, explain it using relevant code examples taken from or consistent with the document context.
- Break complex ideas into steps or bullet points when helpful.
- Keep explanations clear, structured, and easy to understand for a student.
Do not add outside knowledge. Do not guess. Stay within the document content.
"""

DOCUMENT_CHUNKS = []


def extract_text_from_pdfs(files):
    text = ""
    for f in files:
        reader = PdfReader(f.name)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=400):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


def clean_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(text.split())


def retrieve_relevant_chunks(query, top_k=3):
    query_words = clean_words(query)
    if not query_words:
        return []

    scored_chunks = []

    for chunk in DOCUMENT_CHUNKS:
        chunk_words = clean_words(chunk)
        overlap_score = len(query_words & chunk_words)
        if overlap_score > 0:
            scored_chunks.append((overlap_score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:top_k]]


def ask_groq(question):
    if not GROQ_API_KEY:
        return "❌ GROQ API key not found."

    relevant_chunks = retrieve_relevant_chunks(question)

    if not relevant_chunks:
        return "No relevant information found in the uploaded PDFs."

    context = "\n\n".join(relevant_chunks)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Document Context:\n{context}"},
            {"role": "user", "content": question}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROQ_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    return "❌ Error communicating with Groq API."


def process_pdfs(files):
    global DOCUMENT_CHUNKS

    if not files:
        return "❌ No files uploaded."

    text = extract_text_from_pdfs(files)
    DOCUMENT_CHUNKS = chunk_text(text)

    return (
        f"✅ {len(files)} PDF(s) processed\n"
        f"📄 {len(DOCUMENT_CHUNKS)} text chunks created"
    )


def chat_fn(message, history):
    history = history or []

    history.append({"role": "user", "content": message})

    if not DOCUMENT_CHUNKS:
        reply = "⚠️ Please upload and process PDFs first."
    else:
        reply = ask_groq(message)

    history.append({"role": "assistant", "content": reply})
    return history, ""


def clear_all():
    global DOCUMENT_CHUNKS
    DOCUMENT_CHUNKS = []
    return [], "🧹 Cleared. Upload PDFs to start again."


with gr.Blocks() as demo:

    gr.Markdown("""
    # 📘 DocPilotAI – RAG PDF Chatbot

    **Ask questions directly from your own PDF documents.**

    ### ✨ Features
    • Multiple PDF upload  
    • Keyword-based document retrieval  
    • Groq-powered answers  
    • Clean and simple interface  

    ---
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420)

            user_msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask something from the PDF..."
            )

            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            pdf_upload = gr.File(
                label="Upload PDF Files",
                file_types=[".pdf"],
                file_count="multiple"
            )

            process_btn = gr.Button("Process PDFs", variant="primary")

            status = gr.Textbox(
                label="Status",
                value="Upload PDFs to get started",
                interactive=False,
                lines=2
            )

    process_btn.click(process_pdfs, inputs=pdf_upload, outputs=status)
    send_btn.click(chat_fn, inputs=[user_msg, chatbot], outputs=[chatbot, user_msg])
    user_msg.submit(chat_fn, inputs=[user_msg, chatbot], outputs=[chatbot, user_msg])
    clear_btn.click(clear_all, outputs=[chatbot, status])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"))
