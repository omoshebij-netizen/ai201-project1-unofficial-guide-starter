import gradio as gr
from query import ask

# Example questions shown in the UI
EXAMPLE_QUESTIONS = [
    "What time should I go to Newell Hall to avoid the lunch rush?",
    "Which dining location has the best coffee on campus?",
    "Can I use my meal swipe at Tiger Den Chick-fil-A?",
    "What are the best options for students with celiac disease?",
    "Is there any food available on campus after 9pm?",
    "How do I make my Dining Dollars last the whole semester?",
    "What's the difference between Dining Dollars and Tiger Bucks?",
    "Is West Village Commons worth the walk from the main academic buildings?",
]


# Handler function
def handle_query(question: str) -> tuple[str, str, str]:
    """
    Process a user question through the RAG pipeline.
    Returns: (answer, sources_formatted, retrieval_debug)
    """
    if not question or not question.strip():
        return (
            "Please enter a question about Towson University dining.",
            "",
            "",
        )

    try:
        result = ask(question)

        answer = result["answer"]

        # Format sources for display
        sources_lines = []
        seen = set()
        for chunk in result["chunks"]:
            src = chunk["source"]
            dist = chunk["distance"]
            if src not in seen:
                seen.add(src)
                sources_lines.append(f"• {src}  (relevance score: {1 - dist:.2f})")

        sources_display = "\n".join(sources_lines) if sources_lines else "No sources retrieved."

        # Retrieval debug info
        debug_lines = ["Top retrieved chunks:"]
        for i, chunk in enumerate(result["chunks"]):
            preview = chunk["text"][:120].replace("\n", " ")
            debug_lines.append(
                f"\n[{i+1}] {chunk['source']} (distance: {chunk['distance']:.4f})\n"
                f"    {preview}..."
            )
        debug_display = "\n".join(debug_lines)

        return answer, sources_display, debug_display

    except Exception as e:
        error_msg = f"Error: {str(e)}\n\nMake sure you have:\n1. Run ingest.py\n2. Run embed.py\n3. Set GROQ_API_KEY in your .env file"
        return error_msg, "", ""


# Gradio UI
with gr.Blocks(
    title="Towson Dining Unofficial Guide",
    theme=gr.themes.Soft(),
    css="""
    .title-block { text-align: center; margin-bottom: 10px; }
    .subtitle { text-align: center; color: #666; font-size: 14px; margin-bottom: 20px; }
    """,
) as demo:

    gr.HTML("""
    <div class="title-block">
        <h1>🐯 Towson Dining Unofficial Guide</h1>
    </div>
    <div class="subtitle">
        Student-sourced knowledge about Towson University campus dining — searchable with plain-language questions.<br>
        <em>Answers are drawn from student reviews, Reddit threads, Discord tips, and student-written guides.</em>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What time should I go to Newell to avoid crowds?",
                lines=2,
            )

            with gr.Row():
                submit_btn = gr.Button("Ask the Guide", variant="primary")
                clear_btn = gr.Button("Clear")

            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,
            )

            sources_output = gr.Textbox(
                label="Retrieved from (sources)",
                lines=5,
                interactive=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Try asking:")
            for example in EXAMPLE_QUESTIONS:
                gr.Button(example, size="sm").click(
                    fn=lambda q=example: q,
                    outputs=question_input,
                )

    with gr.Accordion("🔍 Retrieval Debug (for evaluation)", open=False):
        debug_output = gr.Textbox(
            label="Top retrieved chunks",
            lines=12,
            interactive=False,
        )

    gr.HTML("""
    <div style="text-align:center; color:#999; font-size:12px; margin-top:20px;">
        Answers are grounded in student-generated documents collected from public sources.
        Always verify hours and prices at <a href="https://www.towson.edu/auxiliaryservices/dining/" target="_blank">towson.edu/dining</a>.
    </div>
    """)

    # Event bindings
    submit_btn.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, debug_output],
    )
    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, debug_output],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[question_input, answer_output, sources_output, debug_output],
    )


if __name__ == "__main__":
    print("Starting Towson Dining Unofficial Guide...")
    print("Make sure you have run ingest.py and embed.py first.")
    print("Open http://localhost:7860 in your browser.\n")
    demo.launch(server_name="0.0.0.0", server_port=7860)
