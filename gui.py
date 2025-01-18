import gradio as gr
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy


def handleGUI(chat):
    """
        Creeaza o interfata grafica Gradio pentru chatbot.

        Args:
            chat (Chat): Obictul de tip Chat.

        Returns:
            None
        """
    # Initialize the evaluator
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    dropdown_choices = ["DBPedia", "COSTAR"]


    default_costar = {
        "CONTEXT": "",
        "OBJECTIVE": "The objective is to be as short as possible; respond only to the queston I will give you, no extra details.",
        "STYLE": "Your style is conversational and accessible, ensuring that the language is easy to understand for a general audience.",
        "TONE": "Maintain a neutral tone that inspires trust and conveys authority.",
        "AUDIENCE": "The audience is general public, with responses tailored to be beginner-friendly.",
        "RESPONSE": "Provide clear, conversational plain text response."
    }

    # Create a Gradio interface with a dropdown and text input
    with gr.Blocks() as demo:
        gr.Markdown("### RAG Chatbot")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Your Message", lines=5, visible=True)

                dropdown = gr.Dropdown(
                    label="Select Chat Type",
                    choices=dropdown_choices,
                    value="DBPedia"  # Default selection
                )

                costar_inputs = [
                    gr.Textbox(label=f"Context", lines=1,
                               value=default_costar['CONTEXT'], visible=False, max_lines=10),
                    gr.Textbox(label=f"Objective", lines=1,
                               value=default_costar['OBJECTIVE'], visible=False, max_lines=5),
                    gr.Textbox(
                        label=f"Style", lines=1, value=default_costar['STYLE'], visible=False, max_lines=5),
                    gr.Textbox(
                        label=f"Tone", lines=1, value=default_costar['TONE'], visible=False, max_lines=5),
                    gr.Textbox(label=f"Audience", lines=1,
                               value=default_costar['AUDIENCE'], visible=False, max_lines=5),
                    gr.Textbox(label=f"Response", lines=1,
                               value=default_costar['RESPONSE'], visible=False, max_lines=5),
                ]

                def toggle_costar(selected_option):
                    if selected_option == "COSTAR":
                        return [gr.update(visible=True, interactive=True) for _ in costar_inputs] + [gr.update(visible=True, interactive=True)]
                    else:
                        return [gr.update(visible=False) for _ in costar_inputs] + [gr.update(visible=True)]

                dropdown.change(
                    toggle_costar,
                    inputs=[dropdown],
                    outputs=costar_inputs + [text_input]
                )

                async def evaluate_relevancy(your_message, context, response):
                    sample = SingleTurnSample(
                        user_input=your_message,
                        response=response,
                        retrieved_contexts=[context]
                    )
                    scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
                    score = await scorer.single_turn_ascore(sample)
                    return score

                with gr.Row():
                    clear_button = gr.Button("Clear")
                    clear_button.variant = "secondary"

                    def clear_inputs():
                        return "", "", "", "", "", "", ""  # Clear all textboxes

                    clear_button.click(
                        clear_inputs,
                        inputs=[],
                        outputs=[text_input] + costar_inputs
                    )

                    submit_button = gr.Button("Submit")
                    submit_button.variant = "primary"

            with gr.Column(scale=1):
                output = gr.Textbox(label="Chatbot Response", lines=5)
                relevancy_score = gr.Textbox(label="Relevancy Score", lines=1, interactive=False)

        def chatbot_inference(selected_option, user_input, *costar_inputs):
            chat.updateCostarData(selected_option, *costar_inputs)
            response = chat.predict(user_input)
            return response

        async def combined_inference(dropdown, text_input, *costar_inputs):
            response = chatbot_inference(dropdown, text_input, *costar_inputs)
            relevancy = await evaluate_relevancy(text_input, costar_inputs[0], response)
            return response, relevancy
        
        submit_button.click(
            combined_inference,
            inputs=[dropdown, text_input, *costar_inputs],
            outputs=[output, relevancy_score]
        )
    # Launch the Gradio interface
    demo.launch(share=True)
