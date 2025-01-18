import gradio as gr


def handleGUI(chat):
    """
        Creeaza o interfata grafica Gradio pentru chatbot.

        Args:
            chat (Chat): Obictul de tip Chat.

        Returns:
            None
        """

    dropdown_choices = ["DBPedia", "COSTAR"]

    def chatbot_interface(selected_option, user_input, c=None, o=None, s=None, t=None, a=None, r=None):
        chat.updateCostarData(selected_option, c, o, s, t, a, r)

        response = chat.predict(user_input)

        return response

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

        submit_button.click(chatbot_interface, inputs=[
                            dropdown, text_input, *costar_inputs], outputs=output)

    # Launch the Gradio interface
    demo.launch(share=True)
