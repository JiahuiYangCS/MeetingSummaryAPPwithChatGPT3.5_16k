import gradio as gr
import os
import tempfile
from summarizer_module import OpenAISummarizer
from PDFtoTXT_module import pdf_to_txt_file

def summarize_and_cleanup(file, api_key, action, typed_text=None):
    if action == "Summarize":
        # Check the file type
        if file and file.name.endswith('.pdf'):
            txt_path = pdf_to_txt_file(file.name, 'PDFoutput.txt')
        else:
            txt_path = file.name if file else None

    elif action == "Summary from typed text":
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with open(temp_file.name, 'w', encoding='utf-8') as f:
            f.write(typed_text)
        txt_path = temp_file.name

    summarizer = OpenAISummarizer(api_key)
    result = summarizer.summarize(txt_path)

    # Save the result to another temporary file
    summary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(summary_file.name, 'w', encoding='utf-8') as f:
        f.write(result)

    return result, summary_file.name, ""


title = """
<strong><span style="font-size: larger;">Meeting Summarizer powered by ChatGPT3.5</span></strong><br>
<span style="font-size: smaller;">api_key: openAI api key<br>
output 0: Display summarized text<br>
output 2: Temporary file deletion indicator<br>
Using ChatGPT 3.5 Turbo 16k large language model<br>
Using the Langchain freamwork with map and reduce summarization methods<br>
Map and reduce prompt words can be modified in the summarize function of the summarizer_module.py file as needed</span>
"""

# Define the gr.Interface with the provided title
interface = gr.Interface(
    
    fn=summarize_and_cleanup, 
    inputs=[
        gr.inputs.File(label="Upload PDF or TXT", optional=True),
        "text",  # A simple text input for the API key
        gr.inputs.Dropdown(choices=["Summarize", "Summary from typed text", "Delete Temporary File"], label="Action"),
        gr.inputs.Textbox(lines=5, placeholder="Type your text here...", label="Typed Text", optional=True)
    ], 
    outputs=[
        "text",  # Display the summary in a textbox
        gr.outputs.File(label="Download Summary"),  # Allow users to download the output
        "text"  # Display message after deletion
    ],
    live=False,  # Only run the function when the submit button is pressed
    flagging_options=None,  # Remove the "Flag" button
    title=title,  # Use the provided title with custom styles
    layout="unaligned"  # Align content to the left
)

interface.launch()