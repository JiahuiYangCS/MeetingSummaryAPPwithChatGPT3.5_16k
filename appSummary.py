import gradio as gr
import os
import tempfile
from summarizer_module import OpenAISummarizer
from PDFtoTXT_module import pdf_to_txt_file

TASK_NAMES = [
    "Generate English Meeting Summary from recording",
    "Generate Chinese Meeting Summary from recording",
    "Generate English Job Description from recording",
    "Generate Chinese Job Description from recording",
]

INPUT_TYPE = [
    "PDF/text file",
    "typed text",
]

def summarize(file, api_key, task, inputType, typed_text=None):
    
    if inputType == "PDF/text file" and file:
        if file.name.endswith('.pdf'):
            txt_path = pdf_to_txt_file(file.name, 'PDFoutput.txt')
        elif file.name.endswith('.txt'):
            txt_path = file.name
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or text file.")

    elif inputType == "typed text" and typed_text:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with open(temp_file.name, 'w', encoding='utf-8') as f:
            f.write(typed_text)
        txt_path = temp_file.name

    if not txt_path:
        raise ValueError("No valid input provided.")

    summarizer = OpenAISummarizer(api_key)
    
    if task == "Generate English Meeting Summary from recording":
        result = summarizer.summarizeEN(txt_path)
    elif task == "Generate Chinese Meeting Summary from recording":
        result = summarizer.summarizeCN(txt_path)
    elif task == "Generate English Job Description from recording":
        result = summarizer.summarizeJDEN(txt_path)
    elif task == "Generate Chinese Job Description from recording":
        result = summarizer.summarizeJDCN(txt_path)
    else:
        raise ValueError(f"Unsupported task: {task}")

   
    # Save the result to another temporary file
    summary_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(summary_file.name, 'w', encoding='utf-8') as f:
        f.write(result)

    return result, summary_file.name


DESCRIPTION = """# MeetingSummary/JD Generate

Powered by ChatGPT3.5

api_key: openAI api key

Using ChatGPT 3.5 Turbo 16k large language model

Using the Langchain freamwork with map and reduce summarization methods

Map and reduce prompt words can be modified in the summarize function of the summarizer_module.py file as needed

Input type: PDF, .txt file, and window typing in

"""
css = """
h1 {
  text-align: center;
}

.contain {
  max-width: 730px;
  margin: auto;
  padding-top: 1.5rem;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Group():
        task = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES,
            value=TASK_NAMES[0]
        )
    with gr.Column():
        inputType = gr.Dropdown(
            label="INPUT_TYPE",
            choices=INPUT_TYPE,
            value="PDF/text file",
            visible=True
        )
        
    api_key = gr.Textbox(label="openai api key", visible=True)
        
    file = gr.inputs.File(label="Upload PDF or TXT",optional=True)
        
    typed_text = gr.Textbox(placeholder="Type your text here...", label="Typed Text", optional=True, visible=True)
        
    btn = gr.Button("Generate")
        
    with gr.Column():
        output_text = gr.Textbox(label="Translated text")

    with gr.Column():
        output_file = gr.outputs.File(label="Download Summary")# Allow users to download the output
    
    btn.click(
        fn=summarize,
        inputs=[
            file, api_key, task, inputType, typed_text,
        ],
        outputs=[output_text, output_file],
        api_name="run",
    )
    
if __name__ == "__main__":
    demo.queue().launch()