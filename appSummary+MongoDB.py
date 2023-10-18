import gradio as gr
import os
import tempfile
from summarizer_module import OpenAISummarizer
from PDFtoTXT_module import pdf_to_txt_file

from datetime import datetime
import openai
from pymongo import MongoClient




# connecting to MangoDB
# setting MangoDB here

client = MongoClient('localhost', 27017)

db = client['your_database_name']

collectionMeeting = db['MeetingSummary']
collectionJD = db['JD']


def save_summary_in_DB(output_text, api_key):
    
    openai.api_key = api_key
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that converts text to JSON format."
        },
        {
            "role": "user",
            "content": f"Convert the following meeting summary into JSON format:\n{output_text}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=10000  # output limit, adjustable 
    )
    
    json_content = response.choices[0].message['content'].strip()

    
    
    
    today_date = datetime.now().strftime('%Y-%m-%d')
    collectionMeeting.insert_one({"date": today_date, "content": json_content})
    
    dataName = today_date
    successfulText = f"Meeting summary {dataName} has been converted and stored in MongoDB!"
    
    
    return successfulText, dataName


def save_JD_in_DB(output_text, api_key):
    
    openai.api_key = api_key
    
    job_description = output_text

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that extracts specific details from job descriptions."
        },
        {
            "role": "user",
            "content": f"""
            From the provided job description, extract the following details:

            Job Title: {{title}}
            Responsibilities: {{responsibilities}}
            Requirements: {{requirements}}
            Duration: {{duration}}
            Salary: {{salary}}

            {job_description}
            """
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=4000  # output limit, adjustable 
    )

    
    JDoutput_text = response.choices[0]['message']['content'].strip().split("\n")

    parsed_data = {line.split(": ")[0]: line.split(": ")[1] for line in JDoutput_text}

    job_data = {
        "_id": parsed_data["Job Title"],
        "responsibilities": parsed_data["Responsibilities"].split('; '),
        "requirements": parsed_data["Requirements"].split('; '),
        "duration": parsed_data["Duration"],
        "salary": parsed_data["Salary"]
    }

    collectionJD.insert_one(job_data)
    
    successfulText = f"Job description for {parsed_data['Job Title']} has been stored in MongoDB!"
    dataName = parsed_data['Job Title']
    
    return successfulText, dataName



def saveInMangodb(task, output_text, api_key):
    if task in ["Generate English Meeting Summary from recording", "Generate Chinese Meeting Summary from recording"]:
        return save_summary_in_DB(output_text, api_key)
    elif task in ["Generate English Job Description from recording", "Generate Chinese Job Description from recording"]:
        return save_JD_in_DB(output_text, api_key)
    else:
        print("Invalid task specified.")
        return None
    
    

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

 
    
    

#################### UI ##########################


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
    
    btnDB = gr.Button("Save the result/data to data base(Mangodb)")
    
    with gr.Column():
        successfulText = gr.Textbox(label="Storage Success Indicator")
        dataName = gr.Textbox(label="data/record named as: ")
    
    btn.click(
        fn=summarize,
        inputs=[
            file, api_key, task, inputType, typed_text,
        ],
        outputs=[output_text, output_file],
        api_name="run",
    )
    
    
    btnDB.click(
        fn=saveInMangodb,  
        inputs=[
            task, output_text, api_key
        ],
        outputs=[successfulText, dataName],
        api_name="run",
    )

    
if __name__ == "__main__":
    demo.queue().launch()