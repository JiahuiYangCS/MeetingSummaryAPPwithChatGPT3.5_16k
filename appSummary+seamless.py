from __future__ import annotations

import gradio as gr
import os
import tempfile
from summarizer_module import OpenAISummarizer
from PDFtoTXT_module import pdf_to_txt_file

import gradio as gr
import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from seamless_communication.models.inference.translator import Translator

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

DEFAULT_TARGET_LANGUAGE = "French"
DESCRIPTION2 = """# SeamlessM4T

[SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.

This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
translation and more, without relying on multiple separate models.

Text Input Max Token Limit = 1024 +- 200

Audio Input Max Length Limit = 60 s
"""

TASK_NAMES2 = [
    "S2ST (Speech to Speech translation)",
    "S2TT (Speech to Text translation)",
    "T2ST (Text to Speech translation)",
    "T2TT (Text to Text translation)",
    "ASR (Automatic Speech Recognition)",
]

# Language dict
language_code_to_name = {
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arb": "Modern Standard Arabic",
    "ary": "Moroccan Arabic",
    "arz": "Egyptian Arabic",
    "asm": "Assamese",
    "ast": "Asturian",
    "azj": "North Azerbaijani",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ceb": "Cebuano",
    "ces": "Czech",
    "ckb": "Central Kurdish",
    "cmn": "Mandarin Chinese",
    "cym": "Welsh",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "eng": "English",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "fra": "French",
    "gaz": "West Central Oromo",
    "gle": "Irish",
    "glg": "Galician",
    "guj": "Gujarati",
    "heb": "Hebrew",
    "hin": "Hindi",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "hye": "Armenian",
    "ibo": "Igbo",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "jav": "Javanese",
    "jpn": "Japanese",
    "kam": "Kamba",
    "kan": "Kannada",
    "kat": "Georgian",
    "kaz": "Kazakh",
    "kea": "Kabuverdianu",
    "khk": "Halh Mongolian",
    "khm": "Khmer",
    "kir": "Kyrgyz",
    "kor": "Korean",
    "lao": "Lao",
    "lit": "Lithuanian",
    "ltz": "Luxembourgish",
    "lug": "Ganda",
    "luo": "Luo",
    "lvs": "Standard Latvian",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mar": "Marathi",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mni": "Meitei",
    "mya": "Burmese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokm\u00e5l",
    "npi": "Nepali",
    "nya": "Nyanja",
    "oci": "Occitan",
    "ory": "Odia",
    "pan": "Punjabi",
    "pbt": "Southern Pashto",
    "pes": "Western Persian",
    "pol": "Polish",
    "por": "Portuguese",
    "ron": "Romanian",
    "rus": "Russian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "sna": "Shona",
    "snd": "Sindhi",
    "som": "Somali",
    "spa": "Spanish",
    "srp": "Serbian",
    "swe": "Swedish",
    "swh": "Swahili",
    "tam": "Tamil",
    "tel": "Telugu",
    "tgk": "Tajik",
    "tgl": "Tagalog",
    "tha": "Thai",
    "tur": "Turkish",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzn": "Northern Uzbek",
    "vie": "Vietnamese",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "yue": "Cantonese",
    "zlm": "Colloquial Malay",
    "zsm": "Standard Malay",
    "zul": "Zulu",
}
LANGUAGE_NAME_TO_CODE = {v: k for k, v in language_code_to_name.items()}

# Source langs: S2ST / S2TT / ASR don't need source lang
# T2TT / T2ST use this
text_source_language_codes = [
    "afr",
    "amh",
    "arb",
    "ary",
    "arz",
    "asm",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "ckb",
    "cmn",
    "cym",
    "dan",
    "deu",
    "ell",
    "eng",
    "est",
    "eus",
    "fin",
    "fra",
    "gaz",
    "gle",
    "glg",
    "guj",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kan",
    "kat",
    "kaz",
    "khk",
    "khm",
    "kir",
    "kor",
    "lao",
    "lit",
    "lug",
    "luo",
    "lvs",
    "mai",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mni",
    "mya",
    "nld",
    "nno",
    "nob",
    "npi",
    "nya",
    "ory",
    "pan",
    "pbt",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
    "yor",
    "yue",
    "zsm", ]
    
TEXT_SOURCE_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in text_source_language_codes]
)


# Target langs:
# S2ST / T2ST
s2st_target_language_codes = [
    "eng",
    "arb",
    "ben",
    "cat",
    "ces",
    "cmn",
    "cym",
    "dan",
    "deu",
    "est",
    "fin",
    "fra",
    "hin",
    "ind",
    "ita",
    "jpn",
    "kor",
    "mlt",
    "nld",
    "pes",
    "pol",
    "por",
    "ron",
    "rus",
    "slk",
    "spa",
    "swe",
    "swh",
    "tel",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "urd",
    "uzn",
    "vie",
]
S2ST_TARGET_LANGUAGE_NAMES = sorted(
    [language_code_to_name[code] for code in s2st_target_language_codes]
)
# S2TT / ASR
S2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES
# T2TT
T2TT_TARGET_LANGUAGE_NAMES = TEXT_SOURCE_LANGUAGE_NAMES


AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translator = Translator(
    model_name_or_card="seamlessM4T_large",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    dtype=torch.float16 if "cuda" in device.type else torch.float32,
)

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

def predict(
        task_name: str,
        audio_source: str,
        input_audio_mic: str | None,
        input_audio_file: str | None,
        input_text: str | None,
        source_language: str | None,
        target_language: str,
    ) -> tuple[tuple[int, np.ndarray] | None, str]:
        task_name = task_name.split()[0]
        source_language_code = (
            LANGUAGE_NAME_TO_CODE[source_language] if source_language else None
        )
        target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

        if task_name in ["S2ST", "S2TT", "ASR"]:
            if audio_source == "microphone":
                input_data = input_audio_mic
            else:
                input_data = input_audio_file

            arr, org_sr = torchaudio.load(input_data)
            new_arr = torchaudio.functional.resample(
                arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
            )
            max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
            if new_arr.shape[1] > max_length:
                new_arr = new_arr[:, :max_length]
                gr.Warning(
                    f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
                )
            torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
        else:
            input_data = input_text

        text_out, wav, sr = translator.predict(
            input=input_data,
            task_str=task_name,
            tgt_lang=target_language_code,
            src_lang=source_language_code,
            ngram_filtering=True,
        )

        if task_name in ["S2ST", "T2ST"]:
            return (sr, wav.cpu().detach().numpy()), text_out
        else:
            return None, text_out




def update_audio_ui(audio_source: str) -> tuple[dict, dict]:
    mic = audio_source == "microphone"
    return (
        gr.update(visible=mic, value=None),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )

def update_input_ui(task_name: str) -> tuple[dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    if task_name == "S2ST":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "S2TT":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "T2ST":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True,
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "T2TT":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True,
                choices=T2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    elif task_name == "ASR":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True,
                choices=S2TT_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            ),  # target_language
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_output_ui(task_name: str) -> tuple[dict, dict]:
    task_name = task_name.split()[0]
    if task_name in ["S2ST", "T2ST"]:
        return (
            gr.update(visible=True, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    elif task_name in ["S2TT", "T2TT", "ASR"]:
        return (
            gr.update(visible=False, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

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
    # First UI set
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
    file = gr.inputs.File(label="Upload PDF or TXT", optional=True)
    typed_text = gr.Textbox(placeholder="Type your text here...", label="Typed Text", optional=True, visible=True)
    btn1 = gr.Button("Generate for Summarize")
    with gr.Column():
        output_text1 = gr.Textbox(label="Summarized text")
    with gr.Column():
        output_file = gr.outputs.File(label="Download Summary")
    
    btn1.click(
        fn=summarize,
        inputs=[file, api_key, task, inputType, typed_text],
        outputs=[output_text1, output_file],
        api_name="run_summarize",
    )
    
    # Separator
    gr.Markdown("### ---")
    
    # Second UI set
    gr.Markdown(DESCRIPTION2)
    with gr.Group():
        task_name = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES2,
            value=TASK_NAMES2[0]
        )
    with gr.Row() as audio_box:
        source_language = gr.Dropdown(
            label="Source language",
            choices=TEXT_SOURCE_LANGUAGE_NAMES,
            value="English",
            visible=False
        )
        target_language = gr.Dropdown(
            label="Target language",
            choices=S2ST_TARGET_LANGUAGE_NAMES,
            value=DEFAULT_TARGET_LANGUAGE
        )
        audio_source = gr.Radio(
            label="Audio source",
            choices=["file", "microphone"],
            value="file"
        )
        input_audio_mic = gr.Audio(
            label="Input speech",
            type="filepath",
            source="microphone",
            visible=False
        )
        input_audio_file = gr.Audio(
            label="Input speech",
            type="filepath",
            source="upload",
            visible=True
        )
        input_text = gr.Textbox(label="Input text", visible=False)
    btn2 = gr.Button("Translate Audio")
    with gr.Column():
        output_audio = gr.Audio(
            label="Translated speech",
            autoplay=False,
            streaming=False,
            type="numpy"
        )
        output_text2 = gr.Textbox(label="Translated text")
    
    # Interaction events for second UI set
    audio_source.change(
        fn=update_audio_ui,
        inputs=audio_source,
        outputs=[input_audio_mic, input_audio_file],
        queue=False,
        api_name=False,
    )

    task_name.change(
        fn=update_input_ui,
        inputs=task_name,
        outputs=[audio_box, input_text, source_language, target_language],
        queue=False,
        api_name=False,
    ).then(
        fn=update_output_ui,
        inputs=task_name,
        outputs=[output_audio, output_text2],
        queue=False,
        api_name=False,
    )

    btn2.click(
        fn=predict,
        inputs=[task_name, audio_source, input_audio_mic, input_audio_file, input_text, source_language, target_language],
        outputs=[output_audio, output_text2],
        api_name="run_translation",
    )



if __name__ == "__main__":
    demo.queue().launch()


