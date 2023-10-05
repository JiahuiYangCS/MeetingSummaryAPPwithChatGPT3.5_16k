import PyPDF2


def pdf_to_txt_file(pdf_path, PDFoutput_txt_path):
    """
    Convert a PDF to a TXT file and save it.
    
    Parameters:
    - pdf_path (str): Path to the PDF file.
    - output_txt_path (str): Path to save the resulting TXT file.
    
    Returns:
    - str: Path of the saved TXT file.
    """
    
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Check if the PDF is encrypted
        if pdf_reader.is_encrypted:
            pdf_reader.decrypt('')
        
        # Extract text from each page
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Save the extracted text to a TXT file
    with open(PDFoutput_txt_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    return PDFoutput_txt_path


'''
PDFoutput_txt_path = pdf_to_txt_file('Altomni AI Engineer & Data Scientist Intern Offer_Jiahui Yang.pdf', 'PDFoutput.txt')
print(output_txt_path)  # This will print the path of the saved TXT file
'''