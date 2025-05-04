import ollama
import base64
import tempfile

# Integrate AI Analysis 
def run_ai_analysis(fig, prompt):
    # Saves the Plotly Chart as a PNG in a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.write_image(tmpfile.name)
        tmpfile_path = tmpfile.name

    # Read image and encode to Base64
    with open(tmpfile_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Prepare AI analysis request
    messages = [{
        'role': 'user',
        'content': prompt,
        'images': [image_data]
    }]
    response = ollama.chat(model='llama3.2-vision', messages=messages)

    return response["message"]["content"], tmpfile_path
