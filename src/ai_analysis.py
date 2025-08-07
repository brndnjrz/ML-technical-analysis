import ollama
import base64
import io
import tempfile

# Integrate AI Analysis 
def run_ai_analysis(fig, prompt):
    """Run AI analysis on a chart"""
    # Convert the figure to a base64 string directly
    
    buf = io.BytesIO()
    fig.write_image(buf, format='png')
    buf.seek(0)
    image_data = base64.b64encode(buf.read()).decode('utf-8')

    # Prepare AI analysis request
    messages = [{
        'role': 'user',
        'content': prompt,
        'images': [image_data]
    }]
    response = ollama.chat(model='llama3.2-vision', messages=messages)

    return response["message"]["content"]
