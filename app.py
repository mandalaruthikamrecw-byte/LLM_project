import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "tinyllama_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(base_model, ADAPTER)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()


def chatbot(question):

    prompt = f"""### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Answer:" in response:
        response = response.split("### Answer:")[-1]

    return response.strip()


demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask any Real Estate question..."
    ),
    outputs=gr.Textbox(lines=6),
    title="🏠 Real Estate AI Chatbot",
    description="Fine-tuned TinyLlama Real Estate Assistant"
)

demo.launch()
