import torch
import transformers
import gradio as gr
from transformers import AutoModelForCausalLM, GPT2Tokenizer, Thread, TextIteratorStreamer

model_name = "odu"
model = AutoModelForCausalLM.from_pretrained("Icetist/model")
tokenizer = GPT2Tokenizer.from_pretrained("Icetist/model")

# Alpaca prompt template
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:
"""

def generate_response(query, input_context="", max_new_tokens=128):
    prompt = ALPACA_PROMPT.format(instruction=query, input=input_context)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
    return generated_text

def chat(query, history):
    history = history or []
    response = generate_response(query)
    history.append((query, response))
    return history, ""  # Return updated history and empty string to clear input

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Enter your query", placeholder="Type your message here...")
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear Conversation")

    submit_button.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_button.click(lambda: ([], ""), outputs=[chatbot, msg], queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True, inbrowser=True)