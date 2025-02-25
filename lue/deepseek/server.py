from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "deepseek-ai/deepseek-llm-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

@app.post("/predict")
def predict(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=100)
    return {"response": tokenizer.decode(output[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

