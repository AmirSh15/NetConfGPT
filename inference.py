import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

MAX_OUTPUT_LENGTH = 100

# Prepare quantized config
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model
model_id =  "cerebras/btlm-3b-8k-base" # "meta-llama/Llama-2-70b-chat-hf" # "tiiuae/falcon-40b"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # load_in_4bit=True, # for QLoRA
    quantization_config=nf4_config,
    trust_remote_code=True,
    device_map="auto",
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# define pipeline config
pipeline_kwargs = {
    "max_new_tokens": 10,
    "task": "text-generation",
    "device": -1,
    "model_kwargs": None,
    "pipeline_kwargs": None,
    "model_id": model_id,
        
}

# define pipeline
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=MAX_OUTPUT_LENGTH,
    model_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "num_return_sequences": 5, "max_length":MAX_OUTPUT_LENGTH},
)

llm = HuggingFacePipeline(pipeline=pipe)

# template = """Question: {question}

# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)
template = """ You are going to be my assistant.
Please try to give me the most beneficial answers to my
question with reasoning for why they are correct.

 Question: {question} Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm

question = "What is the YANG file?"

output_lc = chain.invoke({"question": question})

# save the output in a text file
with open("output.txt", "w") as f:
    f.write(output_lc)



# inference model
prompt = "What is the YANG file?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
# move to cuda
input_ids = input_ids.to("cuda")

# inference
output = model.generate(
    input_ids,
    do_sample=True,
    max_length=MAX_OUTPUT_LENGTH,
    top_k=50,
    top_p=0.95,
    num_return_sequences=5,
    temperature=0.7,
)

# decode
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
print(output_text)
