from transformers import LlamaForCausalLM, AutoTokenizer

path = 'data/result'
model_path = 'musiclang/control_masking_optimized_trained'

model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.pad_token_id=0

# Test the model
prompt = "ʝ1H<NÅɓʌʚǄõƧɥʌʛǄõƧɷʌ"
prompt = "ʝ1H"
ids = tokenizer.encode(prompt, return_tensors='pt')
# Generate text
output = model.generate(ids, max_length=100, do_sample=False, temperature=0.0, pad_token_id=0)

output = output[0].tolist()

text_output = tokenizer.decode(output, skip_special_tokens=True)
print(output)
print(text_output)
# Check the output with original model
program = f'./run data/result2/model.bin -z data/result/tok16000.bin -t 0.0 -n 256 -i "{prompt}"'
import os
result = os.popen(program).read()
print(result)
from pdb import set_trace; set_trace()