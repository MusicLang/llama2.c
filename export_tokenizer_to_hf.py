from tokenizers import SentencePieceUnigramTokenizer, SentencePieceBPETokenizer
import sentencepiece as spm
from tokenizers import normalizers
from transformers import AutoTokenizer
import os
import glob
import shutil
from transformers import LlamaForCausalLM

"""
Load vocab and merges from result
Steps to launch : 
- Copy to {path} the ckpt.pt, the tokenizer.model (renamed appropriately) and the model.bin (for tests)
- python export_tokenizer_to_hf.py
- python test_hf_model.py
"""
path = 'data/result2'
output_hf = 'musiclang/control_masking_optimized_trained' # Change this to the path where you want to save the model

tokenizer_name = 'tokenizer'
files_to_copy = glob.glob('base_tokenizer/*')
extract = True

if not os.path.exists(path):
    os.makedirs(path)

def extract_model(output_hf):
    program = f'python export.py {path} --version -1 --dtype fp32 --checkpoint {path}/ckpt.pt'
    res = os.system(program)
    print(res)
    model = LlamaForCausalLM.from_pretrained(path)
    # Push to hub
    model.save_pretrained(output_hf, push_to_hub=True)

    # Call the program

extract_model(output_hf)


if extract:
    program = f'python sentencepiece_extractor.py --provider sentencepiece --vocab-output-path {path}/vocab.json --merges-output-path {path}/merges.json --model {path}/{tokenizer_name}.model'
    # Call the program
    os.system(program)


LOCAL_MODEL = f'{path}/tokenizer.model'
FINAL_MODEL = f'{path}'
vocab_file = f'{path}/vocab.json'
merges_file = f'{path}/merges.json'

tok = SentencePieceBPETokenizer.from_file(vocab_file, merges_file)
tok.normalizer = normalizers.Sequence([])

tok.save(os.path.join(FINAL_MODEL, 'tokenizer.json'))

# Copy all files
for file in files_to_copy:
    shutil.copy(file, FINAL_MODEL)


hf_tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL)
#hf_tokenizer = tok
spm_tokenizer = spm.SentencePieceProcessor(model_file=LOCAL_MODEL)

# Test the tokenization
x = "ʝ1H<NÅɓʌʚǄõƧɥʌʛǄõƧɷʌʚǄõƧò1H<NÅǋʌʘǄõƄɓʌʖǄôƄò1H<NÅɓʌʚǄõƲɧʌʛǄõƑɸʌʚǄõƲò1H<NÅǋʌʘǄõƄɓʌʖǄõŷʂʌʑǄõŨò1H<NÅñ1H<NÅñ1H<NÅñ1H<NÅñ1H<NÅɶʌʓǃóƧɓʌʚǄõŀɥʌʛǄõƧɮʌʓǄõƑɻʌʓǄõƄò1H<NÅɋʌʚǄõŻɓʌʛǄôŪɮʌʓǄõƧɿʌʛǄõųò1H<NÅɥʌʚǄõƧɷʌʚǄõƧò1H<NÅǋʌʚǄõŀȏʌʘǄõŻȹʌʖǄóŀɋʌʘǄõŦɮʌʖǄõƧɿʌʑǄôųò1H<NÅɉʌʖǄóŻɓʌʚǄõƧɥʌʛǄõƧɷʌʚǄõƧò1H<NÅǋʌʘǄõƄɓʌʖǄõƄò1H<NÅǋʌʑǄõƄɓʌʓǃõŪɷʌʘǃôŮò1H<NÅòʜ"

hf_encode = hf_tokenizer.encode(x, add_special_tokens=False) #
spm_encode = spm_tokenizer.encode_as_pieces(x)
ids_hf = hf_tokenizer.encode(x, add_special_tokens=False)
ids_spm = spm_tokenizer.encode(x)

decode_hf = hf_tokenizer.decode(ids_hf)
decode_spm = spm_tokenizer.decode(ids_spm)

wrong_idxs = [idx for idx, (c,d) in enumerate(zip(ids_hf, ids_spm)) if c != d]
assert len(ids_hf) == len(ids_spm), "Mismatch between HF and SPM tokenization"
assert all([c==d for c,d in zip(ids_hf, ids_spm)]), "Mismatch between HF and SPM tokenization"
assert decode_hf == decode_spm, "Mismatch between HF and SPM tokenization"

# Upload tokenizer to HF at output_hf (to the hub)
hf_tokenizer.save_pretrained(output_hf, push_to_hub=True)
print('GOOD!')

