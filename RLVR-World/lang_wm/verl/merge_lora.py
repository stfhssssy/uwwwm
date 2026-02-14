from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("DeepSeek-R1-Distill-Qwen-1.5B")
peft_model_id = "log/webagent-sft-DeepSeek-R1-Distill-Qwen-1.5B/global_step_xxxxxx"  # the output path
model = PeftModel.from_pretrained(base_model, peft_model_id)
merged_model = model.merge_and_unload()
print(type(merged_model))
merged_model.save_pretrained("webagent-sft-DeepSeek-R1-Distill-Qwen-1.5B-merged-lowest")  # where to save the merged model