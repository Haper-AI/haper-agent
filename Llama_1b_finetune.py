from huggingface_hub import notebook_login
notebook_login()

import json
import pandas as pd
from datasets import Dataset
import random

json_path = "/content/drive/MyDrive/AI agent/all_emails.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 随机添加 category 字段
for item in data:
    item["category"] = random.choice(["important", "non-important"])

# 保存为新的 JSON 文件
new_json_path = "/content/drive/MyDrive/AI agent/all_emails_with_category.json"
with open(new_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

import json
import pandas as pd
from datasets import Dataset

# 读取标注后的 JSON 文件
json_path = "/content/drive/MyDrive/AI agent/all_emails_with_category.json"  # 修改为你的 JSON 文件路径
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换成 Pandas DataFrame
df = pd.DataFrame(data)

# 映射 category 到 0/1
df["label"] = df["category"].map({"important": 1, "non-important": 0})

# 拼接多输入字段
df["text"] = "From: " + df["From"] + "\nSubject: " + df["Subject"] + "\nDate: " + df["Date"] + "\nContent: " + df["Content"]

# 删除不必要的列
df = df[["text", "label"]]

# 转换成 Hugging Face Dataset 格式
dataset = Dataset.from_pandas(df)

from transformers import AutoModelForCausalLM, AutoTokenizer

# 下载 LLaMA-1B
model_id = "meta-llama/Llama-3.2-1B"  # 确保 LLaMA-1B 可用，否则换成 LLaMA-7B 或更小的模型


model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token

# Tokenize 数据
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 训练/测试集拆分
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

print(test_dataset)
print(len(train_dataset[0]['input_ids']))
print(len(train_dataset[1]['input_ids']))

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from peft import LoraConfig, get_peft_model
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# 加载 4-bit 量化 LLaMA-1B
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    num_labels=2,  # 二分类
    device_map="auto",
    max_memory={0: "12GB", "cpu": "16GB"}
)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
# LoRA 配置
lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], bias="none")

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练参数
training_args = TrainingArguments(
    output_dir="./qlora_llama1b",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    label_names=["labels"],
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    report_to="tensorboard",
    logging_dir="./logs",
    logging_steps=1,
    eval_steps=1,
    fp16=True,
    optim="adamw_torch",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    eval_accuracy = accuracy_score(labels, preds)
    return {"accuracy": eval_accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,

)

trainer.train()

import torch

def classify_email(email):
    email_text = f"Sender: {email['sender']}\nSubject: {email['subject']}\nBody: {email['body']}"

    inputs = tokenizer(email_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class = torch.argmax(logits, dim=-1).item()
    return "important" if predicted_class == 1 else "non-important"

# 测试
test_email = {
    "sender": "hr@company.com",
    "subject": "Meeting Reminder",
    "body": "Reminder: Team meeting tomorrow at 10 AM."
}
print(f"📧 邮件分类: {classify_email(test_email)}")

from huggingface_hub import login


# 上传模型
model.push_to_hub("YULI1234/llama-1b-finetuned")
tokenizer.push_to_hub("YULI1234/llama-1b-finetuned")

eval_results = trainer.evaluate()
print(eval_results)
