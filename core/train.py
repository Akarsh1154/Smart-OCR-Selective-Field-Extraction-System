from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering, Trainer, TrainingArguments
import torch
from PIL import Image
from datasets import load_dataset
from core.preprocessing import clean_image
import os
import json

DOCUMENT_TYPE = "receipt"    # change this each run

DATASET_MAP = {
    "receipt": "naver-clova-ix/cord-v2",
    "invoice": "darentang/sroie",
}

FIELD_QUESTIONS = {
    "total.total_price": "What is the total price?",
    "sub_total.subtotal_price": "What is the subtotal?",
    "sub_total.tax_price": "What is the tax?",
    "sub_total.service_price": "What is the service charge?",
}

MODEL_NAME = "microsoft/layoutlmv3-base"
SAVE_PATH = f"/content/drive/MyDrive/models/{DOCUMENT_TYPE}_model"
CHECKPOINT_PATH = f"/content/drive/MyDrive/checkpoints/{DOCUMENT_TYPE}_model"
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-5
SAVE_TOTAL_LIMIT = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=True)
if os.path.exists(SAVE_PATH):
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(SAVE_PATH)
else:
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(MODEL_NAME)
model.to(DEVICE)

dataset = load_dataset(DATASET_MAP[DOCUMENT_TYPE])

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]


def preprocess(example):
    image = clean_image(example["image"])
    gt = json.loads(example["ground_truth"])

    encoding = None
    for field, question in FIELD_QUESTIONS.items():
        keys = field.split(".")
        try:
            answer = str(gt["gt_parse"][keys[0]][keys[1]])
        except Exception:
            continue

        encoding = processor(
            image,
            question,
            padding="max_length",
            truncation=True
        )

        # After
        ans_tokens = processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.tokenize(answer)
        )
        input_ids = encoding["input_ids"]
        start_pos = 0
        end_pos = 0

        for i in range(len(input_ids)):
            if input_ids[i: i + len(ans_tokens)] == ans_tokens:
                start_pos = i
                end_pos = i + len(ans_tokens) - 1
                break

        encoding["start_positions"] = start_pos
        encoding["end_positions"] = end_pos
        break  # use first valid field per example

    return encoding


train_dataset = train_dataset.map(preprocess)
validation_dataset = validation_dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    save_strategy="epoch",
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor
)

trainer.train(resume_from_checkpoint=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None)
trainer.evaluate()

model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)