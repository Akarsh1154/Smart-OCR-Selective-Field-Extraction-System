from transformers import LayoutLMv3Processor,LayoutLMv3ForQuestionAnswering,Trainer,TrainingArguments
import torch 
from PIL import Image 
from datasets import load_dataset
from preprocessing import clean_image
import os

DOCUMENT_TYPE = "receipt"    # change this each run

DATASET_MAP = {
    "receipt": "naver-clova-ix/cord-v2",
    "invoice": "darentang/sroie",
}

MODEL_NAME = "microsoft/layoutlmv3-base"
SAVE_PATH = f"/content/drive/MyDrive/models/{DOCUMENT_TYPE}_model"
CHECKPOINT_PATH = f"/content/drive/MyDrive/checkpoints/{DOCUMENT_TYPE}_model"
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 5e-5
SAVE_TOTAL_LIMIT = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME,apply_ocr=True)
if os.path.exists(SAVE_PATH):
    model=LayoutLMv3ForQuestionAnswering.from_pretrained(SAVE_PATH)
else:
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(MODEL_NAME)
model.to(DEVICE)

dataset = load_dataset(DATASET_MAP[DOCUMENT_TYPE])

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

def preprocess(example):
    image = clean_image(example["image"])
    encoding=processor(
        image,
        example["question"],
        padding="max_length",truncation = True
    )
    ans_text = example["answers"]["text"][0]
    ans_tokens = processor.tokenizer.encode(ans_text)
    start_pos = 0
    end_pos = 0 
    input_ids = encoding["input_ids"]

    for i in range(len(input_ids)):
        if input_ids[i: i + len(ans_tokens)] == ans_tokens:
            start_pos = i
            end_pos = i + len(ans_tokens) - 1
            break
    encoding["start_positions"] = start_pos
    encoding["end_positions"] = end_pos 
    return encoding

train_dataset=train_dataset.map(preprocess)
validation_dataset=validation_dataset.map(preprocess)

training_args = TrainingArguments(
    output_dir = CHECKPOINT_PATH,
    num_train_epochs = EPOCHS,
    per_device_train_batch_size= BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    save_strategy = "epoch",
    save_total_limit = SAVE_TOTAL_LIMIT,
    load_best_model_at_end = True,
    fp16 = True
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset,
    tokenizer = processor
)

trained = trainer.train()
trained_eval = trainer.evaluate()

model.save_pretrained(resume_from_checkpoint=CHECKPOINT_PATH)
processor.save_pretrained(SAVE_PATH)