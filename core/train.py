from transformers import LayoutLMv3Processor,LayoutLMForQuestionAnswering
from PIL import Image
from datasets import load_dataset
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
