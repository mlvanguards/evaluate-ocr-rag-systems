import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor


def create_model(model_name: str = "vidore/colqwen2-v0.1"):
    """
    Load a pre-trained ColQwen2 model and processor.

    Args:
        model_name: The name of the pre-trained model to load (default: "vidore/colqwen2-v0.1")

    Returns:
        A tuple (model, processor) containing the pre-trained model and processor
    """
    model = ColQwen2.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = ColQwen2Processor.from_pretrained(model_name)
    model.eval()
    return model, processor
