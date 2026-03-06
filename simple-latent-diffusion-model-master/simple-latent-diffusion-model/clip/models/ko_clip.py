import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

class KoCLIPWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "Bingsu/clip-vit-base-patch32-ko"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
    def loss(self, inputs):
        outputs = self(inputs)
        return outputs.loss

    def text_encode(self, text, tokenize):
        if tokenize:
            tokens = self.tokenizer(text, padding='max_length', max_length=77, truncation=True, return_tensors="pt")
        else:
            tokens = text
        tokens = tokens.to(self.model.device)
        return self.model.get_text_features(**tokens)
    
    def forward(self, inputs):
        outputs = self.model(**inputs, return_loss=True)
        return outputs # [1, 512] , [1, 512]