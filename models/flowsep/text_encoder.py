import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel
import math


class FlanT5HiddenState(nn.Module):
    def __init__(self, text_encoder_name="google/flan-t5-large", freeze_text_encoder=True, emb_num=3, return_length=50, input_caption=False, all_pos=False):
        super().__init__()
        self.emb_num = emb_num
        self.return_length = return_length
        self.freeze_text_encoder = freeze_text_encoder
        self.text_encoder_name = text_encoder_name
        self.all_pos = all_pos
        if self.all_pos:
            self.position_embedding = self.add_position_embedding(self.return_length*self.emb_num, 1024)

        cache_dir = os.getenv("TRANSFORMERS_CACHE", os.getenv("HF_HOME"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name, cache_dir=cache_dir)
        self.model = T5EncoderModel.from_pretrained(self.text_encoder_name, cache_dir=cache_dir)
        self.input_caption = input_caption
        if self.model:
            if freeze_text_encoder:
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False

        self.empty_hidden_state_cfg = None
        self.device = None

    def add_position_embedding(self, max_sequence_length, embedding_dim):
        position_embeddings = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)
        return position_embeddings

    def get_unconditional_condition(self, batchsize):
        param = next(self.model.parameters())
        if self.freeze_text_encoder:
            assert param.requires_grad == False
        
        if self.empty_hidden_state_cfg is None:
            self.empty_hidden_state_cfg, _ = self([""]*self.emb_num)
        
        hidden_state = torch.cat([self.empty_hidden_state_cfg] * batchsize).float()
        attention_mask = torch.ones((batchsize, hidden_state.size(1))).to(hidden_state.device).float()
        return [hidden_state, attention_mask]

    def forward(self, batch):
        param = next(self.model.parameters())
        if self.freeze_text_encoder:
            assert param.requires_grad == False

        if self.device is None:
            self.device = param.device

        return self.encode_text(batch)

    def encode_text(self, prompt):
        device = self.model.device
        n_gen = len(prompt)
        if n_gen == 1 and self.input_caption is False:
            raise ValueError("n_gen==1 but input_caption is False")
        batch_size = len(prompt[0])
        
        encoder_list = []
        attention_list = []
        for i in range(n_gen):
            batch = self.tokenizer(prompt[i], max_length=self.return_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

            if self.freeze_text_encoder:
                with torch.no_grad():
                    encoder_hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            else:
                encoder_hidden_states = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
            encoder_list.append(encoder_hidden_states)
            attention_list.append(attention_mask)

        if self.input_caption:
            encoder_return = torch.cat(encoder_list, dim=0)
            attention_return = torch.cat(attention_list, dim=0)
        else:
            encoder_return = torch.cat(encoder_list, dim=1)
            if self.all_pos:
                encoder_return = encoder_return + self.position_embedding.unsqueeze(0).expand(encoder_return.shape[0], -1, -1).to(self.device)
            attention_return = torch.cat(attention_list, dim=1)
        return [encoder_return.detach(), attention_return.float()]
