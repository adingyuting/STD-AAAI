from typing import Iterator, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models import Model
from torch.nn.parameter import Parameter
from typing import Any, Dict, Optional, Tuple, Union
from swift import Swift, LoRAConfig
from modelscope import AutoTokenizer


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        raise NotImplementedError('error')
    
    def getembedding(self,x):
        raise NotImplementedError('error')
    
    def gettokenizer(self):
        raise NotImplementedError('error')
    
    def getmonthembedding(self):
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        inputs = self.tokenizer.convert_tokens_to_ids(months)
        #self.tokenizer('January,February,March,April,May,June,July,August,September,October,November,December', 
                  #  return_tensors="pt", return_attention_mask=False)
        #month_ids= inputs['input_ids'].to(self.llm_embd.weight.device).view(-1,1)[::2]
        month_ids = torch.tensor(inputs, device=self.llm_embd.weight.device).view(-1,1)
        month_embedding = self.getembedding(month_ids).view(-1,self.emb_dim)
        return month_embedding
    
    def getweekembedding(self):
        weeks = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
        inputs = self.tokenizer.convert_tokens_to_ids(weeks)
        week_ids = torch.tensor(inputs, device=self.llm_embd.weight.device).view(-1,1)
        week_embedding = self.getembedding(week_ids).view(-1,self.emb_dim)
        return week_embedding

class Phi2(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 2560

        llm = Model.from_pretrained('AI-ModelScope/phi-2', trust_remote_code=True)
        llm = llm.model  # PhiForCausalLM stores the base model under "model"

        if layers is not None:
            llm.layers = llm.layers[:layers]

        for pblock in llm.layers:
            # Locate the attention module. Some implementations expose it as
            # `self_attn` (e.g. Phi-2), while others nest it under
            # `pblock.mixer.inner_attn`.
            attn = getattr(pblock, "self_attn", None)
            if attn is None:
                mixer = getattr(pblock, "mixer", None)
                if mixer is not None:
                    attn = getattr(mixer, "inner_attn", mixer)

            if attn is not None:
                if hasattr(attn, "causal"):
                    attn.causal = causal
                elif hasattr(attn, "is_causal"):
                    attn.is_causal = causal

        for name, param in llm.named_parameters():
            param.requires_grad_(False)

        if lora:
            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['Wqkv'],
                    lora_alpha=32,
                    lora_dropout=0.)

            llm = Swift.prepare_model(llm, lora_config, trust_remote_code=True).model

        self.llm_embd = llm.embed_tokens  # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        self.llm_h = llm.layers  # ModuleList (B,len,emb_dim) ->  (B,len,emb_dim)
        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm_h.named_parameters()):
                if 'ln' in name: # or 'mlp' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/phi-2", trust_remote_code=True)

    def forward(self, x: torch.FloatTensor, attention_mask: Optional[torch.Tensor] = None):

        hidden_state = x
        batch_size, seq_len, _ = hidden_state.size()
        position_ids = torch.arange(seq_len, device=hidden_state.device).unsqueeze(0).expand(batch_size, -1)

        for layer in self.llm_h:
            attn = getattr(layer, 'self_attn', None)
            if attn is None:
                mixer = getattr(layer, 'mixer', None)
                if mixer is not None:
                    attn = getattr(mixer, 'inner_attn', mixer)
            if attn is None or not hasattr(attn, 'rotary_emb'):
                raise RuntimeError('Attention block lacks rotary_emb')
            pos_emb = attn.rotary_emb(hidden_state, position_ids)
            if pos_emb is None:
                raise RuntimeError('rotary_emb returned None')
            hidden_state = layer(hidden_state, attention_mask=attention_mask, position_embeddings=pos_emb)

        out = hidden_state

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm_embd(x)
    
    def gettokenizer(self):

        return self.tokenizer 
    
    def getmonthembedding(self):
        inputs = self.tokenizer('January,February,March,April,May,June,July,August,September,October,November,December', 
                    return_tensors="pt", return_attention_mask=False)
        month_ids= inputs['input_ids'].to(self.llm_embd.weight.device).view(-1,1)[::2]
        month_embedding = self.getembedding(month_ids).view(-1,self.emb_dim)
        return month_embedding


class GPT2(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 768

        self.llm = Model.from_pretrained('AI-ModelScope/gpt2',trust_remote_code=True)

        if not layers is None:

            self.llm.transformer.h = self.llm.transformer.h[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q_attn','c_attn'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        # self.llm_embd = llm.transformer.wte # wte:50257,1->51200,768  (B,len,1) -> (B,len,emb_dim) # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'ln' in name  or 'wpe' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/gpt2", trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.transformer.wte(x)
    
    def gettokenizer(self):

        return self.tokenizer 


class Transformer(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()


        self.emb_dim = 768

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=12)
        self.llm = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=3)


    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(x)

        return out

   

class LLAMA3(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 4096

        self.llm = Model.from_pretrained('LLM-Research/Meta-Llama-3-8B-Instruct',trust_remote_code=True)

        print(self.llm)

        if not layers is None:

            self.llm.model.layers = self.llm.model.layers[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoRAConfig(
                    r=16,
                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        # self.llm_embd = llm.transformer.wte # wte:50257,1->51200,768  (B,len,1) -> (B,len,emb_dim) # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'norm' in name  or 'wpe' in name:
                    param.requires_grad = True

        self.tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct", trust_remote_code=True)

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.model.embed_tokens(x)
    
    def gettokenizer(self):

        return self.tokenizer 