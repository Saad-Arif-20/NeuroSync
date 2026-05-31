import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class MultimodalLLMAssistant(nn.Module):
    def __init__(self, llm_model_name="microsoft/phi-2", embedding_dim=512):
        super().__init__()
        
        # 1. Load the base LLM (We use Phi-2 as a placeholder for a lightweight LLM)
        # In a real training scenario on a T4 GPU, you would use quantization (load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(llm_model_name)
        config.pad_token_id = self.tokenizer.eos_token_id
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
            low_cpu_mem_usage=True
        )
        
        # 2. Configure LoRA (Low-Rank Adaptation)
        # We freeze the base model and only train tiny adapter layers
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()
        
        # 3. Projection Layer
        # The LLM expects embeddings of a certain dimension (e.g., 2048 or 4096)
        # We project our 512-dim shared embedding from Phase 1 into the LLM's input dimension
        llm_hidden_size = self.llm.config.hidden_size
        self.visual_projection = nn.Linear(embedding_dim, llm_hidden_size)
        torch.nn.init.normal_(self.visual_projection.weight, std=0.01)
        torch.nn.init.zeros_(self.visual_projection.bias)
        
    def forward(self, input_embeddings, input_ids=None, attention_mask=None, labels=None):
        """
        input_embeddings: The embeddings coming from our Phase 1 model (Image or EEG)
        input_ids: Text prompts (e.g., "Describe this image: ")
        """
        # Ensure precision matches the LLM to prevent Float/Half crashes
        input_embeddings = input_embeddings.to(dtype=self.visual_projection.weight.dtype)
        
        # Project our custom embedding into the LLM's dimension
        projected_embeddings = self.visual_projection(input_embeddings)
        
        # Add a sequence length dimension if it's missing (batch_size, 1, hidden_size)
        if len(projected_embeddings.shape) == 2:
            projected_embeddings = projected_embeddings.unsqueeze(1)
            
        # Get the embeddings for the text prompt
        if input_ids is not None:
            text_embeddings = self.llm.get_input_embeddings()(input_ids)
            
            # Concatenate the projected visual embedding WITH the text embeddings
            # (batch_size, 1 + text_seq_len, hidden_size)
            inputs_embeds = torch.cat([projected_embeddings, text_embeddings], dim=1)
            
            # We also need to extend the attention mask
            extended_attention_mask = torch.cat(
                [torch.ones((attention_mask.shape[0], 1), device=attention_mask.device), attention_mask], 
                dim=1
            )
            
            if labels is not None:
                # We don't want to calculate loss on the visual embedding token, so we pad it with -100
                extended_labels = torch.cat(
                    [torch.full((labels.shape[0], 1), -100, device=labels.device), labels],
                    dim=1
                )
            else:
                extended_labels = None
                
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )
        else:
            # Generate mode (no text prompt, just the image/EEG embedding)
            outputs = self.llm(inputs_embeds=projected_embeddings)
            
        return outputs

    def generate(self, input_embeddings, prompt_text="This neural signal pattern corresponds to a", max_new_tokens=5):
        """
        Inference function to generate text from an image or EEG embedding.
        """
        device = input_embeddings.device
        input_embeddings = input_embeddings.to(dtype=self.visual_projection.weight.dtype)
        projected_embeddings = self.visual_projection(input_embeddings).unsqueeze(1)
        
        tokens = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        text_embeddings = self.llm.get_input_embeddings()(tokens.input_ids)
        
        inputs_embeds = torch.cat([projected_embeddings, text_embeddings], dim=1)
        
        # FIX: Ensure everything is float16 to prevent mixed dtype crashes
        inputs_embeds = inputs_embeds.to(dtype=torch.float16)
        
        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return prompt_text + " " + self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
