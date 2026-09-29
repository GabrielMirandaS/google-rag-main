import os
import torch

# --- FIX PARA DIRECTML / PYTORCH SEM DTENSOR ---
import torch.distributed.tensor

if not hasattr(torch.distributed.tensor, "DTensor"):
    class DummyDTensor:
        pass
    torch.distributed.tensor.DTensor = DummyDTensor
# -----------------------------------------------

import torch_directml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 1. Configurar GPU AMD via DirectML
device = torch_directml.device()
print(f">> Usando GPU AMD via DirectML: {device}")

# 2. Carregar Modelo e Tokenizer leve
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f">> Carregando {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, use_safetensors=True
)
model.to(device)

# 3. Carregar o Dataset de Direito Brasileiro (celsowm/legal_br_sft)
print(">> Baixando dataset jurídico (isso pode demorar na primeira vez)...")
# Usando split="train[:2000]" pegamos apenas os 2 mil primeiros para testar.
# Quando quiser treinar no dataset todo (165k exemplos), mude para split="train"
dataset = load_dataset("celsowm/legal_br_sft", split="train[:2000]")

print(">> Dataset baixado! Formatando os dados de chat...")

# Função que converte a coluna 'messages' usando o formato nativo do modelo
def formatar_conversa(exemplo):
    # Pega as mensagens (system, user, assistant) e transforma na string de treino
    texto = tokenizer.apply_chat_template(
        exemplo["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": texto}

# Aplica a função em todo o dataset
dataset = dataset.map(formatar_conversa)

# 4. Configurações de Treino ajustadas para DirectML
sft_config = SFTConfig(
    dataset_text_field="text", 
    max_length=512, # Aumentei um pouco para caber os textos de Direito
    output_dir="./logs_treino",
    num_train_epochs=2, # Reduzi as épocas (repetições) para não demorar muito no teste inicial
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=10, # Vai printar no terminal a cada 10 passos em vez de todo passo
    save_strategy="no",
    use_cpu=False,
    bf16=False,
    fp16=False,
)

# 5. Inicializar o Treinador
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
)

print(f">> Iniciando o fine-tuning na GPU AMD com {len(dataset)} exemplos...")
trainer.train()

# 6. Salvar o modelo em formato Hugging Face
output_dir = "./modelo_safetensors"
print(f">> Salvando modelo em {output_dir}...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(">> Treino concluído com sucesso!")