from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_API_URL = os.getenv("HF_API_URL", "https://router.huggingface.co/v1/chat/completions")

if not HF_API_KEY:
   raise RuntimeError("Variável de ambiente HF_API_KEY não configurada")

app = FastAPI(title="Talency IA Advisor")


class ChatRequest(BaseModel):
   mensagem: str


class ChatResponse(BaseModel):
   resposta: str


@app.get("/health")
def health_check():
   return {"status": "ok"}


def chamar_huggingface(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.7,
    }

    print(f"\n--- TENTANDO CHAMAR MODELO: {HF_MODEL} ---")
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=body, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ ERRO {response.status_code} vindo da Hugging Face:")
            print(f"DETALHE DO ERRO: {response.text}")
        
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices")
        if not choices:
            raise ValueError(f"Resposta sem choices. JSON recebido: {data}")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise ValueError(f"Resposta sem content. JSON recebido: {data}")

        print("✅ Sucesso! Resposta gerada.")
        return content

    except requests.exceptions.RequestException as e:
        erro_detalhado = f"Erro HTTP: {str(e)}"
        
        if e.response is not None:
            erro_detalhado += f" | RESPOSTA DA API: {e.response.text}"
        
        print(f"🚨 EXCEPTION CAPTURADA: {erro_detalhado}")
        raise HTTPException(status_code=500, detail=erro_detalhado)

    except Exception as e:
        print(f"🚨 ERRO INTERNO: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno no Python: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
   system_prompt = (
       "Você é o assistente de IA da plataforma Talency, focada em trilhas "
       "profissionais do futuro. Você recebe mensagens com um campo TIPO que "
       "indica o que deve fazer. "
       "Se TIPO for SUGESTAO_ESTUDO, gere uma sugestão de próximos passos de estudo. "
       "Se TIPO for RESUMO_CONTEUDO, gere um resumo claro e curto do texto. "
       "Se TIPO for MOTIVACAO, gere uma mensagem de motivação personalizada. "
       "Responda sempre em português, de forma objetiva, sem mencionar que é IA."
   )

   resposta = chamar_huggingface(system_prompt, request.mensagem)
   return ChatResponse(resposta=resposta)
