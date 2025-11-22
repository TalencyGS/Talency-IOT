from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import time

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL")
HF_API_URL = os.getenv("HF_API_URL")

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY não configurada")
if not HF_MODEL:
    raise RuntimeError("HF_MODEL não configurado")
if not HF_API_URL:
    raise RuntimeError("HF_API_URL não configurada")

app = FastAPI(title="Talency IA Advisor")

class ChatRequest(BaseModel):
    mensagem: str

class ChatResponse(BaseModel):
    resposta: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model": HF_MODEL}

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

    print(f"\nChamando {HF_API_URL} com model={HF_MODEL}")
    try:
        r = requests.post(HF_API_URL, headers=headers, json=body, timeout=60)
        print("HF STATUS:", r.status_code)
        if r.status_code != 200:
            # mensagens frequentes que ajudam a diagnosticar
            if r.status_code == 403 and "requires authorization" in r.text.lower():
                raise HTTPException(
                    status_code=502,
                    detail="Modelo requer autorização na Hugging Face. Aceite os termos do modelo na sua conta."
                )
            if r.status_code in (404, 400):
                raise HTTPException(
                    status_code=502,
                    detail=f"Verifique o repo-id do modelo. Valor atual: {HF_MODEL}. Corpo: {r.text}"
                )
            if r.status_code in (429, 503):
                raise HTTPException(
                    status_code=503,
                    detail=f"Ratelimit ou modelo inicializando. Tente novamente. Corpo: {r.text}"
                )
            raise HTTPException(status_code=502, detail=f"Erro HF {r.status_code}: {r.text}")

        data = r.json()
        choices = data.get("choices")
        if not choices:
            raise HTTPException(status_code=502, detail=f"Resposta sem choices. JSON: {data}")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise HTTPException(status_code=502, detail=f"Resposta sem message.content. JSON: {data}")
        return content

    except requests.exceptions.RequestException as e:
        detail = f"Erro HTTP ao chamar HF: {e}"
        if getattr(e, "response", None) is not None:
            detail += f" | Corpo: {e.response.text}"
        raise HTTPException(status_code=502, detail=detail)

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
