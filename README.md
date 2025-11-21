# Talency IA Advisor

Serviço de IA generativa baseado em Deep Learning para apoiar a plataforma **Talency** com sugestões de estudo, resumos de trilhas e mensagens de motivação.

---

## 1. Configuração

### Criar ambiente virtual
```bash
python -m venv .venv
```

### Ativar
Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### Instalar dependências
```bash
pip install -r requirements.txt
```

### Criar arquivo `.env`
```env
HF_API_KEY=SEU_TOKEN
HF_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
HF_API_URL=https://router.huggingface.co/v1/chat/completions
```

---

## 2. Execução

```bash
uvicorn main:app --reload
```

Acesse a documentação em:
```
http://127.0.0.1:8000/docs
```

---

## 3. Endpoint Principal

### POST `/chat`

#### Request:
```json
{
  "mensagem": "TIPO: SUGESTAO_ESTUDO..."
}
```

#### Response:
```json
{
  "resposta": "Texto gerado pela IA"
}
```

Tipos suportados:
- `TIPO: SUGESTAO_ESTUDO`
- `TIPO: RESUMO_CONTEUDO`
- `TIPO: MOTIVACAO`

---

## 4. Integração

- Finalização de teste → Backend .NET → IA → sugestão personalizada  
- Resumo de conteúdo → Web/Mobile → IA → resumo rápido  
- Dashboard → IA → mensagem motivacional usando dados de IoB  

---

## 5. IoB

O app captura eventos de estudo e envia ao backend.  
Esses dados são usados para enriquecer prompts enviados à IA, personalizando recomendações e motivação.

---

## 6. Tecnologias

- Python  
- FastAPI  
- Hugging Face LLM  
- Uvicorn  
- Pydantic  
- python-dotenv

---

## 7. Próximos passos

- Criar endpoints independentes para cada função  
- Adicionar logs em banco NoSQL  
- Ajustar prompts para cada trilha  
