from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

# Imports do Google Cloud
from google.oauth2 import service_account
from google.cloud import storage
from google import genai
from google.genai import types as genai_types

# Agent Platform para gerenciamento do RAG
import agentplatform
from agentplatform import types
import unicodedata
import re

app = FastAPI(
    title="Gemini RAG API",
    description="API para ingestão de documentos e consultas usando Agent Platform / Vertex AI"
)

# ==========================================
# CONFIGURAÇÕES E CREDENCIAIS
# ==========================================

PROJECT_ID = "pgm-datalake-prod"
LOCATION = "us-west1"  # Mude se seu bucket/vertex estiverem em outra região
GCS_BUCKET_NAME = "pgm-ai-rag-dev"
SERVICE_ACCOUNT_PATH = "./api-rag-gemini-dev.json"

RETRIEVAL_TOP_K = 25
VECTOR_DISTANCE_THRESHOLD = 0.5
MODEL_NAME = "gemini-2.5-flash"

try:
    # 1. Definimos o escopo para acesso geral ao Google Cloud
    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    # 2. Passamos o escopo na hora de carregar o arquivo
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        scopes=SCOPES
    )

    # Init do Cloud Storage
    storage_client = storage.Client(project=PROJECT_ID, credentials=credentials)

    # Init do Agent Platform Client (RAG: corpus, import, retrieve)
    agent_client = agentplatform.Client(
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials
    )

    # Init do Google Gen AI SDK (substitui vertexai.generative_models, deprecado)
    genai_client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials,
    )

except Exception as e:
    print(f"Erro ao carregar credenciais: {e}")


# ==========================================
# MODELOS PYDANTIC (Schemas)
# ==========================================
class QueryRequest(BaseModel):
    query: str
    corpus_name: str


# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def get_corpus(display_name: str):
    """Retorna o corpus existente com o display_name informado, ou None."""
    corpora_list = agent_client.rag.list_corpora()

    for corpus in corpora_list.rag_corpora:
        if corpus.display_name == display_name:
            return corpus

    return None


def get_or_create_corpus(display_name: str):
    """Retorna o corpus existente ou cria um novo (usado no upload)."""
    corpus = get_corpus(display_name)
    if corpus is not None:
        return corpus

    print(f"Criando novo corpus dinamicamente: {display_name}")

    rag_corpus_config = types.RagCorpus(display_name=display_name)
    return agent_client.rag.create_corpus(rag_corpus=rag_corpus_config)


def extract_retrieved_contexts(retrieval_response) -> list[dict]:
    """Normaliza os trechos retornados pelo retrieve_contexts em fontes da API."""
    fontes = []
    contexts = getattr(retrieval_response, "contexts", None)

    # O SDK pode expor a lista em response.contexts ou response.contexts.contexts
    context_list = []
    if contexts is not None:
        nested = getattr(contexts, "contexts", None)
        if nested is not None:
            context_list = list(nested)
        elif hasattr(contexts, "__iter__") and not isinstance(contexts, (str, bytes)):
            context_list = list(contexts)

    for ctx in context_list:
        text = getattr(ctx, "text", None) or ""
        if not text:
            continue

        title = (
            getattr(ctx, "source_display_name", None)
            or getattr(ctx, "source_uri", None)
            or getattr(ctx, "title", None)
            or "documento"
        )
        fontes.append({
            "titulo": title,
            "texto_utilizado": text,
            "score":getattr(ctx, "score", None)
        })

    return fontes


def retrieve_fontes(corpus, query: str) -> list[dict]:
    """Busca trechos relevantes no corpus (retrieval puro, sem geração)."""
    rag_retrieval_config = genai_types.RagRetrievalConfig(
        top_k=RETRIEVAL_TOP_K,
        filter=genai_types.RagRetrievalConfigFilter(
            vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
            # vector_similarity_threshold=0.3
        ),
    )

    retrieval_response = agent_client.rag.retrieve_contexts(
        vertex_rag_store=genai_types.VertexRagStore(
            rag_resources=[
                genai_types.VertexRagStoreRagResource(rag_corpus=corpus.name)
            ],
        ),
        query=types.RagQuery(
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        ),
    )

    return extract_retrieved_contexts(retrieval_response)


def build_grounded_prompt(query: str, fontes: list[dict]) -> str:
    blocos = []
    for i, fonte in enumerate(fontes, start=1):
        blocos.append(f"[{i}] ({fonte['titulo']})\n{fonte['texto_utilizado']}")

    contexto = "\n\n".join(blocos)

    return (
        "Você é um assistente que responde APENAS com base no contexto fornecido abaixo.\n"
        "Se a resposta não estiver no contexto, diga claramente que não encontrou a informação "
        "nos documentos indexados.\n"
        "Não use conhecimento externo.\n"
        "Responda de forma clara e objetiva em português.\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pergunta: {query}"
    )


# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/api/v1/upload")
async def upload_and_index_documents(
    corpus_name: str = Form(..., description="Nome do contexto para agrupar os documentos"),
    # REMOVIDO O 'List[]' para o Swagger gerar um botão direto de upload
    arquivo: UploadFile = File(...)
):
    """
    Recebe um arquivo via botão de upload no Swagger, envia para o GCS e indexa no RAG.
    """
    try:
        corpus = get_or_create_corpus(corpus_name)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)

        # 1. SANITIZAÇÃO BLINDADA (Remove acentos, espaços e caracteres especiais)
        # Transforma "Anotações do Gemini" em "Anotacoes_do_Gemini"
        nome_sem_acento = unicodedata.normalize('NFKD', arquivo.filename).encode('ASCII', 'ignore').decode('utf-8')
        safe_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', nome_sem_acento)

        # 2. Usar o nome sanitizado para salvar no GCS
        destination_blob_name = f"rag_docs/{corpus_name}/{safe_filename}"
        blob = bucket.blob(destination_blob_name)

        # Lê e faz upload
        content = await arquivo.read()
        blob.upload_from_string(content, content_type=arquivo.content_type)

        # 3. A URI agora estará 100% limpa (apenas ASCII)
        gcs_uri = f"gs://{GCS_BUCKET_NAME}/{destination_blob_name}"
        print(f"Arquivo salvo no GCS: {gcs_uri}")

        # 4. Continua com o import_files normal...

        print(f"Iniciando indexação do arquivo no corpus {corpus.name}...")

        # Importando o arquivo único
        response = agent_client.rag.import_files(
            name=corpus.name,
            import_config=types.ImportRagFilesConfig(
                # Aponta a origem do arquivo para o Cloud Storage
                gcs_source=genai_types.GcsSource(uris=[gcs_uri]),

                # Configura como o arquivo será "picotado" (Chunking)
                rag_file_transformation_config=types.RagFileTransformationConfig(
                    rag_file_chunking_config=types.RagFileChunkingConfig(
                        chunk_size=1600,
                        chunk_overlap=300,
                    )
                )
            )
        )

        return {
            "status": "success",
            "corpus_id": corpus.name,
            "arquivo_enviado": arquivo.filename,
            "arquivos_importados": int(response.imported_rag_files_count)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieve")
async def retrieve_only(request: QueryRequest):
    """
    Apenas retrieval: devolve os trechos relevantes do corpus, sem gerar resposta com Gemini.
    """
    try:
        corpus = get_corpus(request.corpus_name)
        if corpus is None:
            raise HTTPException(
                status_code=404,
                detail=f"Corpus '{request.corpus_name}' não encontrado. Faça o upload de documentos primeiro."
            )

        fontes = retrieve_fontes(corpus, request.query)

        return {
            "query": request.query,
            "corpus_name": request.corpus_name,
            "quantidade": len(fontes),
            "fontes": fontes
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrieve_resposta")
async def retrieve_and_generate(request: QueryRequest):
    """
    Retrieval + resposta: busca trechos relevantes e gera a resposta com Gemini.
    """
    try:
        corpus = get_corpus(request.corpus_name)
        if corpus is None:
            raise HTTPException(
                status_code=404,
                detail=f"Corpus '{request.corpus_name}' não encontrado. Faça o upload de documentos primeiro."
            )

        fontes = retrieve_fontes(corpus, request.query)

        if not fontes:
            return {
                "query": request.query,
                "resposta": (
                    "Não encontrei informações nos documentos indexados "
                    "para responder essa pergunta."
                ),
                "fontes": [],
            }

        prompt = build_grounded_prompt(request.query, fontes)
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        return {
            "query": request.query,
            "resposta": response.text,
            "fontes": fontes,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
