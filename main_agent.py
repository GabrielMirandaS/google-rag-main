# ============================================================================
# Imports e Dependências
# ============================================================================
import os
import asyncio
from pathlib import Path
from urllib.parse import quote

# Google Cloud
from google.oauth2 import service_account
from google import genai
from google.genai import types as genai_types

# Agent Platform SDK (Vertex AI Search RAG)
import agentplatform
from agentplatform import types

# Google ADK (Agent Development Kit)
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types_sdk

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# Configurações — ajuste conforme seu ambiente GCP
# ============================================================================
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "pgm-datalake-prod")
LOCATION = os.getenv("GCP_LOCATION", "us-west1")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "pgm-ai-rag-dev")
AGENT_PLATFORM_CORPUS_NAME = os.getenv(
    "AGENT_PLATFORM_CORPUS_NAME", "test-agent"
)
SERVICE_ACCOUNT_PATH = os.getenv(
    "GCP_SERVICE_ACCOUNT_PATH", str(Path(__file__).parent / "api-rag-gemini-dev.json")
)
MODEL_NAME = os.getenv("GENAI_MODEL_NAME", "gemini-flash-latest")

RETRIEVAL_TOP_K = 25
VECTOR_DISTANCE_THRESHOLD = 0.5


# ============================================================================
# Inicialização dos clientes GCP
# ============================================================================
def load_credentials() -> service_account.Credentials:
    """Carrega as credenciais da service account GCP."""
    if not Path(SERVICE_ACCOUNT_PATH).exists():
        raise FileNotFoundError(
            f"Arquivo de credenciais não encontrado: '{SERVICE_ACCOUNT_PATH}'. "
            "Defina a variável de ambiente GCP_SERVICE_ACCOUNT_PATH ou coloque o "
            "JSON no diretório do projeto."
        )
    return service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


credentials = load_credentials()

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

agent_client = agentplatform.Client(
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

print("Clientes Google Cloud inicializados com sucesso.")


# ============================================================================
# Helpers de RAG — corpus, retrieval e prompt
# ============================================================================
def resolve_corpus(display_name: str) -> object | None:
    """Retorna o objeto corpus existente ou None."""
    corpora = agent_client.rag.list_corpora()
    for corpus in corpora.rag_corpora:
        if corpus.display_name == display_name:
            return corpus
    return None


def ensure_corpus(display_name: str) -> object:
    """Retorna o corpus existente ou cria um novo automaticamente."""
    existing = resolve_corpus(display_name)
    if existing is not None:
        return existing
    print(f"[RAG] Corpus '{display_name}' não encontrado. Criando novo corpus…")
    return agent_client.rag.create_corpus(
        rag_corpus=types.RagCorpus(display_name=display_name)
    )


def retrieve_contexts_from_agente(
    query: str, corpus_name: str | None = None
) -> tuple[list[dict], str]:
    """
    Executa retrieval vetorial no corpus e devolve (fontes, status).

    Retorna:
        fontes: lista de dicts com 'titulo', 'texto_utilizado', 'score'
        status: 'success', 'no_corpus', 'no_results', ou 'error'
    """
    target_corpus = corpus_name or AGENT_PLATFORM_CORPUS_NAME

    corpus_obj = resolve_corpus(target_corpus)
    if corpus_obj is None:
        return [], "no_corpus"

    rag_retrieval_config = genai_types.RagRetrievalConfig(
        top_k=RETRIEVAL_TOP_K,
        filter=genai_types.RagRetrievalConfigFilter(
            vector_distance_threshold=VECTOR_DISTANCE_THRESHOLD,
        ),
    )

    try:
        response = agent_client.rag.retrieve_contexts(
            vertex_rag_store=genai_types.VertexRagStore(
                rag_resources=[
                    genai_types.VertexRagStoreRagResource(rag_corpus=corpus_obj.name)
                ],
            ),
            query=types.RagQuery(
                text=query,
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    except Exception as exc:
        print(f"[RAG][ERROR] Falha ao consultar Vertex AI Search: {exc}")
        return [], "error"

    contexts = getattr(response, "contexts", None)
    context_list = []
    if contexts is not None:
        nested = getattr(contexts, "contexts", None)
        if nested is not None:
            context_list = list(nested)
        elif hasattr(contexts, "__iter__") and not isinstance(contexts, (str, bytes)):
            context_list = list(contexts)

    fontes = []
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
            "score": getattr(ctx, "score", None),
        })

    if not fontes:
        return [], "no_results"

    return fontes, "success"


def build_context_block(fontes: list[dict]) -> str:
    """Monta o bloco de contexto formatado para o prompt."""
    blocos = []
    for idx, f in enumerate(fontes, start=1):
        blocos.append(f"[{idx}] ({f['titulo']})\n{f['texto_utilizado']}")
    return "\n\n".join(blocos)


# ============================================================================
# Ferramentas (Tools) do Agente ADK
# ============================================================================

AGENTS_BUILTIN = {
    "clima": "/builtin/feature/Weather",
    "calculator": "/builtin/feature/Calculator",
    "code_interpreter": "/builtin/feature/CodeInterpreter",
    "web_search": "/builtin/feature/WebSearch",
}


def consultar_base_conhecimento(
    pergunta: str,
    agente_state: object | None = None,
) -> dict:
    """
    Tool ADK — consulta a base de conhecimento no Vertex AI Search.

    Usa os built-ins do ADK (clima, calculadora, etc.) quando a pergunta
    corresponde a um agente embutido conhecido. Caso contrário, realiza retrieval
    vetorial no corpus RAG configurado.

    Args:
        pergunta: a pergunta/original do usuário.
        agente_state: estado opcional do ADK (ignorado aqui).

    Returns:
        dict com 'status' ('builtin', 'rag', 'no_corpus', 'no_results', 'error')
        e 'conteudo' (texto já formatado como resposta).
    """
    pergunta_lower = pergunta.strip().lower()

    # --- 1. Verifica se é uma consulta a um agente embutido (builtin) ---
    for label, builtin_id in AGENTS_BUILTIN.items():
        if label in pergunta_lower:
            try:
                from google.adk.artifacts.gcs_artifact import GcsArtifact
                raise SystemExit(builtin_id)
            except SystemExit as se:
                raise  # re-raise o builtin_id para o ADK consumir
            except Exception as exc:
                return {
                    "status": "error",
                    "conteudo": f"Erro ao resolver agente embutido '{label}': {exc}",
                }

    # --- 2. Retrieval vetorial no corpus ---
    try:
        fontes, status = retrieve_contexts_from_agente(pergunta)
    except Exception as exc:
        print(f"[consultar_base_conhecimento][ERROR] {exc}")
        return {
            "status": "error",
            "conteudo": f"Erro ao acessar a base de conhecimento: {exc}",
        }

    if status == "no_corpus":
        return {
            "status": "no_corpus",
            "conteudo": (
                "Nenhuma base de conhecimento configurada encontrada. "
                "Carregue documentos primeiro para que eu possa consultá-los."
            ),
        }

    if status == "no_results":
        return {
            "status": "no_results",
            "conteudo": (
                "Não encontrei informações relevantes na base de conhecimento "
                "para esta pergunta. Tente reformular ou verifique se há "
                "documentos indexados."
            ),
        }

    if status == "error":
        return {
            "status": "error",
            "conteudo": "Ocorreu um erro ao consultar a base de conhecimento. Verifique a conexão com o Vertex AI.",
        }

    # --- 3. Sucesso — retorna contexto formatado + fontes ---
    contexto_texto = build_context_block(fontes)
    referencia_fontes = "\n".join(
        f"- {f['titulo']}" for f in fontes
    )
    return {
        "status": "rag",
        "conteudo": (
            f"As informações foram recuperadas da base de conhecimento.\n\n"
            f"{contexto_texto}\n"
            f"\n--- Fontes consultadas ---\n{referencia_fontes}\n\n"
            f"Pergunta original: {pergunta}"
        ),
        "fontes": fontes,
    }


# --- Tool que responde diretamente ao usuário (resposta ancorada no contexto) ---
def perguntar_ao_assistente(
    pergunta: str,
    agente_state: object | None = None,
) -> dict:
    """
    Tool ADK — faz retrieval + geração de resposta com Gemini em um único passo.

    Ideal para o fluxo final: o usuário pergunta, a tool busca contexto no RAG
    e gera a resposta ancorada nos documentos recuperados.
    """
    try:
        fontes, status = retrieve_contexts_from_agente(pergunta)
    except Exception as exc:
        return {
            "status": "error",
            "conteudo": f"Erro ao acessar a base de conhecimento: {exc}",
        }

    if status in ("no_corpus", "no_results"):
        return {
            "status": status,
            "conteudo": (
                "Não encontrei informações nos documentos. "
                "Carregue documentos antes de fazer perguntas."
                if status == "no_corpus"
                else "Nenhum documento relevante recuperado para esta pergunta."
            ),
        }

    # Monta prompt estrito: o modelo deve responder APENAS com o contexto
    blocos = []
    for idx, f in enumerate(fontes, start=1):
        blocos.append(f"[{idx}] ({f['titulo']})\n{f['texto_utilizado']}")
    contexto = "\n\n".join(blocos)

    prompt = (
        "Você é um assistente que responde APENAS com base no contexto fornecido.\n"
        "Se a resposta não estiver no contexto, diga claramente que não encontrou "
        "a informação nos documentos indexados.\n"
        "Não use conhecimento externo.\n"
        "Responda de forma clara e objetiva em português.\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pergunta: {pergunta}"
    )

    try:
        resposta = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return {
            "status": "answered",
            "conteudo": resposta.candidates[0].content.parts[0].text if resposta.candidates else resposta.text,
            "fontes": fontes,
        }
    except Exception as exc:
        print(f"[perguntar_ao_assistente][ERROR] Falha na geração: {exc}")
        return {
            "status": "error",
            "conteudo": f"Erro ao gerar resposta: {exc}",
        }


# ============================================================================
# Definição do Agente (Google ADK)
# ============================================================================
AGENT_MODEL = MODEL_NAME

agente_principal = Agent(
    name="agente_principal_v1",
    model=AGENT_MODEL,
    description=(
        "Assistente de conhecimento corporativo com integração ao Vertex AI Search RAG. "
        "Consulta base de documentos (corpus) para responder perguntas. "
        "Também pode usar agentes embutidos: clima, calculadora, pesquisa web, "
        "interpretação de código e geração de imagem."
    ),
    instruction=(
        "Você é um assistente de conhecimento corporativo prestativo e preciso.\n"
        "Sua principal fonte de conhecimento é a base de documentos indexada no Vertex AI Search.\n"
        "Sempre utilize a ferramenta 'consultar_base_conhecimento' para recuperar "
        "informações da base antes de responder.\n"
        "Se não houver informação suficiente nos documentos, informe isso ao usuário "
        "de forma clara e educada.\n"
        "Você também pode usar os agentes embutidos listados acima (clima, calculadora, etc.)\n"
        "quando a pergunta do usuário se encaixar nessas categorias.\n"
        "Sempre responda em português (pt-br)."
    ),
    tools=[consultar_base_conhecimento, perguntar_ao_assistente],
)

print(f'Agente criado: "{agente_principal.name}" usando "{AGENT_MODEL}".')


# ============================================================================
# Execução do Agente via ADK Runner
# ============================================================================
APP_NAME = "google_rag_agent"
USER_ID = "user_1"
SESSION_ID = "session_001"


async def run_session(runner, queries: list[str]) -> None:
    """Executa uma lista de perguntas no agente e imprime as respostas."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    print(f"Sessão criada: app='{APP_NAME}', session='{SESSION_ID}'")

    runner = Runner(
        agent=agente_principal,
        app_name=APP_NAME,
        session_service=session_service,
    )
    print(f"Runner criado para '{runner.agent.name}'.")

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f">>> Usuário: {query}")
        content = genai_types_sdk.Content(
            role="user", parts=[genai_types_sdk.Part(text=query)]
        )
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=content
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    answer = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    answer = f"Agente escalou: {event.error_message or 'Sem detalhes.'}"
                else:
                    answer = "(sem resposta final)"
                print(f"<<< Agente: {answer}")
                break


async def main():
    """Exemplo de uso: roda queries de demonstração no agente."""
    await run_session(
        Runner(
            agent=agente_principal,
            app_name=APP_NAME,
            session_service=InMemorySessionService(),
        ),
        [
            "Qual é o conteúdo do nosso corpus de conhecimento?",
            "Consulte a base de conhecimento.",
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())