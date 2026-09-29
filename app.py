import streamlit as st
import time
# Importa as estruturas do seu main.py
from main import genai_client, agent_client, QueryRequest, RETRIEVAL_TOP_K, VECTOR_DISTANCE_THRESHOLD

st.set_page_config(page_title="RAG GCP Tester", page_icon="🔍", layout="wide")

# Barra Lateral (Configurações do Projeto)
st.sidebar.title("⚙️ Configurações RAG")
st.sidebar.info("Projeto: google-rag-main")

corpus_input = st.sidebar.text_input("Nome do Corpus / Base Vetorial", value="corpus-documentacao-dev")
st.sidebar.markdown(f"**Top K:** `{RETRIEVAL_TOP_K}`")
st.sidebar.markdown(f"**Threshold:** `{VECTOR_DISTANCE_THRESHOLD}`")

# Área Principal
st.title("🤖 Tester de Interface - RAG Engine")
st.caption("Integração com Agent Platform + Vertex AI (Gemini 2.5 Flash)")

# Formulário de Teste
with st.form("query_form"):
    user_query = st.text_area("Digite a pergunta para testar o sistema:", height=100)
    submitted = st.form_submit_button("🔍 Executar Consulta RAG")

if submitted:
    if not user_query.strip():
        st.warning("Por favor, digite uma pergunta válida.")
    else:
        # 1. Validação do Schema Pydantic
        try:
            req = QueryRequest(query=user_query, corpus_name=corpus_input)
            st.success("✅ Contrato Pydantic (QueryRequest) validado com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro na validação Pydantic: {e}")
            st.stop()

        # 2. Execução da Busca RAG e Chamada ao Modelo
        with st.spinner("Buscando documentos e gerando resposta..."):
            start_time = time.time()
            
            try:
                # Simulação / Chamada dos Clientes GCP
                # -------------------------------------------------------------
                # Aqui você chama a função principal do seu backend. Exemplo:
                # context_docs = agent_client.retrieve(corpus=req.corpus_name, query=req.query)
                # response = genai_client.generate_content(...)
                # -------------------------------------------------------------
                
                # Exemplo de saída para teste visual:
                elapsed_time = round(time.time() - start_time, 2)
                
                st.subheader("💡 Resposta Sintetizada")
                st.write("Esta é a resposta gerada pelo **Gemini 2.5 Flash** com base nos contextos encontrados.")
                
                # Exibição dos Metadados do Teste
                col1, col2 = st.columns(2)
                col1.metric("Tempo de Resposta", f"{elapsed_time} s")
                col2.metric("Status da Requisição", "200 OK")

                # Exibição do Contexto Retornado (Top K)
                with st.expander("📚 Ver Documentos de Contexto Recuperados (Agent Platform)"):
                    st.json({
                        "retrieval_top_k": RETRIEVAL_TOP_K,
                        "vector_threshold": VECTOR_DISTANCE_THRESHOLD,
                        "retrieved_chunks": [
                            {"doc_id": "doc_01", "score": 0.89, "text": "Trecho referente à autenticação GCP..."},
                            {"doc_id": "doc_02", "score": 0.74, "text": "Trecho sobre variáveis de ambiente no .env..."}
                        ]
                    })

            except Exception as e:
                st.error(f"❌ Erro durante a execução da pipeline RAG: {e}")