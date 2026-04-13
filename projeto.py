import os
import time
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- IMPORTS DA OPENAI VOLTARAM AQUI ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from doc_parse import Parse # O seu módulo de extração

# ==========================================
# 0. CONFIGURAÇÃO DA CHAVE DE API
# ==========================================
# Insira sua chave real aqui (ou use python-dotenv para carregar de um arquivo .env)
os.environ["OPENAI_API_KEY"] = 

# ==========================================
# 1. SETUP DO BANCO VETORIAL (OPENAI EMBEDDINGS)
# ==========================================
print("A carregar modelo de embeddings da OpenAI...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="rag_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

print("A extrair e processar o PDF...")
textos, metadados, ids_chroma, documentos = Parse.parsionar_documento("relatorio_input_test.pdf")
docs = [Document(page_content=doc.page_content, metadata=meta) for doc, meta in zip(textos, metadados)]

vector_store.add_documents(docs)
print(f"Adicionados {len(docs)} documentos à vector store.")

# ==========================================
# 2. CONFIGURAÇÃO DA CHAIN DO REDATOR (GPT-4.1-nano)
# ==========================================
print("A inicializar o modelo gpt-4.1-nano...")
llm_redator = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)

# Como o GPT tem janela de contexto gigante, podemos voltar para 4 pedaços (mais contexto)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

template = """
Você é um Auditor Sénior especialista em normas ESG e relatórios financeiros IFRS S2.
A sua tarefa é redigir o trecho solicitado baseado ESTRITAMENTE nos documentos fornecidos.

REGRAS CRÍTICAS:
1. ZERO ALUCINAÇÃO: Baseie-se apenas nas informações do "Contexto RAG".
2. CITAÇÕES OBRIGATÓRIAS: Para todo o dado numérico ou afirmação técnica, cite a fonte no formato (Fonte: [arquivo], Pág: [X]).
3. TOM: Corporativo, formal e analítico.
4. DADOS FALTANTES: Se não houver dados no contexto, escreva: "Informação não disponível nos documentos analisados."

DOCUMENTOS FORNECIDOS (Contexto RAG):
{contexto_rag}

TÓPICO A SER ESCRITO:
{pergunta}
"""
prompt_redator = PromptTemplate.from_template(template)

def formatar_documentos(docs):
    return "\n\n".join([f"Trecho (Pág {d.metadata.get('pagina', 'N/A')}): {d.page_content}" for d in docs])

chain_redator = (
    {"contexto_rag": retriever | formatar_documentos, "pergunta": RunnablePassthrough()}
    | prompt_redator 
    | llm_redator 
    | StrOutputParser()
)

# ==========================================
# 3. LOOP DE GERAÇÃO EXTENSA (16 Tópicos)
# ==========================================
estrutura_relatorio = [
    # --- GOVERNANÇA ---
    "Governança Climática: Papel do Conselho de Administração e Comitês de Assessoramento",
    "Governança Climática: Atribuições do Comitê Executivo e Fórum de Baixo Carbono",
    "Governança Climática: Vinculação da Remuneração Executiva a Metas de Sustentabilidade",
    
    # --- ESTRATÉGIA ---
    "Estratégia de Transição Climática: O Papel da Mineração na Economia de Baixo Carbono",
    "Estratégia: Metas de Descarbonização para 2030 e 2050 (Escopos 1 e 2)",
    "Estratégia: Metas de Descarbonização e Abordagem para a Cadeia de Valor (Escopo 3)",
    "Estratégia: Esforços de Mitigação e Uso de Créditos de Carbono de Alta Integridade",
    
    # --- GESTÃO DE RISCOS E OPORTUNIDADES ---
    "Gestão de Riscos: Metodologia de Análise de Cenários Climáticos e Premissas",
    "Análise de Riscos de Transição: Exposição a Regulamentações e Precificação de Carbono",
    "Análise de Riscos de Transição: Impactos e Taxação no Setor de Transporte Marítimo (IMO)",
    "Análise de Riscos Físicos: Vulnerabilidade de Ativos a Eventos Climáticos Extremos",
    "Oportunidades Climáticas: Expansão de Mega Hubs e Briquetes de Minério de Ferro",
    "Oportunidades Climáticas: Circularidade (Waste to Value) e Metais para Transição Energética",
    
    # --- MÉTRICAS E METAS ---
    "Métricas: Desempenho Atual e Metodologia do Inventário de Emissões de Escopos 1 e 2",
    "Métricas: Desempenho Atual e Composição das Categorias de Emissão de Escopo 3",
    "Métricas de Resiliência: Gestão Hídrica e Operações em Áreas de Estresse Hídrico"
]

print("\n🚀 A iniciar a geração do Relatório ESG Oficial via OpenAI...")
print(f"Total de secções a serem escritas: {len(estrutura_relatorio)}\n")

relatorio_completo = "# Relatório de Sustentabilidade e Clima (Padrão IFRS S2)\n\n"
relatorio_completo += "> *Documento gerado de forma autónoma via IA com base nos dados fornecidos.*\n\n"

for i, topico in enumerate(estrutura_relatorio, 1):
    print(f"⏳ [{i}/{len(estrutura_relatorio)}] A escrever a secção: {topico}...")
    
    comando_geracao = (
        f"Escreva a secção oficial do relatório intitulada: '{topico}'. "
        "Concentre-se apenas neste tópico. Desenvolva o texto com pelo menos 3 parágrafos robustos."
    )
    
    texto_secao = chain_redator.invoke(comando_geracao)
    
    relatorio_completo += f"## {topico}\n\n"
    relatorio_completo += texto_secao + "\n\n"
    relatorio_completo += "---\n\n"
    
    # Uma pequena pausa de 2 segundos ajuda a evitar problemas de Rate Limit na API da OpenAI
    time.sleep(2) 

# ==========================================
# 4. EXPORTAÇÃO PARA MARKDOWN
# ==========================================
nome_arquivo = "relatorio_esg_final.md"
with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
    arquivo.write(relatorio_completo)

print(f"\n✅ Relatório concluído com sucesso!")
print(f"📁 Ficheiro guardado em: {nome_arquivo}")