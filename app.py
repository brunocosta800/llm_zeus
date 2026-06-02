import os
import time
import uuid
import concurrent.futures # IMPORTANTE: Adicione esta linha
from flask import Flask, request, jsonify
from dotenv import load_dotenv

import markdown
from weasyprint import HTML

from supabase.client import Client, create_client
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from doc_parse import Parse

load_dotenv()

app = Flask(__name__)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("As variáveis SUPABASE_URL e SUPABASE_SERVICE_KEY precisam estar no .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
llm_redator = ChatOpenAI(model="gpt-5.4", temperature=0.1)

def buscar_documentos_supabase(pergunta, cnpj_empresa):
    try:
        vetor_pergunta = embeddings.embed_query(pergunta)
        resposta = supabase.rpc("match_documents", {
            "query_embedding": vetor_pergunta,
            "match_count": 50,
            "filter": {"cnpj": cnpj_empresa} 
        }).execute()
        
        if not resposta.data:
            return "Informação não disponível nos documentos analisados."

        docs_formatados = []
        for doc in resposta.data:
            conteudo = doc.get('content', '')
            metadata = doc.get('metadata', {})
            pagina = metadata.get('pagina', 'N/A')
            fonte = metadata.get('fonte', 'Desconhecida')
            docs_formatados.append(f"Trecho (Fonte: {fonte}, Pág {pagina}): {conteudo}")
            
        return "\n\n".join(docs_formatados)
    except Exception as e:
        print(f"Erro na busca vetorial: {e}")
        return "Erro ao recuperar contexto do banco de dados."

template_auditoria = """
Você é um Auditor Sénior especialista em auditoria de ciclo de receitas e conformidade fiscal.
Sua missão é realizar uma análise técnica, identificar discrepâncias e redigir questionamentos fundamentados nos documentos fornecidos.

DADOS DA EMPRESA:
- CNPJ: {cnpj}
- Regime Tributário: {regime_tributario}
- Ano Fiscal: {ano_fiscal}

MATRIZ DE RISCO E GATILHOS (CRITÉRIOS PARA QUESTIONAMENTO):
1. DISCREPÂNCIA FISCAL-CONTÁBIL: Compare o "Valor Total Anual" do Livro Fiscal com o "Saldo Total Anual" do Balancete/Razão do MESMO ANO. Se não baterem, o gatilho é acionado.
2. VARIAÇÃO ANUAL: Aumento ou redução de receita superior a 2% em relação ao ano anterior.
3. DISTORÇÃO MENSAL: Receita de um mês específico divergindo mais de 2% da média mensal.
4. CONFORMIDADE TRIBUTÁRIA: Multiplique o "Valor Total Contábil" pela alíquota padrão. Compare com o "Imposto Retido/Informado". Diferença > 2% aciona o gatilho.

REGRAS CRÍTICAS DE FORMATAÇÃO E SÍNTESE (MUITO IMPORTANTE):
- NOME DAS FONTES: Nunca imprima hashes ou números longos de arquivos. Resuma o nome (Ex: em vez de "177956...-razao_24_25.pdf", escreva apenas "Razão 2025").
- SÍNTESE DE DADOS: Se vários meses tiverem o mesmo valor, AGRUPE-OS em uma única linha (Ex: "Jan a Out: R$ 125.000,00 mensais"). Jamais liste 12 meses idênticos um embaixo do outro.
- USO DE TABELAS: Sempre que houver mais de 3 valores para apresentar (como meses ou comparativos de impostos), use uma tabela Markdown limpa.
- CÁLCULOS LIMPOS: Proibido o uso de LaTeX ou notações como `\frac`. Use textos simples em uma linha só (Ex: Variação = (100 - 50) / 50 * 100 = 100%).

DOCUMENTOS FORNECIDOS (Contexto RAG):
{contexto_rag}

TÓPICO DE AUDITORIA A SER EXECUTADO:
{pergunta}

ESTRUTURA OBRIGATÓRIA DA RESPOSTA:
**1. Extração de Dados**
(Use uma tabela Markdown se houver muitos dados, ou tópicos muito curtos. Cite a fonte de forma resumida ao lado).

**2. Constatação**
(Máximo de 2 parágrafos diretos e objetivos).

**3. Análise Técnica**
(Apresente a lógica matemática de forma limpa, direta e em linha única. Destaque o resultado final em negrito).

**4. Questionamento à Administração**
> (Use o sinal de maior '>' para criar a citação). Seja cirúrgico e breve na pergunta.
"""

prompt_auditoria = PromptTemplate.from_template(template_auditoria)

chain_auditoria = (
    {
        "contexto_rag": lambda x: buscar_documentos_supabase(x["pergunta"], x["cnpj"]), 
        "pergunta": lambda x: x["pergunta"],
        "cnpj": lambda x: x["cnpj"],
        "regime_tributario": lambda x: x["regime_tributario"],
        "ano_fiscal": lambda x: x["ano_fiscal"]
    }
    | prompt_auditoria 
    | llm_redator 
    | StrOutputParser()
)

template_conclusao = """
Você é um Auditor Sénior. A sua tarefa é ler as constatações de um relatório recém-gerado e emitir um Parecer Final.

RELATÓRIO DE AUDITORIA GERADO (Análises 1 a 4):
{relatorio_gerado}

REGRAS PARA A CONCLUSÃO:
1. Resumo de Riscos: Sintetize em bullet points os principais problemas encontrados.
2. Classificação: Atribua uma (e apenas uma) cor/status: Sem ressalvas, Com ressalvas, Adverso ou Abstenção.

ESTRUTURA OBRIGATÓRIA DA RESPOSTA (Use Markdown):
* **Classificação de Risco:** **[SUA CLASSIFICAÇÃO]**
* **Justificativa:** [Sua justificativa sintética de 2 a 3 linhas]

**Resumo Executivo de Riscos:**
* [Ponto de risco 1]
* [Ponto de risco 2]
"""
prompt_conclusao = PromptTemplate.from_template(template_conclusao)
chain_conclusao = prompt_conclusao | llm_redator | StrOutputParser()

estrutura_relatorio = [
    "Análise 1: Discrepância Fiscal-Contábil (Comparação entre Saldo de Saídas do Livro Fiscal e Receitas no Razão/Balancete)",
    "Análise 2: Variação Anual de Receita (Verificação de flutuações superiores a 2% entre exercícios)",
    "Análise 3: Distorção Mensal (Verificação de meses com pico ou queda maior que 2% da média do ano)",
    "Análise 4: Conformidade Tributária (Cruzamento da alíquota efetiva retida/paga com o regime tributário informado)"
]


@app.route('/api/ingerir-documentos', methods=['POST'])
def api_ingerir_documentos():
    try:
        dados = request.get_json()

        cnpj = dados.get('cnpj')
        arquivos = dados.get('arquivos')
        doc_type = dados.get('doc_type', 'documento_contabil')
        ano_fiscal = dados.get('ano_fiscal', '2025')

        if not all([cnpj, arquivos]):
            return jsonify({"erro": "Campos obrigatórios: cnpj, arquivos"}), 400

        textos, metadados, ids_chroma, documentos = Parse.parsionar_documento(
            lista_documentos=arquivos,
            cnpj=cnpj
        )

        if not textos:
            return jsonify({"erro": "Nenhum texto pôde ser extraído."}), 422

        tamanho_lote = 50
        total_inserido = 0

        for i in range(0, len(textos), tamanho_lote):
            lote_textos = textos[i:i + tamanho_lote]
            lote_metadados = metadados[i:i + tamanho_lote]

            textos_puros = [
                item.page_content if hasattr(item, 'page_content') else str(item)
                for item in lote_textos
            ]

            vetores = embeddings.embed_documents(textos_puros)
            registros_db = []

            for txt_puro, meta, vetor in zip(textos_puros, lote_metadados, vetores):
                meta_enriquecido = meta.copy() if meta else {}
                meta_enriquecido['doc_type'] = doc_type
                meta_enriquecido['ano_fiscal'] = ano_fiscal

                registros_db.append({
                    "id": str(uuid.uuid4()),
                    "content": txt_puro,    
                    "metadata": meta_enriquecido,
                    "embedding": vetor
                })

            supabase.table("documents").insert(registros_db).execute()
            total_inserido += len(registros_db)

        return jsonify({
            "status": "sucesso",
            "mensagem": f"Foram inseridos {total_inserido} trechos financeiros no banco de dados."
        }), 200

    except Exception as e:
        print(f"Erro na ingestão: {e}")
        return jsonify({"erro": "Falha na ingestão", "detalhes": str(e)}), 500

@app.route('/api/gerar-relatorio', methods=['POST'])
def api_gerar_relatorio():
    try:
        dados = request.get_json()
        
        cnpj = dados.get('cnpj')
        regime_tributario = dados.get('regime_tributario')
        ano_fiscal = '2025'
        
        if not all([cnpj, regime_tributario]):
            return jsonify({"erro": "Campos obrigatórios: nome, cnpj, regime_tributario"}), 400
        
        print(f"\n[API] Iniciando Auditoria Paralela para: {cnpj}")
        
        relatorio_completo = f"# Relatório de Auditoria de Ciclo de Receitas e Conformidade Fiscal\n## CNPJ:** {cnpj} | **Regime Tributário:** {regime_tributario}\n\n"

        resultados_topicos = [""] * len(estrutura_relatorio)
        corpo_analises = ""
        
        def gerar_topico(indice, topico):
            print(f"Processando: {topico[:20]}...")
            comando_geracao = (
                f"Execute a auditoria correspondente ao tópico: '{topico}'. "
                "Recupere os dados financeiros, efetue os cálculos comparativos (como variações percentuais) e verifique se o gatilho da matriz de risco foi atingido."
            )
            
            inputs_chain = {
                "pergunta": comando_geracao,
                "ano_fiscal": ano_fiscal,
                "cnpj": cnpj,
                "regime_tributario": regime_tributario
            }
            
            return indice, chain_auditoria.invoke(inputs_chain)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futuros = [executor.submit(gerar_topico, i, topico) for i, topico in enumerate(estrutura_relatorio)]
            
            for futuro in concurrent.futures.as_completed(futuros):
                indice, texto_secao = futuro.result()
                resultados_topicos[indice] = f"### {estrutura_relatorio[indice]}\n\n{texto_secao}\n\n---\n\n"
                corpo_analises += f"Tópico {indice+1}:\n{texto_secao}\n\n"
                
        relatorio_completo += "".join(resultados_topicos)

        print("Processando: Conclusão Final...")
        texto_conclusao = chain_conclusao.invoke({"relatorio_gerado": corpo_analises})
        relatorio_completo += f"### Parecer Final e Conclusão\n\n{texto_conclusao}\n\n---\n\n"

        cid_hash = str(uuid.uuid4()) 
        print("Convertendo Markdown para PDF de Auditoria...")
        
        # IMPORTANTE: Adicione a extensão 'nl2br' e 'sane_lists' para o markdown não quebrar as listas e as fórmulas
        html_content = markdown.markdown(
            relatorio_completo, 
            extensions=['tables', 'nl2br', 'sane_lists']
        )
        
        html_com_estilo = f"""
        <html>
            <head>
                <style>
                    @page {{ margin: 2.5cm; }}
                    body {{ 
                        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; 
                        line-height: 1.8; /* Aumentado para dar mais respiro à leitura */
                        color: #333; 
                        font-size: 11pt;
                    }}
                    h1 {{ 
                        color: #1a252f; 
                        border-bottom: 2px solid #34495e; 
                        padding-bottom: 10px; 
                        text-align: center;
                        font-size: 16pt;
                        margin-bottom: 40px;
                    }}
                    h2 {{ color: #2c3e50; font-size: 14pt; margin-top: 30px; margin-bottom: 15px; }}
                    h3 {{ 
                        color: #2980b9; 
                        background-color: #f7f9fa; 
                        padding: 12px 15px; 
                        border-left: 5px solid #2980b9; 
                        font-size: 12pt;
                        margin-top: 35px;
                        margin-bottom: 20px;
                    }}
                    p {{ text-align: justify; margin-bottom: 20px; }}
                    
                    /* Estilização limpa para as Listas */
                    ul, ol {{ margin-top: 10px; margin-bottom: 25px; padding-left: 20px; }}
                    li {{ margin-bottom: 12px; padding-left: 5px; }}
                    
                    /* Estilização para as Tabelas que o LLM vai gerar */
                    table {{ 
                        width: 100%; 
                        border-collapse: collapse; 
                        margin: 25px 0; 
                        font-size: 10pt;
                    }}
                    th {{ background-color: #ecf0f1; border-bottom: 2px solid #bdc3c7; text-align: left; padding: 10px; }}
                    td {{ border-bottom: 1px solid #ecf0f1; padding: 10px; }}
                    
                    blockquote {{ 
                        border-left: 4px solid #e74c3c; 
                        background-color: #fcf3f2; 
                        margin: 25px 0; 
                        padding: 15px 20px; 
                        font-style: italic; 
                        color: #c0392b;
                    }}
                    
                    hr {{ border: 0; border-top: 1px solid #bdc3c7; margin: 40px 0; }}
                    .footer {{ 
                        font-size: 8pt; 
                        color: #7f8c8d; 
                        text-align: center; 
                        margin-top: 50px; 
                        border-top: 1px solid #ecf0f1; 
                        padding-top: 15px; 
                    }}
                </style>
            </head>
            <body>
                {html_content}
                <div class="footer">
                    Documento gerado eletronicamente por Agente de Auditoria AI.<br>
                    Hash de Autenticidade: {cid_hash}
                </div>
            </body>
        </html>
        """
        
        pdf_bytes = HTML(string=html_com_estilo).write_pdf()
        nome_arquivo_pdf = f"auditoria_{cid_hash}.pdf"
        
        supabase.storage.from_("relatorios").upload(
            path=nome_arquivo_pdf,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf"}
        )
        
        pdf_url = supabase.storage.from_("relatorios").get_public_url(nome_arquivo_pdf)

        dados_db = {
            "cid_hash": cid_hash,
            "cnpj": cnpj,
            "relatorio_markdown": relatorio_completo
        }
        supabase.table("reports").insert(dados_db).execute()

        return jsonify({
            "status": "sucesso",
            "cid_hash": cid_hash,
            "url_pdf": pdf_url, 
            "relatorio_markdown": relatorio_completo
        }), 200

    except Exception as e:
        print(f"Erro na geração/upload: {e}")
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)