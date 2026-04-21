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
1. DISCREPÂNCIA FISCAL-CONTÁBIL: Isole a análise ano a ano. Extraia o "Valor Total Anual" do Livro Fiscal e compare com o "Saldo Total Anual" do Balancete/Razão do MESMO ANO. Calcule a diferença (gap). Se não baterem, o gatilho é acionado.
2. VARIAÇÃO ANUAL (RELEVÂNCIA): Se houver aumento ou redução de receita superior a 2% em relação ao ano anterior.
3. DISTORÇÃO MENSAL (LINEARIDADE): Se a receita de um mês específico divergir mais de 2% da média mensal do exercício.
4. CONFORMIDADE TRIBUTÁRIA (Variação de Imposto Calculado): O cálculo deve ser feito comparando totais. Multiplique o "Valor Total Contábil" pela alíquota padrão do {regime_tributario} para encontrar o "Imposto Esperado". Compare esse valor com o "Imposto Retido/Informado" no Livro Fiscal. Se a diferença percentual entre eles (ex: 120.000 vs 95.000) for maior que 2%, acione o gatilho.

REGRAS CRÍTICAS DE EXECUÇÃO:
- ZERO ALUCINAÇÃO: Proibido inferir dados. Se o documento não cita o valor, declare "Informação não disponível".
- RASTREABILIDADE (CITAÇÕES): Todo dado numérico deve ser seguido de (Fonte: [Arquivo], Seção: [X]).
- LÓGICA ANALÍTICA: Antes de responder, compare os valores entre os diferentes arquivos (ex: Livro Fiscal vs. Razão).
- FORMATAÇÃO MATEMÁTICA: Proibido o uso de notação LaTeX (como \[ \], $ ou \text). Apresente os cálculos em texto plano de forma direta.
- ISOLAMENTO TEMPORAL (MUITO CRÍTICO): Nunca cruze dados de anos diferentes. Se você está avaliando o Livro Fiscal de 2024, compare EXCLUSIVAMENTE com o Razão/Balancete de 2024.
- DATAMENTO (MUITO CRÍTICO): Resgate dados apenas do ano fiscal vigente, {ano_fiscal}, e de um ano antes. Nunca fique comparando anos fiscais distantes.
- PRIORIDADE DE DOCUMENTOS: Antes de declarar uma discrepância, verifica obrigatoriamente se existe algum documento do tipo "Resposta da Gestão", "Nota Técnica" ou "Justificativa" no contexto e usa as informações ali contidas para sanar dúvidas fiscais.

DOCUMENTOS FORNECIDOS (Contexto RAG):
{contexto_rag}

TÓPICO DE AUDITORIA A SER EXECUTADO:
{pergunta}

ESTRUTURA DA RESPOSTA ESPERADA (PASSO A PASSO OBRIGATÓRIO):
1. Extração de Dados: Liste os valores brutos encontrados no contexto (Ex: Total Contábil 2024 = X, Total Fiscal 2024 = Y).
2. Constatação: Descreva o fato encontrado comparando os dados extraídos.
3. Análise Técnica: Mostre a fórmula matemática aplicada em texto plano para provar se o limite de 2% foi ultrapassado ou se há gap.
4. Questionamento (SE APLICÁVEL): Formule a pergunta formal para a administração. Se não houver distorção, escreva "Nenhum apontamento a ser feito".
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
Você é um Auditor Sénior. A sua tarefa é ler as constatações de um relatório de auditoria recém-gerado e emitir um Parecer Final e uma Classificação de Risco.

RELATÓRIO DE AUDITORIA GERADO (Análises 1 a 4):
{relatorio_gerado}

REGRAS PARA A CONCLUSÃO:
1. Resumo de Riscos: Sintetize em 1 ou 2 parágrafos os principais problemas financeiros/fiscais encontrados nas análises acima (se houverem).
2. Classificação: Atribua uma (e apenas uma) das seguintes cores à situação final da empresa, baseada na gravidade dos apontamentos:
   - Sem ressalvas: Limpo ou sem ressalvas (Nenhum gatilho ou distorção relevante acionada).
   - Com ressalvas: Com ressalvas (Distorções leves, problemas de linearidade ou variações justificáveis de receita).
   - Adverso: Adverso (Divergências fiscais/contábeis claras ou cálculos errados de impostos).
   - Abstenção: Abstenção ou Negativa (Falta grave de dados, sumiço de bases de cálculo ou omissão massiva).
3. Justificativa da Cor: Explique sinteticamente (2-3 linhas) o que levou a essa cor.

ESTRUTURA DA RESPOSTA ESPERADA:
**Classificação de Risco:** [COR]
**Justificativa:** [Sua justificativa sintética]

**Resumo Executivo de Riscos:**
[Seu resumo das 4 análises]
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
        
        html_content = markdown.markdown(relatorio_completo, extensions=['tables'])
        
        html_com_estilo = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: 'Times New Roman', serif; line-height: 1.5; padding: 2em; color: #111; }}
                    h1 {{ color: #000; border-bottom: 3px double #000; padding-bottom: 5px; text-align: center; }}
                    h2 {{ color: #333; margin-bottom: 5px; }}
                    h3 {{ color: #444; background-color: #f4f4f4; padding: 5px; border-left: 4px solid #555; }}
                    p {{ font-size: 11pt; }}
                    hr {{ border: 0; border-top: 1px dashed #ccc; margin: 25px 0; }}
                    .footer {{ font-size: 9px; color: #666; text-align: center; margin-top: 50px; border-top: 1px solid #eee; padding-top: 10px; }}
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