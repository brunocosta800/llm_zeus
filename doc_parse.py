import os
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from supabase.client import Client, create_client
import tempfile

class Parse:
    
    @staticmethod
    def parsionar_documento(lista_documentos, cnpj):
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("As variáveis SUPABASE_URL e SUPABASE_SERVICE_KEY precisam estar no .env")

        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

        """
        Extrai e divide o texto dos PDFs usando LlamaParse.
        Processa arquivo por arquivo para garantir a rastreabilidade da fonte.
        """
        
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not llama_key:
            raise ValueError("ERRO: A variável LLAMA_CLOUD_API_KEY não foi encontrada no .env!")

        parser = LlamaParse(
            api_key=llama_key,
            result_type="markdown"
        )

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

        textos = []
        metadados = []
        ids_vetores = []
        todos_documentos = []

        id_contador = 1
        
        for caminho_arquivo in lista_documentos:
            nome_fonte_real = os.path.basename(caminho_arquivo)
            print(f"   -> Baixando e Lendo: {nome_fonte_real}...")
    
            try:
                res_bytes = supabase.storage.from_("evidences").download(caminho_arquivo)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(res_bytes)
                    caminho_temporario = tmp_file.name
                    
                doc_carregado = parser.load_data([caminho_temporario])
                todos_documentos.extend(doc_carregado)
                
                texto_markdown = doc_carregado[0].text if doc_carregado else ""
                md_header_splits = markdown_splitter.split_text(texto_markdown)
                
                os.remove(caminho_temporario)
                
            except Exception as e:
                print(f"Erro ao processar {nome_fonte_real}: {e}")

                if 'caminho_temporario' in locals() and os.path.exists(caminho_temporario):
                    os.remove(caminho_temporario)
            
            for pedaco in md_header_splits:
                pedaco.metadata["cnpj"] = cnpj    
                pedaco.metadata["fonte"] = nome_fonte_real
                
                id_string = str(id_contador)
                pedaco.metadata["id"] = id_string
                
                textos.append(pedaco.page_content) 
                metadados.append(pedaco.metadata)
                ids_vetores.append(id_string)
                
                id_contador += 1

        return textos, metadados, ids_vetores, todos_documentos