import os
from llama_parse import LlamaParse

from langchain_text_splitters import MarkdownHeaderTextSplitter

class Parse():
    def parsionar_documento(documento):
        os.environ["LLAMA_CLOUD_API_KEY"] = "llx-XSv9a06oBoBoBTkqWcwFEAaVRfYRyPMGplU1XTrL0KfQ4Km5"

        documentos = LlamaParse(result_type="markdown").load_data(documento)

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(str(documentos))

        textos = []
        metadados = []
        ids_chroma = []

        id_contador = 1
        for pedaco in md_header_splits:
            pedaco.metadata["id_empresa"] = "vale" 
            pedaco.metadata["ano_fiscal"] = "2025"    
            pedaco.metadata["fonte"] = "balanco_2025.pdf"
            
            id_string = str(id_contador)
            pedaco.metadata["id"] = id_string
            
            textos.append(pedaco) 
            metadados.append(pedaco.metadata)
            ids_chroma.append(id_string)
            
            id_contador += 1

        return textos, metadados, ids_chroma, documentos