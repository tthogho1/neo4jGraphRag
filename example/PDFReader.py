from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.docstore.document import Document 
import glob
  
# demodata/ *.pdf の読み込み 
def load_pdf(path: str= "demodata/*.pdf") -> list: 

    pdf_resources = [] 
    for file in glob.glob(path): 
        print(file) 
        loader = PyPDFLoader(file) 
        pages = loader.load_and_split() 
        file_text = ''.join([x.page_content for x in pages]) 
        doc = Document(page_content=file_text, metadata={'source': file}) 
        pdf_resources.append(doc) 

    return pdf_resources 

# テキストのチャンク分割 
def split_text(docs: list) -> list:     

    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=350, 
        chunk_overlap=100, 
    ) 

    chunked_resources = text_splitter.split_documents(docs) 
    return chunked_resources 