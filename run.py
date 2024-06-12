
# |%%--%%| <c9AXsMSZl5|NgWEZicMYY>
r"""°°°
# lOAD QUANTIZED MODEL
°°°"""
# |%%--%%| <NgWEZicMYY|cgdY3m6XoC>

AceGptModelName = "FreedomIntelligence/AceGPT-13B"

# |%%--%%| <cgdY3m6XoC|xaIkSdCqVo>

from transformers import BitsAndBytesConfig,AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=bfloat16,
)




tokenizer = AutoTokenizer.from_pretrained(AceGptModelName)
model = AutoModelForCausalLM.from_pretrained(
    AceGptModelName, quantization_config=quantization_config, device_map={"": 0}
)

#|%%--%%| <xaIkSdCqVo|XRXZs9dyCC>

tokenizer = AutoTokenizer.from_pretrained(AceGptModelName)


#|%%--%%| <XRXZs9dyCC|auHRX0BOfW>

def create_prompt(context, history ,patient, doctor):
    prompt_template = (
        f"### Context\n{context}\n{history}\n\n### PATIENT\n{patient}\n\n### DOCTOR\n{doctor}</s>"
    )
    return prompt_template
#|%%--%%| <auHRX0BOfW|HKORVZfg1n>
r"""°°°
#Model Using
°°°"""
#|%%--%%| <HKORVZfg1n|LrJWcimgC9>
from peft import LoraConfig 
from peft.mapping import get_peft_model
lora_config = LoraConfig.from_pretrained(
"models/parm1"
)
peft_model = get_peft_model(model, lora_config)

#|%%--%%| <LrJWcimgC9|bfYw1fZjo3>
r"""°°°
## PDFProcessor Class
°°°"""
# |%%--%%| <bfYw1fZjo3|xuPB8E5Mlp>

import os
import re
import PyPDF2
# folder_path= 'phy/'
class PDFProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    @staticmethod
    def clean_text(text):
        page_number_pattern = r'\bPage\b\s*\d+|\b\d+\b\s*(?:/\s*\d+)?'
        square_bracket_pattern = r'\[.*?\]'
        url_pattern = r'http[s]?://\S+|www\.\S+'
        text = re.sub(page_number_pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(square_bracket_pattern, '', text)
        text = re.sub(url_pattern, '', text)
        return text

    def read_pdfs_in_folder(self):
        concatenated_text = ""
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(self.folder_path, filename)
                text_output = ''
                with open(file_path, 'rb') as pdf_object:
                    pdf_reader = PyPDF2.PdfReader(pdf_object)
                    for page in pdf_reader.pages:
                        text_output += page.extract_text()
                concatenated_text += text_output + " "
        return concatenated_text


# |%%--%%| <xuPB8E5Mlp|WBmp3KxdVu>
r"""°°°
## TextEmbedder Class
°°°"""
# |%%--%%| <WBmp3KxdVu|Ktur6VjYvQ>

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd

class TextEmbedder:
    def __init__(self, model_name, model_kwargs=None, encode_kwargs=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def create_chunks(self, text):
        text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
        )
        docs = text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    def save_chunks_to_csv(self, chunks, output_file):
        df = pd.DataFrame({'chunks': chunks})
        df.to_csv(output_file, encoding='utf-8', index=False)


# |%%--%%| <Ktur6VjYvQ|BGcuFwdEI3>
r"""°°°
## ChromaStore Class
°°°"""
# |%%--%%| <BGcuFwdEI3|Kui5XKZr7r>

from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

class ChromaStore:
    def __init__(self, persist_directory, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    @staticmethod
    def calculate_chunk_ids(chunks):
        last_page_id = None
        current_chunk_index = 0
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id
        return chunks

    def add_to_chroma(self, chunks: list[Document]):
        chunks_with_ids = self.calculate_chunk_ids(chunks)
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"])
        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
        if new_chunks:
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
            self.db.persist()

    def query_rag(self, query_text, k=5):
        results = self.db.similarity_search_with_score(query_text, k)
        return results


# |%%--%%| <Kui5XKZr7r|tS6zj3lw1d>
r"""°°°
# RAG main Class
°°°"""
# |%%--%%| <tS6zj3lw1d|bVO2BC41gk>

folder_path = 'phy/'
pdf_processor = PDFProcessor(folder_path)
all_text = pdf_processor.read_pdfs_in_folder()
all_text = PDFProcessor.clean_text(all_text)

# Step 2: Embed and Chunk Text
model_name = "asafaya/bert-medium-arabic"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
text_embedder = TextEmbedder(model_name, model_kwargs, encode_kwargs)
chunks = text_embedder.create_chunks(all_text)
text_embedder.save_chunks_to_csv(chunks, 'rag.csv')

documents = []
for chunk in chunks:
    # Provide appropriate page_content for each chunk
    document = Document(page_content=chunk, metadata={"source": "source_value", "page": "page_value"})
    documents.append(document)
    
# Step 3: Add Chunks to Chroma and Query
chroma_store = ChromaStore("chroma", text_embedder.embeddings)
chroma_store.add_to_chroma(documents)
# query_results = chroma_store.query_rag("انا مش حابب شكلي اعمل ايه يا دكتور ؟")






# |%%--%%| <bVO2BC41gk|BaFBVgPlLF> 
class ai:
    def __init__(self):
        self.history=""
    def run(self,text):
        from transformers import TextStreamer
        query_results = chroma_store.query_rag(text)
        text = create_prompt(query_results,self.history, text, "")
        streamer = TextStreamer(text)
        self.history= self.history + text
        device = "cuda:0"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        outputs = peft_model.generate(**inputs,streamer=streamer,max_new_tokens=50)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    def clear_history(self):
        self.history=""



#|%%--%%| <BaFBVgPlLF|ERza08Sxaz>
chat = ai()
chat.run("hello")


#|%%--%%| <ERza08Sxaz|qY1DXY2pFp>

chat.clear_history()

#|%%--%%| <qY1DXY2pFp|eQTo7bddFR>



