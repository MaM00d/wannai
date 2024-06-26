import gdown
def cut_till_any_substring(string, substrings):
    cut_index = len(string)  # Start with the full length of the string
    for substring in substrings:
        index = string.find(substring)
        if index != -1:
            cut_index = min(cut_index, index)
            return string[:cut_index]
        return string




class ai:
    def __init__(self):
        from peft import LoraConfig 
        self.device = "cuda:0"
        from transformers import BitsAndBytesConfig,AutoTokenizer, AutoModelForCausalLM
        from peft.mapping import get_peft_model
        from torch import bfloat16

        model_id = "FreedomIntelligence/AceGPT-13B-chat"
        self.chroma_location = "./chroma"

        url = "https://drive.google.com/drive/folders/10FjpYFSlIAkZyRB3ibT-hrjXyVe19wSb?usp=sharing"
        gdown.download_folder(url)


        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bfloat16,
        )


        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        mainmodel = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, device_map={"": 0}
        )

        lora_config = LoraConfig.from_pretrained(
            "./model"
        )
        self.model = get_peft_model(mainmodel, lora_config)
    def create_prompt(self,context, history ,patient, doctor):
        prompt_template = (
                f"[<>Wanas<>]<s>###Context\n{context}\n{history}\n###friend:{patient}\n###therapis:{doctor}"
        )
        return prompt_template
    def run(self,message,history):
        #context = self.rag.query_rag(message)
        context = "انت دكتور نفسي ترد على صديقك تحاول ان تساعده على حل مشاكله النفسية"
        prompt = self.create_prompt(context,history,message,"")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=250)
        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(resp)
        cutted =resp[(len(prompt)-3):] 
        # print(cutted)
        finalresp = cut_till_any_substring(cutted,["###","<\s>","[<>Wanas<>]"])
        # print(finalresp)
        return  finalresp




from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

class rag:
    def __init__(self,CHROMA_PATH):
        self.CHROMA_PATH = CHROMA_PATH
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("loading embedding model and index ....")

        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model_id = "asafaya/bert-large-arabic"

        self.embeddings = HuggingFaceEmbeddings(
            model_name = embedding_model_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    def clean_text(text):
        import re
        # This regex pattern matches common page number patterns like "Page 1", "1", "1/10", "Page: 1", etc.
        page_number_pattern = r'\bPage\b\s*\d+|\b\d+\b\s*(?:/\s*\d+)?'
        # This regex pattern matches square brackets and their contents, including Arabic text
        square_bracket_pattern = r'\[.*?\]'
        # This regex pattern matches URLs
        url_pattern = r'http[s]?://\S+|www\.\S+'
        # Substitute matching patterns with an empty string
        text = re.sub(page_number_pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(square_bracket_pattern, '', text)
        text = re.sub(url_pattern, '', text)
        return text
    def read_pdfs_in_folder(self,folder_path):
        import PyPDF2
        import os
        concatenated_text = ""
        
        # List all files in the given folder
        for filename in os.listdir(folder_path):
            # Check if the file is a PDF
            if filename.endswith('.pdf'):
                # Construct the full file path
                file_path = os.path.join(folder_path, filename)
                # Open the PDF file
                text_output = ''
                with open(file_path, 'rb') as pdf_object:
                    pdf_reader = PyPDF2.PdfReader(pdf_object)
                    for i in pdf_reader.pages:
                        text_output += i.extract_text()
                concatenated_text += text_output + " "  # Add a space to separate texts from different pages
        
        return concatenated_text
    def chunking(self,all_text):
        from langchain_experimental.text_splitter import SemanticChunker
        text_splitter = SemanticChunker(embeddings,breakpoint_threshold_type="percentile")


        docs = text_splitter.create_documents([all_text])

        chuncks=[]
        for doc in docs:
            chunk = doc.page_content
            chuncks.append(chunk)
        return chuncks
    def calculate_chunk_ids(self,chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks
    def add_to_chroma(self,chunks: list[Document]):
        # Load the existing database.
        db = Chroma(
            persist_directory=self.CHROMA_PATH, embedding_function=self.embeddings
        )

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")
    def create_rag(self):
        text = self.read_pdfs_in_folder("./source")
        chuncks = self.chunking(text)
        chuncks = self.calculate_chunk_ids(chuncks)
        self.add_to_chroma(chuncks)
    def query_rag(self,query_text: str):
        # Prepare the DB.
        embedding_function =self.embeddings 
        db = Chroma(persist_directory=self.CHROMA_PATH, embedding_function=embedding_function)
        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        return results


class emotion:
    def __init__(self):
        from torch import device,cuda
        from transformers import AutoModelForSequenceClassification,BertTokenizerFast

        url = "https://drive.google.com/drive/folders/1U85_04elDF5L2SZXMGjkcNsjjZSCTzj4?usp=sharing"
        gdown.download_folder(url)

        self.device = device("cuda" if cuda.is_available() else "cpu")
        BERT_MODEL_NAME = 'aubmindlab/bert-base-arabertv02-twitter'
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME,device=self.device)
        save_directory = "./cc"
        self.model = AutoModelForSequenceClassification.from_pretrained(save_directory).to(self.device)
    def run(self,text,max_length=128):
        from torch import no_grad,argmax
        self.model.eval()
        encoding = self.tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True).to(self.device)
        with no_grad():
            outputs = self.model(**encoding)
        predicted_class = argmax(outputs.logits,dim=1)[0].item()
        return predicted_class
