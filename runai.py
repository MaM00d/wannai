def cut_till_any_substring(string, substrings):
    cut_index = len(string)  # Start with the full length of the string
    for substring in substrings:
        index = string.find(substring)
        if index != -1:
            cut_index = min(cut_index, index)
    return string[:cut_index]



def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
class ai:
    def __init__(self):

        from peft import LoraConfig 
        self.device = "cuda:0"
        from transformers import BitsAndBytesConfig,AutoTokenizer, AutoModelForCausalLM
        from peft.mapping import get_peft_model
        from torch import bfloat16

        model_id = "FreedomIntelligence/AceGPT-13B-chat"
        chroma_location = "./chroma"



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
            "wanas"
        )
        self.model = get_peft_model(mainmodel, lora_config)
        # self.rag = ChromaStore(chroma_location)


    def create_prompt(self,context, history ,patient, doctor):
        prompt_template = (
                f"[<>Wanas<>]<s>###Context\n{context}\n{history}\n###friend:{patient}\n###therapis:{doctor}"
        )
        return prompt_template


    def run(self,message,history):
        #context = self.rag.query_rag(message)

        prompt = self.create_prompt("",history,message,"")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(resp)
        cutted = remove_prefix(resp,prompt)
        print(cutted)
        finalresp = cut_till_any_substring(cutted,["###","<\s>","[<>Wanas<>]"])
        print(finalresp)
        return finalresp




class ChromaStore:
    def __init__(self, persist_directory):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores.chroma import Chroma
        model_name = "asafaya/bert-medium-arabic"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        self.persist_directory = persist_directory
        self.db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


    def query_rag(self, query_text, k=5):
        results = self.db.similarity_search_with_score(query_text, k)
        return results
