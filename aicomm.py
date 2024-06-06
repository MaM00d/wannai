import socket
# ,torch
# from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model
HEADERSIZE = 10
# class server:
def create_prompt(history, patient, doctor):
    prompt_template = (
        f"### HISTORY\n{history}\n\n### PATIENT\n{patient}\n\n### DOCTOR\n{doctor}</s>"
    )
    return prompt_template

def process_message(message):
    splited = message.split("~~~")
    if len(splited)>2 or len(splited)<2:
        return "error"
    prompt = create_prompt(splited[0],splited[1],"اديك في السقف تمحر اديك في الارض تفحر اديك في الجيركين تركن ")
#Ai CAll
    response = prompt.split("DOCTOR\n")[1][:-4]

    return response

def parse_message(msg):
    length = str(len(msg.encode('utf-8')))
    while len(length)<HEADERSIZE:
        length ="0"+length
    msg = length+msg
    print(f"message:'{msg}'")
    return msg

def unparse_message(msg):
    msglen = int(msg[:HEADERSIZE])

def receive_message(cs):
    print("Server is listening to the client")
    full_msg = ''
    new_msg = True
    while True:
        msg = cs.recv(16)
        print("message:{msg}")
        if new_msg:
            print("new msg len:",msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False
        print(f"full message length: {msglen}=={len(full_msg)}")
        full_msg += msg.decode("utf-8")
        if len(full_msg)-HEADERSIZE == msglen:
            message = parse_message(process_message(full_msg[HEADERSIZE:]))
            print(f"Sent to client: {message}")
            cs.sendall(bytes(message,'utf-8'))
            new_msg = True
            full_msg = ""
                

def start_server(host='0.0.0.0', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Server listening on {host}:{port}")
        client_socket, client_address = server_socket.accept()
        print(f"Connected by {client_address}")
        receive_message(client_socket)

        # while True:
        #     message = receive_message(client_socket)
        #     if message is None:
        #         print("no message")
        #         break
            # result = process_message(message)
        #     send_message(client_socket, result)

        # print(f"Connection with {client_address} closed")

# class ai:
#     def __init__(self):
#
#         self.device = "cuda:0"
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )
#
#         model_id = "FreedomIntelligence/AceGPT-13B"
#
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         mainmodel = AutoModelForCausalLM.from_pretrained(
#             model_id, quantization_config=quantization_config, device_map={"": 0}
#         )
#
#         lora_config = LoraConfig.from_pretrained(
#             "TheMETeam/wanas_model"
#         )
#         self.model = get_peft_model(mainmodel, lora_config)
#
#     def create_prompt(self,history, patient, doctor):
#         prompt_template = (
#             f"### HISTORY\n{history}\n\n### PATIENT\n{patient}\n\n### DOCTOR\n{doctor}</s>"
#         )
#         return prompt_template
#
#     def run(self,text):
#         splited = text.split("~~~")
#         if len(splited)>2 or len(splited)<2:
#             return "error"
#         prompt = self.create_prompt(splited[0],splited[1],"")
#
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         outputs = self.model.generate(**inputs, max_new_tokens=50)
#
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    start_server()

