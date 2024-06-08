import socket
HEADERSIZE = 10
def create_prompt(history, patient, doctor):
    prompt_template = (
        f"### HISTORY\n{history}\n\n### PATIENT\n{patient}\n\n### DOCTOR\n{doctor}</s>"
    )
    return prompt_template
def process_message(message):
    splited = message.split("~~~")
    if len(splited)>2 or len(splited)<2:
        return "error"
    history = splited[1]
    msg = splited[0]
    print("history: "+history)
    print("msg: "+msg)
    prompt = create_prompt(history,msg,"اديك في السقف تمحر اديك في الارض تفحر اديك في الجيركين تركن ")
    #Ai CAll
    response = prompt.split("DOCTOR\n")[1][:-4]

    return response
class Buffer:

    def __init__(self,sock):
        self.sock = sock
        self.buffer = b''

    def get_line(self):
        while b'!~!~!' not in self.buffer:
            data = self.sock.recv(1024)
            if not data: # socket closed
                return None
            self.buffer += data
        line,sep,self.buffer = self.buffer.partition(b'!~!~!')
        return line.decode()
if __name__ == "__main__":
    s = socket.socket()
    s.bind(('0.0.0.0',12345))
    s.listen()
    while True:
        c,a = s.accept()
        with c:
            print('Connected:',a)
            b = Buffer(c)
            while True:
                line = b.get_line()
                if line is None:
                    break
                msg=process_message(line)+"!~!~!"
                c.sendall(bytes(msg,'utf-8'))
                # print('line:',line)
                # print('msg:',msg)
                
        print('Disconnected:',a)

