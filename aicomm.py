import socket,time,gdown
from zipfile import ZipFile 
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


class server:
    def __init__(self):
        gdown.download("https://drive.google.com/file/d/14VXJ04KugJSoLsPtm49BT5T3cihind4q/view?usp=sharing",".")
        with ZipFile("./checkpoint-70.zip", 'r') as zObject: 
          
            # Extracting all the members of the zip  
            # into a specific location. 
            zObject.extractall( 
        path="./wanas") 
        from runai import ai
        self.el_ai = ai()
        # start server
        self.s = socket.socket()
        self.s.bind(('0.0.0.0',12345))
        self.s.listen()
    def run(self):
        while True:
            c,a = self.s.accept()
            with c:
                print('Connected:',a)
                b = Buffer(c)
                while True:
                    line = b.get_line()
                    if line is None:
                        break
                    # process message logic (calling ai)
                    message,history = self.unparse(line)
                    msg = self.process(message,history)
                    packet = self.parse(msg)
                    # processed_message = process_message(line)
                    # parse message
                    # msg=message+"!~!~!"
                    # encode and send the response back
                    c.sendall(bytes(packet,'utf-8'))
                    # print('line:',line)
                    # print('msg:',msg)
                    
            print('Disconnected:',a)
    def process(self,message,history):

        #Ai CAll
        # time.sleep(10)

        response = self.el_ai.run(message,history)
        # response = "hello from ai"
        return response

    def unparse(self,text):
        splited = text.split("~~~")
        if len(splited)>2 or len(splited)<2:
            return "error"
        history = splited[1]
        msg = splited[0]
        print("history: "+history)
        print("msg: "+msg)
        return msg,history
    def parse(self,message):
        msg=message+"!~!~!"
        return msg


if __name__ == "__main__":
    serv = server()
    serv.run()
