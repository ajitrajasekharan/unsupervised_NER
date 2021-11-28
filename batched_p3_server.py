import time
import sys 
from http.server import BaseHTTPRequestHandler, HTTPServer
import batched_instance

HOST_NAME = ''
PORT_NUMBER = 8086
disp_count = 0 
RESP_CODE=200


class MyHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

    def do_GET(self):
        global disp_count
        print("IN Get:",self.path)
        if (self.path != "/favicon.ico"):
            print("***Handle HTTP:",disp_count,self.path)
            disp_count += 1
            self.process(RESP_CODE)
        else:
            print("    +++Skipping favico request")
            self.send_response(RESP_CODE)

    def process(self,status_code):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        obj = batched_instance.get_instance()
        obj.handler(self)
        #response =  bytes(content, 'UTF-8')
        #self.wfile.write(response)


if __name__ == '__main__':
    server_class = HTTPServer
    if len(sys.argv) >= 2:
        PORT_NUMBER =int(sys.argv[1])
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
