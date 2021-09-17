#!/usr/bin/env python

import instance
import sys
"""
Very simple HTTP server in python.

Usage::
    ./dummy-web-server.py [<port>]

Send a GET request::
    curl http://localhost

Send a HEAD request::
    curl -I http://localhost

Send a POST request::
    curl -d "foo=bar&bin=baz" http://localhost

"""
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_PORT=8086

HTTPD=None

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()

    def do_GET(self):
        global HTTPD
        self._set_headers()
        if (self.path != "/favicon.ico"):
            print( self.path)
            obj = instance.get_instance()
            try:
                obj.handler(self)
            except:
                print("Shutting down")
                HTTPD.shutdown()
                sys.exit(1)
                raise SystemExit

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers()
        try:
           self.wfile.write("<html><body><h1>POST!</h1></body></html>")
        except:
           print("Unexpected error:", sys.exc_info()[0])

def run(server_class=HTTPServer, handler_class=S, port=DEFAULT_PORT):
    global HTTPD
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    HTTPD = httpd
    print('Starting httpd on port:', port,'...')
    try:
        httpd.serve_forever()
    except:
        httpd.shutdown()
        print("Main run")
        sys.exit(1)
        raise SystemExit

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 3:
        run(port=int(argv[1]))
    else:
        print("Usage: port [ 8070 ] <desc file name>")
