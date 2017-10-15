#!/usr/bin/env python
# coding=utf8

import argparse
import SocketServer


class CaculatorHandlerMixIn():

    def handle(self):
        self.data = self.rfile.readline().strip()
        print("%s:%s request expression: %s" %
              (self.client_address[0], self.client_address[1], self.data))
        try:
            result = eval(self.data)
            if type(result) not in [int, long, float]:
                result = "'%s' is not a valid arithmetic expression" % (
                    self.data,)
        except:
            result = "'%s' is not a valid arithmetic expression" % (self.data,)
        self.wfile.write(str(result))


class TCPHandler(CaculatorHandlerMixIn, SocketServer.StreamRequestHandler):
    pass


class UDPHandler(CaculatorHandlerMixIn, SocketServer.DatagramRequestHandler):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", "-p", type=int, default=9000, help="the server port, default is 9000")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tcp", "-t", action="store_true", help="serve by tcp, default")
    group.add_argument("--udp", "-u", action="store_true", help="serve by udp")
    args = parser.parse_args()
    serverAddress = ('', args.port)

    handlerClass = TCPHandler
    serverClass = SocketServer.ThreadingTCPServer
    if args.udp:
        handlerClass = UDPHandler
        serverClass = SocketServer.ThreadingUDPServer
    caculatorServer = serverClass(serverAddress, handlerClass)
    caculatorServer.serve_forever()
