#!/usr/bin/env python
# coding=utf8

import argparse
import socket
import sys


class CaculatorClient(object):

    def __init__(self, serverAddress):
        self.serverAddress = serverAddress

    def caculate(self, expression):
        pass


class TCPClient(CaculatorClient):

    def caculate(self, expression):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(self.serverAddress)
            sock.sendall(expression + "\n")
            result = sock.recv(1024)
            return result
        finally:
            sock.close()


class UDPClient(CaculatorClient):

    def caculate(self, expression):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(expression + "\n", self.serverAddress)
            sock.settimeout(2)
            try:
                result = sock.recv(1024)
            except socket.timeout as e:
                result = "failed to caculate expression %s : %s" % (
                    expression, e.message)
            return result
        finally:
            sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", "-s", default="localhost", help="the server name or ip address")
    parser.add_argument(
        "--port", "-p", type=int, default=9000, help="the server port, default is 9000")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tcp", "-t", action="store_true", help="tcp client, default")
    group.add_argument("--udp", "-u", action="store_true", help="udp client")
    args = parser.parse_args()
    serverAddress = (args.server, args.port)
    while True:
        try:
            expression = raw_input("enter arithmetic expression: ")
            print("expression: " + expression)
            ClientClass = TCPClient
            if args.udp:
                ClientClass = UDPClient
            client = ClientClass(serverAddress)
            result = client.caculate(expression)
            print(result)
        except Exception as e:
            print("exception occurred: " + e.message)
            e_type, e_value, traceback = sys.exc_info()
            sys.excepthook(e_type, e_value, traceback)
            break
