#!/usr/bin/env python
# coding=utf8

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import cgi
import urlparse


class TrivialHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed_path = urlparse.urlparse(self.path)
        message_parts = [
            'Client Values:',
            'client_address=%s (%s)' % (
                self.client_address, self.address_string()),
            'method=%s' % self.command,
            'path=%s' % self.path,
            'real_path=%s' % parsed_path.path,
            'query=%s' % parsed_path.query,
            'request_version=%s' % self.request_version,
            '',
            'Server Values:',
            'server_version=%s' % self.server_version,
            'sys_version=%s' % self.sys_version,
            'protocol_version=%s' % self.protocol_version,
            '',
            'Headers Received:',
        ]
        for key, value in self.headers.items():
            message_parts.append('%s=%s' % (key, value))
        message_parts.append('')
        message = '\r\n'.join(message_parts)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(message)

    def do_POST(self):
        # begin the response
        self.send_response(200)
        self.end_headers()
        self.wfile.write('Client: %s\n' % str(self.client_address))
        self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
        self.wfile.write('Path: %s\n' % self.path)
        self.wfile.write('Method: %s\n' % self.command)
        self.wfile.write('Form Data:\n')
        # parse the form data posted
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={
                                    'REQUEST_METHOD': "POST",
                                    'CONTENT_TYPE': self.headers['content-type']
                                }
                                )
        # Echo back information about what was posted in the form
        for field_key in form.keys():
            field_item = form[field_key]
            if isinstance(field_item, list):
                self.wfile.write("\t%s has multiple values:\n" % field_key)
                for item in field_item:
                    if item.filename:
                        file_content = item.file.read()
                        content_len = len(file_content)
                        del file_content
                        self.wfile.write('\t\tfile { filename %s, length %d bytes }\n'
                                         % (item.filename, content_len))
                    else:
                        self.wfile.write('\t\t%s\n' % item.value)
            elif field_item.filename:
                # The field contains an uploaded file
                file_data = field_item.file.read()
                file_len = len(file_data)
                del file_data
                self.wfile.write('\t%s: file { filename"%s", length %d bytes }\n'
                                 % (field_key, field_item.filename, file_len))
            else:
                # Regular form value
                self.wfile.write('\t%s=%s\n' % (field_key, field_item.value))


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass

if __name__ == '__main__':
    serverAddress = ('localhost', 8090)
    server = ThreadingHTTPServer(serverAddress, TrivialHTTPRequestHandler)
    print("serve on port 8090")
    server.serve_forever()
