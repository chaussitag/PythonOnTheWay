#!/usr/bin/env python
#coding=utf8

'''
Created on 2015年10月22日

@author: daiguozhou
'''

import cgi
import mimetypes
import os
import posixpath
import sys
import time
import urllib
import urlparse
from wsgiref.simple_server import make_server

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


class FileContentIter(object):
    """Iterate a local file chunk by chunk.

    Arguments are the file path, and size of each chunk returned, default to 1M bytes.
    """
    def __init__(self, path, chunk_size=1024 * 1024):
        self.path = path
        self.chunk_size = chunk_size

    def __iter__(self):
        with open(self.path, 'rb') as f:
            parts = f.read(self.chunk_size)
            while parts:
                yield parts
                parts = f.read(self.chunk_size)

# some code was borrowed from SimpleHTTPServer

class FileExplorerApp(object):
    """Define a wsgi-compatable application.

    This class is used as the wsgi application, pass it to a wsgi server,
    the wsgi server will create an instance of this class.
    object of this class is an iterator (using the yield keyword)
    """

    DEFAULT_ERROR_MESSAGE = """\
<head>
<title>Error response</title>
</head>
<body>
<h1>Error response</h1>
<p>Error code %(code)d.
<p>Message: %(message)s.
<p>Error code explanation: %(code)s = %(explain)s.
</body>
"""

    weekdayname = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    monthname = [None,
                 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if not mimetypes.inited:
        mimetypes.init()  # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream',  # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
    })

    def __init__(self, environ, start_response):
        """The constructor, called by the wsgi server.

        The 'start_response' argument is a wsgi-server provided callable object,
        must be called before the wsgi-server iterate over this object.
        """
        self.environ = environ
        self.start_response = start_response
        self.request_method = environ['REQUEST_METHOD']
        self.path = environ['PATH_INFO']

    def __iter__(self):
        """response to a GET request to some path.

        Only the 'GET' method is supported.
        Return the file content if requested path is a file,
        return the children list if requested path is a directory.
        """
        if self.request_method != 'GET':
            response_status, response_headers, response_content = self.onError(
                501, 'Not Implemented', 'Server does not support this operation')
        else:
            response_status, response_headers, response_content = self.onGet()

        # start_response() must be called before yielding any data returned to the wsgi-server.
        self.start_response(response_status, response_headers)
        for content_part in response_content:
            yield content_part

    def onError(self, error_code, short_message, long_message):
        response_status = "%d %s" % (error_code, short_message)
        response_headers = [('Content-type', "text/html")]
        response_content = [self.DEFAULT_ERROR_MESSAGE % {
            'code': error_code,
            'message': short_message,
            'explain': long_message,
        }]
        return (response_status, response_headers, response_content)

    def onGet(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            parts = urlparse.urlsplit(self.path)
            if not parts.path.endswith('/'):
                # redirect browser - doing basically what apache does
                response_status = '301 Moved Permanently'
                new_parts = (parts[0], parts[1], parts[2] + '/',
                             parts[3], parts[4])
                new_url = urlparse.urlunsplit(new_parts)
                response_headers = [('Location', new_url)]
                response_content = ['']
                return (response_status, response_headers, response_content)

            else:
                has_index_html = False
                for index in ["index.html", "index.htm"]:
                    index = os.path.join(path, index)
                    if os.path.exists(index):
                        path = index
                        has_index_html = True
                        break

                if not has_index_html:
                    return self.onListDirectory(path)

        return self.onFileContent(path)

    def onFileContent(self, file_path):
        ctype = self.guess_type(file_path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(file_path, 'rb')
        except IOError:
            return self.onError(404, 'Not Found', 'Nothing matches the given URI')

        try:
            response_status = '200 OK'
            response_headers = [('Content-type', ctype), ]
            fs = os.fstat(f.fileno())
            response_headers.append(('Content-Length', str(fs[6])))
            response_headers.append(
                ("Last-Modified", self.date_time_string(fs.st_mtime)))
            return (response_status, response_headers, FileContentIter(file_path))
        finally:
            f.close()

    def onListDirectory(self, dir_path):
        """Helper to produce a directory listing (absent index.html).

        Return value is either a file object, or None (indicating an
        error).  In either case, the headers are sent, making the
        interface the same as for send_head().

        """
        try:
            file_list = os.listdir(dir_path)
        except os.error:
            return self.onError(404, "No permission to list directory", 'Nothing matches the given URI')

        file_list.sort(key=lambda a: a.lower())
        f = StringIO()
        displaypath = cgi.escape(urllib.unquote(self.path))
        f.write('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write("<html>\n<title>Directory listing for %s</title>\n" %
                displaypath)
        f.write("<body>\n<h2>Directory listing for %s</h2>\n" % displaypath)
        f.write("<hr>\n<ul>\n")
        for name in file_list:
            fullname = os.path.join(dir_path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            f.write('<li><a href="%s">%s</a>\n'
                    % (urllib.quote(linkname), cgi.escape(displayname)))
        f.write("</ul>\n<hr>\n</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        response_status = '200 OK'
        response_headers = []
        encoding = sys.getfilesystemencoding()
        response_headers.append(
            ("Content-type", "text/html; charset=%s" % encoding))
        response_headers.append(("Content-Length", str(length)))
        response_content = [f.getvalue()]
        f.close()
        return (response_status, response_headers, response_content)

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.

        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)

        """
        # abandon query parameters
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        # Don't forget explicit trailing slash when normalizing. Issue17324
        trailing_slash = path.rstrip().endswith('/')
        path = posixpath.normpath(urllib.unquote(path))
        words = path.split('/')
        words = filter(None, words)
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)
        if trailing_slash:
            path += '/'
        return path

    def date_time_string(self, timestamp=None):
        """Return the current date and time formatted for a message header."""
        if timestamp is None:
            timestamp = time.time()
        year, month, day, hh, mm, ss, wd, y, z = time.gmtime(timestamp)
        s = "%s, %02d %3s %4d %02d:%02d:%02d GMT" % (
            self.weekdayname[wd],
            day, self.monthname[month], year,
            hh, mm, ss)
        return s

    def guess_type(self, path):
        """Guess the type of a file.

        Argument is a PATH (a filename).

        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.

        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.

        """

        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']

if __name__ == '__main__':
    server = make_server('localhost', 8000, FileExplorerApp)
    sock_name = server.socket.getsockname()
    print("Serving HTTP on %s, port %s" % (sock_name[0], sock_name[1]))
    server.serve_forever()
