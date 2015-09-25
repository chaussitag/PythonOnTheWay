#!/usr/bin/env python
# coding=utf8
'''
Created on 2015年9月14日

@author: daiguozhou
'''

import argparse
import itertools
import mimetools
import mimetypes
import os
import urllib2


class MultipartForm(object):

    def __init__(self):
        self.form_fields = []
        self.files = []
        self.boundary = mimetools.choose_boundary()

    def get_content_type(self):
        return 'multipart/form-data; boundary=%s' % self.boundary

    def add_field(self, name, value):
        """Add a simple field to the form data"""
        self.form_fields.append((name, value))

    def add_file(self, field_name, file_name, fp, mimetype=None):
        """Add a file to be uploaded"""
        if mimetype == None:
            mimetype = mimetypes.guess_type(
                file_name)[0] or "application/octet-stream"
        filecontent = fp.read()
        self.files.append((field_name, file_name, mimetype, filecontent))

    def __str__(self, *args, **kwargs):
        """Return a string representation of the form data, including attached files"""
        parts_boundery = '--' + self.boundary
        parts = []
        "Add the form fields"
        parts.extend(
            [
                parts_boundery,
                'Content-Disposition: form-data; name="%s"' % name,
                '',
                value,
            ]
            for name, value in self.form_fields
        )

        "Add the file contents"
        parts.extend(
            [
                parts_boundery,
                'Content-Disposition: file; name="%s"; filename="%s"' % (
                    field_name, file_name),
                'Content-type: %s' % content_type,
                '',
                file_content,
            ]
            for field_name, file_name, content_type, file_content in self.files
        )

        flattendened = list(itertools.chain(*parts))
        flattendened.append("--" + self.boundary + "--")
        flattendened.append('')
        return '\r\n'.join(flattendened)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", "--name", "-n", dest="form_fields", action="append", default=[],
                        nargs=2, metavar=('field_name', 'filed_value'),
                        help="specify a form field as: '--field filed_name field_value'")
    parser.add_argument("--file", "-f", dest="files", action="append", default=[],
                        nargs=3, metavar=('field_name', 'file_path', 'file_name'),
                        help="specify path to a file to be uploaded as: '--file field_name file_path file_name'")
    args = parser.parse_args()

    form = MultipartForm()
    for filedinfo in args.form_fields:
        filed_name = filedinfo[0].strip()
        field_value = filedinfo[1].strip()
        form.add_field(filed_name, field_value)

    for fileinfo in args.files:
        field_name = fileinfo[0].strip()
        file_path = fileinfo[1].strip()
        file_name = fileinfo[2].strip()
        if not os.path.exists(file_path):
            print("the file '%s' not exist" % file_path)
            continue
        with open(file_path) as fp:
            form.add_file(field_name, file_name, fp)

    form_content = str(form)

    request = urllib2.Request("http://localhost:8090")
    request.add_header(
        "User-agent", 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0')
    request.add_header('Content-type', form.get_content_type())
    request.add_header('Content-length', len(form_content))
    request.add_data(form_content)

    print('Outgoing data:')
    print(request.get_data())

    response = urllib2.urlopen(request)

    print('')
    print('Server response:')
    print(response.read())
