# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals

import sys


class Parser(object):
    def __init__(self, name, encoding='utf-8'):
        self.name = name
        self.encoding = encoding

    def to_unicode(self, text):
        '''
        Convert char encoding to unicode
        :param text:
        :return:
        '''
        if sys.version_info[0] < 3:
            text_alt = text.encode('utf-8', 'error')
            text_alt = text_alt.decode('string_escape', errors='ignore')
            text_alt = text_alt.decode('utf-8')
            return text_alt
        else:
            return text

    def connect(self):
        '''
        Return connection object for this parser type
        :return:
        '''
        raise NotImplemented

    def close(self):
        '''
        Kill this parser
        :return:
        '''
        raise NotImplemented


class ParserConnection(object):
    '''
    Default connection object assumes local parser object
    '''

    def __init__(self, parser):
        self.parser = parser

    def _connection(self):
        raise NotImplemented

    def parse(self, document, text):
        return self.parser.parse(document, text)
