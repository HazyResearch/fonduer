import logging
import os
import re
import shutil
import subprocess
from builtins import object, range, str, zip
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from editdistance import eval as editdist  # Alternative library: python-levenshtein


class VisualLinker(object):
    def __init__(self, time=False, verbose=False):
        self.logger = logging.getLogger(__name__)
        self.pdf_file = None
        self.verbose = verbose
        self.time = time
        self.coordinate_map = None
        self.pdf_word_list = None
        self.html_word_list = None
        self.links = None
        self.pdf_dim = None
        delimiters = (
            r"([\(\)\,\?\u2212\u201C\u201D\u2018\u2019\u00B0\*']|(?<!http):|\.$|\.\.\.)"
        )
        self.separators = re.compile(delimiters)

        # Check if poppler-utils is installed AND the version is 0.36.0 or above
        if shutil.which("pdfinfo") is None or shutil.which("pdftotext") is None:
            raise RuntimeError("poppler-utils is not installed or they are not in PATH")
        version = subprocess.check_output(
            "pdfinfo -v", shell=True, stderr=subprocess.STDOUT, universal_newlines=True
        )
        m = re.search(r"\d\.\d{2}\.\d", version)
        if int(m.group(0).replace(".", "")) < 360:
            raise RuntimeError(
                "Installed poppler-utils's version is %s, but should be 0.36.0 or above"
                % m.group(0)
            )

    def parse_visual(self, document_name, sentences, pdf_path):
        self.sentences = sentences
        self.pdf_file = (
            pdf_path if os.path.isfile(pdf_path) else pdf_path + document_name + ".pdf"
        )
        if not os.path.isfile(self.pdf_file):
            self.pdf_file = self.pdf_file[:-3] + "PDF"
        try:
            self.extract_pdf_words()
        except RuntimeError as e:
            self.logger.exception(e)
            return
        self.extract_html_words()
        self.link_lists(search_max=200)
        for sentence in self.update_coordinates():
            yield sentence

    def extract_pdf_words(self):
        self.logger.debug(
            "pdfinfo '{}' | grep -a Pages | sed 's/[^0-9]*//'".format(self.pdf_file)
        )
        num_pages = subprocess.check_output(
            "pdfinfo '{}' | grep -a Pages | sed 's/[^0-9]*//'".format(self.pdf_file),
            shell=True,
        )
        pdf_word_list = []
        coordinate_map = {}
        for i in range(1, int(num_pages) + 1):
            self.logger.debug(
                "pdftotext -f {} -l {} -bbox-layout '{}' -".format(
                    str(i), str(i), self.pdf_file
                )
            )
            html_content = subprocess.check_output(
                "pdftotext -f {} -l {} -bbox-layout '{}' -".format(
                    str(i), str(i), self.pdf_file
                ),
                shell=True,
            )
            soup = BeautifulSoup(html_content, "html.parser")
            pages = soup.find_all("page")
            pdf_word_list_i, coordinate_map_i = self._coordinates_from_HTML(pages[0], i)
            pdf_word_list += pdf_word_list_i
            # update coordinate map
            coordinate_map.update(coordinate_map_i)
        self.pdf_word_list = pdf_word_list
        self.coordinate_map = coordinate_map
        if len(self.pdf_word_list) == 0:
            raise RuntimeError(
                "Words could not be extracted from PDF: %s" % self.pdf_file
            )
        # take last page dimensions
        page_width, page_height = (
            int(float(pages[0].get("width"))),
            int(float(pages[0].get("height"))),
        )
        self.pdf_dim = (page_width, page_height)
        if self.verbose:
            self.logger.info("Extracted {} pdf words".format(len(self.pdf_word_list)))

    def _coordinates_from_HTML(self, page, page_num):
        pdf_word_list = []
        coordinate_map = {}
        block_coordinates = {}
        blocks = page.find_all("block")
        i = 0  # counter for word_id in page_num
        for block in blocks:
            x_min_block = int(float(block.get("xmin")))
            y_min_block = int(float(block.get("ymin")))
            lines = block.find_all("line")
            for line in lines:
                y_min_line = int(float(line.get("ymin")))
                y_max_line = int(float(line.get("ymax")))
                words = line.find_all("word")
                for word in words:
                    xmin = int(float(word.get("xmin")))
                    xmax = int(float(word.get("xmax")))
                    for content in self.separators.split(word.getText()):
                        if len(content) > 0:  # Ignore empty characters
                            word_id = (page_num, i)
                            pdf_word_list.append((word_id, content))
                            coordinate_map[word_id] = (
                                page_num,
                                y_min_line,
                                xmin,
                                y_max_line,
                                xmax,
                            )
                            block_coordinates[word_id] = (y_min_block, x_min_block)
                            i += 1
        # sort pdf_word_list by page, block top then block left, top, then left
        pdf_word_list = sorted(
            pdf_word_list,
            key=lambda word_id__: block_coordinates[word_id__[0]]
            + coordinate_map[word_id__[0]][1:3],
        )
        return pdf_word_list, coordinate_map

    def extract_html_words(self):
        html_word_list = []
        for sentence in self.sentences:
            for i, word in enumerate(sentence.words):
                html_word_list.append(((sentence.stable_id, i), word))
        self.html_word_list = html_word_list
        if self.verbose:
            self.logger.info("Extracted {} html words".format(len(self.html_word_list)))

    def link_lists(self, search_max=100, edit_cost=20, offset_cost=1):
        # NOTE: there are probably some inefficiencies here from rehashing words
        # multiple times, but we're not going to worry about that for now

        def link_exact(l, u):
            l, u, L, U = get_anchors(l, u)
            html_dict = defaultdict(list)
            pdf_dict = defaultdict(list)
            for i, (_, word) in enumerate(self.html_word_list[l:u]):
                if html_to_pdf[l + i] is None:
                    html_dict[word].append(l + i)
            for j, (_, word) in enumerate(self.pdf_word_list[L:U]):
                if pdf_to_html[L + j] is None:
                    pdf_dict[word].append(L + j)
            for word, html_list in list(html_dict.items()):
                pdf_list = pdf_dict[word]
                if len(html_list) == len(pdf_list):
                    for k in range(len(html_list)):
                        html_to_pdf[html_list[k]] = pdf_list[k]
                        pdf_to_html[pdf_list[k]] = html_list[k]

        def link_fuzzy(i):
            (_, word) = self.html_word_list[i]
            l = u = i
            l, u, L, U = get_anchors(l, u)
            offset = int(L + float(i - l) / (u - l) * (U - L))
            searchIndices = np.clip(offset + search_order, 0, M - 1)
            cost = [0] * search_max
            for j, k in enumerate(searchIndices):
                other = self.pdf_word_list[k][1]
                if (
                    word.startswith(other)
                    or word.endswith(other)
                    or other.startswith(word)
                    or other.endswith(word)
                ):
                    html_to_pdf[i] = k
                    return
                else:
                    cost[j] = int(editdist(word, other)) * edit_cost + j * offset_cost
            html_to_pdf[i] = searchIndices[np.argmin(cost)]
            return

        def get_anchors(l, u):
            while l >= 0 and html_to_pdf[l] is None:
                l -= 1
            while u < N and html_to_pdf[u] is None:
                u += 1
            if l < 0:
                l = 0
                L = 0
            else:
                L = html_to_pdf[l]
            if u >= N:
                u = N
                U = M
            else:
                U = html_to_pdf[u]
            return l, u, L, U

        def display_match_counts():
            matches = sum(
                [
                    html_to_pdf[i] is not None
                    and self.html_word_list[i][1]
                    == self.pdf_word_list[html_to_pdf[i]][1]
                    for i in range(len(self.html_word_list))
                ]
            )
            total = len(self.html_word_list)
            self.logger.info(
                "({:d}/{:d}) = {:.2f}".format(matches, total, matches / total)
            )
            return matches

        N = len(self.html_word_list)
        M = len(self.pdf_word_list)

        try:
            assert N > 0 and M > 0
        except Exception:
            self.logger.exception("N = {} and M = {} are invalid values.".format(N, M))

        html_to_pdf = [None] * N
        pdf_to_html = [None] * M
        search_radius = search_max // 2

        # first pass: global search for exact matches
        link_exact(0, N)
        if self.verbose:
            self.logger.debug("Global exact matching:")
            display_match_counts()

        # second pass: local search for exact matches
        for i in range(((N + 2) // search_radius) + 1):
            link_exact(
                max(0, i * search_radius - search_radius),
                min(N, i * search_radius + search_radius),
            )
        if self.verbose:
            self.logger.debug("Local exact matching:")
            display_match_counts()

        # third pass: local search for approximate matches
        search_order = np.array(
            [(-1) ** (i % 2) * (i // 2) for i in range(1, search_max + 1)]
        )
        for i in range(len(html_to_pdf)):
            if html_to_pdf[i] is None:
                link_fuzzy(i)
        if self.verbose:
            self.logger.debug("Local approximate matching:")
            display_match_counts()

        # convert list to dict
        matches = sum(
            [
                html_to_pdf[i] is not None
                and self.html_word_list[i][1] == self.pdf_word_list[html_to_pdf[i]][1]
                for i in range(len(self.html_word_list))
            ]
        )
        total = len(self.html_word_list)
        if self.verbose:
            self.logger.debug(
                "Linked {:d}/{:d} ({:.2f}) html words exactly".format(
                    matches, total, matches / total
                )
            )
        self.links = OrderedDict(
            (self.html_word_list[i][0], self.pdf_word_list[html_to_pdf[i]][0])
            for i in range(len(self.html_word_list))
        )

    def _calculate_offset(self, listA, listB, seedSize, maxOffset):
        wordsA = zip(*listA[:seedSize])[1]
        wordsB = zip(*listB[:maxOffset])[1]
        offsets = []
        for i in range(seedSize):
            try:
                offsets.append(wordsB.index(wordsA[i]) - i)
            except Exception:
                pass
        return int(np.median(offsets))

    def display_links(self, max_rows=100):
        html = []
        pdf = []
        j = []
        for i, l in enumerate(self.links):
            html.append(self.html_word_list[i][1])
            for k, b in enumerate(self.pdf_word_list):
                if b[0] == self.links[self.html_word_list[i][0]]:
                    pdf.append(b[1])
                    j.append(k)
                    break
        try:
            assert len(pdf) == len(html)
        except Exception:
            self.logger.exception("PDF and HTML are not the same length")

        total = 0
        match = 0
        for i, word in enumerate(html):
            total += 1
            if word == pdf[i]:
                match += 1
        self.logger.info((match, total, match / total))

        data = {
            # 'i': range(len(self.links)),
            "html": html,
            "pdf": pdf,
            "j": j,
        }
        pd.set_option("display.max_rows", max_rows)
        self.logger.info(pd.DataFrame(data, columns=["html", "pdf", "j"]))
        pd.reset_option("display.max_rows")

    def update_coordinates(self):
        for sentence in self.sentences:
            (page, top, left, bottom, right) = list(
                zip(
                    *[
                        self.coordinate_map[self.links[((sentence.stable_id), i)]]
                        for i in range(len(sentence.words))
                    ]
                )
            )
            sentence.page = list(page)
            sentence.top = list(top)
            sentence.left = list(left)
            sentence.bottom = list(bottom)
            sentence.right = list(right)
            yield sentence
        if self.verbose:
            self.logger.debug("Updated coordinates in database")
