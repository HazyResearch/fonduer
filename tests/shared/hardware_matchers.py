"""Hardware matchers."""
import csv

from fonduer.candidates.matchers import (
    DictionaryMatch,
    Intersect,
    LambdaFunctionMatcher,
    RegexMatchSpan,
    Union,
)
from fonduer.utils.data_model_utils import get_row_ngrams, overlap

temp_matcher = RegexMatchSpan(rgx=r"(?:[1][5-9]|20)[05]", longest_match_only=False)

# Transistor Naming Conventions as Regular Expressions ###
eeca_rgx = (
    r"([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}"
    r"[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)"
)
jedec_rgx = r"(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)"
jis_rgx = r"(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})"
others_rgx = (
    r"((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|"
    r"TIPL|DTC|MMBT|SMMBT|PZT|FZT|STD|BUV|PBSS|KSC|CXT|FCX|CMPT){1}"
    r"[\d]{2,4}[A-Z]{0,5}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)"
)

part_rgx = "|".join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=True)


def get_digikey_parts_set(path):
    """Get all transistor parts from digikey part dictionary."""
    all_parts = set()
    with open(path, "r") as csvinput:
        reader = csv.reader(csvinput)
        for line in reader:
            (part, url) = line
            all_parts.add(part)
    return all_parts


# Dictionary of known transistor parts ###
dict_path = "tests/data/digikey_part_dictionary.csv"
part_dict_matcher = DictionaryMatch(d=get_digikey_parts_set(dict_path))


def common_prefix_length_diff(str1, str2):
    """Calculate common prefix length difference."""
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            return min(len(str1), len(str2)) - i
    return 0


def part_file_name_conditions(attr):
    """Check part file name conditions."""
    file_name = attr.sentence.document.name
    if len(file_name.split("_")) != 2:
        return False
    if attr.get_span()[0] == "-":
        return False
    name = attr.get_span().replace("-", "")
    return (
        any(char.isdigit() for char in name)
        and any(char.isalpha() for char in name)
        and common_prefix_length_diff(file_name.split("_")[1], name) <= 2
    )


add_rgx = r"^[A-Z0-9\-]{5,15}$"

part_file_name_lambda_matcher = LambdaFunctionMatcher(func=part_file_name_conditions)
part_file_name_matcher = Intersect(
    RegexMatchSpan(rgx=add_rgx, longest_match_only=True), part_file_name_lambda_matcher
)

part_matcher = Union(part_rgx_matcher, part_dict_matcher, part_file_name_matcher)

# CE Voltage Matcher
ce_keywords = set(["collector emitter", "collector-emitter", "collector - emitter"])
ce_abbrevs = set(["ceo", "vceo"])
ce_v_max_rgx_matcher = RegexMatchSpan(rgx=r"\d{1,2}[05]", longest_match_only=False)


def ce_v_max_conditions(attr):
    """Check ce_v_max conditions."""
    return overlap(
        ce_keywords.union(ce_abbrevs), get_row_ngrams(attr, spread=[0, 3], n_max=3)
    )


ce_v_max_row_matcher = LambdaFunctionMatcher(func=ce_v_max_conditions)


def ce_v_max_more_conditions1(attr):
    """Check ce_v_max conditions."""
    text = attr.sentence.text
    if (
        attr.char_start > 1
        and text[attr.char_start - 1] == "-"
        and text[attr.char_start - 2] not in [" ", "="]
    ):
        return False
    return True


def ce_v_max_more_conditions(attr):
    """Check ce_v_max conditions."""
    text = attr.sentence.text
    if attr.char_start != 0 and text[attr.char_start - 1] == "/":
        return False
    if (
        attr.char_start > 1
        and text[attr.char_start - 1] == "-"
        and text[attr.char_start - 2] not in [" ", "="]
    ):
        return False
    if "vcb" in attr.sentence.text.lower():
        return False
    for i in range(attr.char_end + 1, len(text)):
        if text[i] == " ":
            continue
        if text[i].isdigit():
            break
        if text[i].upper() != "V":
            return False
        else:
            break
    return True


def attr_in_table(attr):
    """Check attribute is in table."""
    return attr.sentence.is_tabular()


attr_in_table_matcher = LambdaFunctionMatcher(func=attr_in_table)

ce_v_whole_number = LambdaFunctionMatcher(func=ce_v_max_more_conditions)

volt_matcher = Intersect(
    ce_v_max_rgx_matcher, attr_in_table_matcher, ce_v_max_row_matcher, ce_v_whole_number
)
