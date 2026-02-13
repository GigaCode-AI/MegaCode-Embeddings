"""
* Merge of split_source_code_by_subtrees.py and parse_comments.py
* The Stack format
* Parsing logic is derived from language: .h can be C or C++; language is already in sources
* Added identifier to objects
* Added ids to comments and snippets
* Store only offsets; storing full text would be too heavy
"""
import json
import glob
import os
import argparse
import uuid
from typing import List, Tuple, Dict
from multiprocessing import Process

import tqdm
from tree_sitter import Language, Parser, Node

# import grammars
import tree_sitter_python
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_c_sharp
import tree_sitter_css
import tree_sitter_go
# import tree_sitter_haskell
import tree_sitter_java
import tree_sitter_javascript
# import tree_sitter_kotlin
import tree_sitter_rust
# import tree_sitter_zig
import tree_sitter_julia
import tree_sitter_lua
import tree_sitter_ruby
# import tree_sitter_ocaml
import tree_sitter_php
import tree_sitter_scala
import tree_sitter_typescript
# import tree_sitter_verilog

import tree_sitter_r


def is_comment_py(n: Node) -> Tuple[bool, bool]:
    """
    \"\"\"
    docstring
    \"\"\"

    # comment

    \"string\"
    """
    # if n.type == "expression_statement" and len(n.children) == 1 and n.children[0].type == "string":
    #     x = n.children[0]
    if n.type == "comment":
        return True, False
    elif n.type == "string" and n.parent.type == "expression_statement" and n.parent.child_count == 1:
        return True, True
    return False, False

def is_comment_cpp(n: Node) -> Tuple[bool, bool]:
    """
    Single-line: "// foo"
    Multi-line: "/* foo */"
    """
    if n.type == "comment":
        if n.text.lstrip().startswith(b"/*"):
            return True, True
        return True, False
    return False, False

def is_comment_java(n: Node) -> Tuple[bool, bool]:
    """
    line_comment: "// foo"
    block_comment: "/* foo */" (can be multi-line)
    """
    if n.type == "line_comment":
        return True, False
    elif n.type == "block_comment":
        return True, True
    return False, False

def is_comment_scala(n: Node) -> Tuple[bool, bool]:
    """
    line_comment: "// foo"
    block_comment: "/* foo */" (can be multi-line)
    """
    if n.type == "comment":
        return True, False
    elif n.type == "block_comment":
        return True, True
    return False, False

def is_comment_julia(n: Node) -> Tuple[bool, bool]:
    """
    line_comment: "# foo"
    string_literal: '" foo "' (can be multi-line)
    """
    if n.type == "line_comment":
        return True, False
    elif n.type == "string_literal":
        return True, True
    return False, False

def is_comment_lua(n: Node) -> Tuple[bool, bool]:
    """
    Single-line: "-- foo"
    Multi-line: "--[[ foo --]]"
    """
    if n.type == "comment":
        if n.text.lstrip().startswith(b"--[["):
            return True, True
        return True, False
    return False, False

def is_comment_ruby(n: Node) -> Tuple[bool, bool]:
    if n.type == "comment":
        if n.text.lstrip().startswith(b"=begin"):
            return True, True
        return True, False
    return False, False

def is_comment_r(n: Node) -> Tuple[bool, bool]:
    if n.type == "comment":
        return True, False
    # elif n.type == "string":  # rarely used for comments; usually just strings
    #     return True, True
    return False, False

def search_identifier(node, lang):
    """
    Algorithm idea:
    BFS to depth 2 looking for identifier.
    If not found, take the first node whose type ends with identifier.

    There are tricky cases, e.g. in C++:
    virtual n0::CompID TypeID() const override {
		return n0::GetCompTypeID<CompAABB>();
	}
    Here 8 nodes end with identifier, and the first one is not the one we need.
    """
    # Object name; position depends on language and object type, but it always exists and is "identifier".
    # if lang == "C" and node.type == "struct_specifier":
    #     id_type = "type_identifier"  # via ?
    # elif lang == "C++" and node.type == "namespace_definition":
    #     id_type = "namespace_identifier"
    # elif lang == "C++" and node.type == "function_definition":
    #     id_type = "type_identifier"  # can also be identifier when outside class
    # elif lang == "C++" and node.type == "class_specifier":
    #     id_type = "type_identifier"
    # elif lang == "Go" and node.type == "type_declaration":
    #     id_type = "type_identifier"  # via type_spec
    # elif lang == "JavaScript" and node.type == "method_definition":
    #     id_type = "property_identifier"
    # elif lang == "PHP":
    #     id_type = "name"
    # elif lang == "Ruby" and node.type == "class":
    #     id_type = "constant"
    # elif lang == "Rust" and node.type == "struct_item":
    #     id_type = "type_identifier"
    # elif lang == "TypeScript" and node.type == "method_definition":
    #     id_type = "property_identifier"
    # elif lang == "TypeScript" and node.type == "class_declaration":
    #     id_type = "type_identifier"
    # else:
    #     id_type = "identifier"

    def is_identifier(t):
        if lang == "PHP":
            return t == "name"
        elif lang == "Ruby" and node.type == "class":
            return t == "constant"
        else:
            return t == "identifier"

    nodes = list(node.children)
    depth = 0
    curr = len(nodes)
    first = -1
    i = 0
    for n in nodes:
        if is_identifier(n.type):
            return n.text.decode()
        elif n.type.endswith("identifier") and first < 0:
            first = i
        nodes += n.children
        i += 1
        if i == curr:
            depth += 1
            curr = len(nodes)
            # C examples where identifier is at depth 3:
            # int* selecao (int *vetorInicial, int *tamanho, int *vetorOrdenado){}
            # static inline void* pred_blkp(void *bp){}
            if depth == 5:  # Capped at a small number; 2 was too low, see examples above
                break
    if first >= 0:
        return nodes[first].text.decode()

    # if lang == "C" and node.type == "struct_specifier":
    #     # struct {
    #     #         int regNo;
    #     #         int pSolved;
    #     #         double cgpa;
    #     # }
    #     # https://stackoverflow.com/questions/3628938/a-structure-without-a-structure-name
    #     return ""
    # if lang == "C++" and node.type == "namespace_definition":
    #     # namespace {
    #     # /// Include the patterns defined in the Declarative Rewrite framework.
    #     # #include "ToyCombine.inc"
    #     # }
    #     return ""
    # if lang == "Rust" and node.type == "impl_item":
    #     # impl ! {}
    #     return ""
    #
    # msg = ""
    # msg += f"[LANG] {lang}\n"
    # msg += f"[TYPE] {node.type}\n"
    # msg += f"[NODE] {node.text.decode()}\n"
    # msg += f"[CHILDREN]\n"
    # for x in node.children:
    #     msg += f"[TYPE] {x.type} [TEXT] {x.text}\n"
    # raise Exception(msg)
    return ""

def parse_object(node, text_enc, res, lang, scope, types, parse_scope):
    children = node.children

    if lang == "R":
        # In R a function is defined via assignment; using only function_definition would lose its name.
        # On the other hand, assignment defines many things (e.g. plain variables).
        if not (node.type == "binary_operator" and children[-1].type == "function_definition"):
            return scope
    elif lang == "Ruby" and node.type == "class":
        # In Ruby both the class and the "class" keyword are nodes with type "class"
        if len(node.children) == 0:
            return scope
    else:
        if node.type not in types:
            return scope

    assert text_enc[node.start_byte:node.end_byte] == node.text
    res.append({
        "id": str(uuid.uuid4()),
        # "text": node.text.decode(),
        "scope": scope,
        "type": node.type,
        "identifier": search_identifier(node, lang),
        "start_byte": node.start_byte,
        "end_byte": node.end_byte,
        "start_point": list(node.start_point),  # (row, col) tuple
        "end_point": list(node.end_point)  # (row, col) tuple
    })

    # parse scope prefix
    if parse_scope:
        # In Python all children except the last correspond to the signature; same in other langs (spot-checked).
        # Checked each language explicitly: OK, with a couple of exceptions
        n0 = children[0]
        if lang == "Ruby":
            n1 = children[1]
        else:
            n1 = children[-2]
        try:
            s = text_enc[n0.start_byte - n0.start_point[1]:n1.end_byte + 1].decode()
        except UnicodeDecodeError:  # rare
            s = ""
        return scope + s
    return scope

# Swift is also in the top languages but no Python bindings found.
# .h is the only extension shared by multiple languages: C and C++.
langs_map = {
    "C": {
        "lang": Language(tree_sitter_c.language()),
        "ext": [".c", ".h"],
        "types": ["struct_specifier", "function_definition"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp
    },  # same as go
    "C#": {
        "lang": Language(tree_sitter_c_sharp.language()),
        "ext": [".cs"],
        "types": ["class_declaration", "method_declaration", "namespace_declaration"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp
    },  # same as go
    "C++": {
        "lang": Language(tree_sitter_cpp.language()),
        "ext": [".cpp", ".h", ".hpp"],
        "types": ["class_specifier", "function_definition", "namespace_definition"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp,
    },
    "CSS": {
        "lang": Language(tree_sitter_css.language()),
        "ext": [".css"],
        "types": ["rule_set"],
        "parse_scope": False,
        "is_comment_fn": is_comment_cpp
    },
    "Go": {
        "lang": Language(tree_sitter_go.language()),
        "ext": [".go"],
        "types": ["function_declaration", "type_declaration"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp
    },  # Ideally also parse groups of single-line "//" comments; objects are often documented that way
    "Java": {
        "lang": Language(tree_sitter_java.language()),
        "ext": [".java"],
        "types": ["class_declaration", "method_declaration"],
        "parse_scope": True,
        "is_comment_fn": is_comment_java
    },  # same as go
    "JavaScript": {
        "lang": Language(tree_sitter_javascript.language()),
        "ext": [".js", ".jsx"],
        "types": ["class_declaration", "method_definition", "function_declaration"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp
    },  # same as go
    "Julia": {
        "lang": Language(tree_sitter_julia.language()),
        "ext": [".jl"],
        "types": ["struct_definition", "function_definition"],
        "parse_scope": True,
        "is_comment_fn": is_comment_julia
    },
    # Tried this, version 1.1.0
    # https://github.com/tree-sitter-grammars/tree-sitter-kotlin
    # It's buggy: goes into an infinite loop on bad inputs, e.g. b"interface foo\n  @bar"
    # Worst part: the hang can only be stopped with kill -9 {pid}
    # So it's unclear how to use it safely.
    # Unfortunately, out of millions of Kotlin files at least 3 trigger this. If you really need Kotlin:
    # 1. Run parsing on all data with Kotlin enabled
    # 2. Manually kill hung processes (e.g. 3 out of 64)
    # 3. Disable Kotlin
    # 4. Re-parse what didn't finish (shards that didn't start + those that hung)
    # "Kotlin": {
    #     "lang": Language(tree_sitter_kotlin.language()),
    #     "ext": [".kt", ".kts"],
    #     "types": ["class_declaration", "function_declaration"],
    #     "parse_scope": True,
    #     "is_comment_fn": is_comment_java
    # },
    "Lua": {
        "lang": Language(tree_sitter_lua.language()),
        "ext": [".lua"],
        "types": ["function_declaration"],
        "parse_scope": False,
        "is_comment_fn": is_comment_lua
    },
    "PHP": {
        "lang": Language(tree_sitter_php.language_php()),
        "ext": [".php"],
        "types": ["function_definition", "class_declaration", "method_declaration"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp
    },
    "Python": {
        "lang": Language(tree_sitter_python.language()),
        "ext": [".py"],
        "types": ["function_definition", "class_definition"],
        "parse_scope": True,
        "is_comment_fn": is_comment_py
    },
    "R": {
        "lang": Language(tree_sitter_r.language()),
        "ext": [".R", ".r"],
        # Classes are rare here and are a hassle.
        # WARNING: functions actually appear under "binary_operator", see comments in parse_object
        "types": ["function_definition"],
        "parse_scope": False,
        "is_comment_fn": is_comment_r
    },
    "Ruby": {
        "lang": Language(tree_sitter_ruby.language()),
        "ext": [".rb"],
        "types": ["class", "method"],  # had module too but removed
        "parse_scope": True,
        "is_comment_fn": is_comment_ruby,
    },
    "Rust": {
        "lang": Language(tree_sitter_rust.language()),
        "ext": [".rs"],
        "types": ["struct_item", "function_item", "impl_item"],
        "parse_scope": True,
        "is_comment_fn": is_comment_java
    },
    "Scala": {
        "lang": Language(tree_sitter_scala.language()),
        "ext": [".scala"],
        "types": ["class_definition", "function_definition"],
        "parse_scope": True,
        "is_comment_fn": is_comment_scala
    },
    "TypeScript": {
        "lang": Language(tree_sitter_typescript.language_typescript()),
        "ext": [".ts", ".tsx"],
        "types": ["function_declaration", "class_declaration", "method_definition"],
        "parse_scope": True,
        "is_comment_fn": is_comment_cpp
    },  # same as go
}


def parse(text: str, lang: str) -> Tuple[List[Dict], List[Dict]]:
    # setup parser
    parser = Parser()
    parser.language = langs_map[lang]["lang"]

    # parse
    text_enc = text.encode()
    tree = parser.parse(text_enc)

    # traverse
    comment_fn = langs_map[lang]["is_comment_fn"]
    objects = []
    comments = []
    visited = set()

    def traverse(node, scope):
        # Compared this traversal with the authors' non-recursive one:
        # https://github.com/tree-sitter/py-tree-sitter/blob/master/examples/walk_tree.py
        # Used recursion because it's convenient for passing scope.
        if node.id in visited:
            # Haven't seen cases that hit this (checked), but keep it for safety
            return
        visited.add(node.id)
        is_comment, is_multi_line = comment_fn(node)
        if is_comment:
            assert text_enc[node.start_byte:node.end_byte] == node.text
            comments.append({
                "id": str(uuid.uuid4()),
                # "text": node.text.decode(),
                "is_multi_line": is_multi_line,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
                "start_point": list(node.start_point),
                "end_point": list(node.end_point)
            })
            return
        scope = parse_object(
            node, text_enc, objects, lang, scope,
            types=langs_map[lang]["types"], parse_scope=langs_map[lang]["parse_scope"]
        )
        for x in node.children:
            traverse(x, scope)

    traverse(tree.root_node, "")
    return objects, comments


def main(args):
    """
    Input format: see parse_the_stack_v2.py
    """
    # create output dir
    os.makedirs(args.output_dir)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))

    def job_fn(job_id):
        pid = os.getpid()  # To restore job->process mapping
        paths_job = [path for i, path in enumerate(paths) if i % args.n_jobs == job_id]
        done = 0
        for i, path_in in enumerate(paths_job):
            path_out = os.path.join(args.output_dir, os.path.basename(path_in))
            with open(path_in) as fin, open(path_out, "w") as fout:
                for line in tqdm.tqdm(fin, desc=f"[job {job_id}] [pid {pid}] shard {i + 1} / {len(paths_job)}", position=job_id, leave=False):
                    repo = json.loads(line)
                    for d in repo["files"]:
                        lang = d["language"]
                        ext = os.path.splitext(d["path"])[1]
                        d["snippets"] = []
                        d["comments"] = []
                        if (lang in langs_map.keys()) and (ext in langs_map[lang]["ext"]):
                            if d["length_bytes"] > 1_000_000:
                                # count     1000000
                                # mean         3505
                                # std         23995
                                # min             8
                                # 25%           521
                                # 50%          1241
                                # 75%          3016
                                # 90%          6908
                                # 95%         11579
                                # 99%         32967
                                # 99.9%      129352
                                # 99.99%     784707
                                # max       6262747

                                # There is a repo eirikvaa/TPG4850-VR-Gruppe-1 with many C files,
                                # each several MB, with almost every line as an object; parsed it's ~1TB on disk.
                                d["status"] = "too_long_file"
                            elif d["num_lines"] > 10_000:
                                # count     1000000
                                # mean          109
                                # std           525
                                # min             1
                                # 25%            22
                                # 50%            47
                                # 75%           101
                                # 90%           214
                                # 95%           348
                                # 99%           951
                                # 99.9%        3516
                                # 99.99%      19160
                                # max         84559
                                d["status"] = "too_many_rows"
                            elif d["max_line_length"] > 10_000:
                                # count     1000000
                                # mean          101
                                # std           355
                                # min             3
                                # 25%            64
                                # 50%            85
                                # 75%           116
                                # 90%           155
                                # 95%           195
                                # 99%           357
                                # 99.9%         824
                                # 99.99%       7370
                                # max        192136
                                d["status"] = "too_long_line"
                            elif d["src_encoding"].lower() != "utf-8":
                                d["status"] = "bad_encoding"
                            else:
                                try:
                                    d["snippets"], d["comments"] = parse(d["content"], d["language"])
                                    d["status"] = "success"
                                except Exception as e:
                                    # print(e)  # usually RecursionError in traverse
                                    # print("[PROJECT]", repo["repo_name"])
                                    # print("[FILE]", d["path"])
                                    # print(d)
                                    d["status"] = type(e).__name__
                                    # raise
                        else:
                            d["status"] = "not_implemented"
                    fout.write(json.dumps(repo) + "\n")
                    done += 1
                    if done == args.limit:
                        return

    jobs = []
    for j in range(args.n_jobs):
        job = Process(target=job_fn, args=(j,))
        jobs.append(job)
        job.start()
    for job in jobs:
        job.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="directory with .json files, each file is github shard")
    parser.add_argument("--output_dir", help="new dir with .json files with same format")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1, help="number of repos")
    main(parser.parse_args())
