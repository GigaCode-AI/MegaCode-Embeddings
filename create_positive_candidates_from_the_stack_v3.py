"""
Differences from v2:
* Removed logic related to is_hard. If needed, do it outside this script.
"""
import json
import glob
import re
import os
import argparse
import string
from multiprocessing import Process

import tqdm

spaces = string.whitespace
expression_words = re.compile("\w+")
parsed_langs = {
    "C",
    "C#",
    "C++",
    "Go",
    "Java",
    "JavaScript",
    "Julia",  # absent in the-stack-v2-smol-train
    "Kotlin",
    "Lua",
    "PHP",
    "Python",
    "R",
    "Ruby",
    "Rust",
    "Scala",  # absent in the-stack-v2-smol-train
    "TypeScript"
}


def clean_multi_line_comment(s: str, lang: str) -> str:
    """
    Strip comment markers from multi-line comments
    """
    if lang == "Python":
        return s.strip('"\'' + spaces)
    elif lang == "Julia":
        return s.strip('"' + spaces)
    elif lang == "Lua":
        # --[[
        # foo
        # --]]

        # --[[--
        # foo
        # ]]

        # --[[
        # foo
        # ]]

        # Rarely each line inside comments is also marked (as in .cpp); not handling that case separately
        return s.lstrip("-[" + spaces).rstrip("-]" + spaces)
    elif lang == "Ruby":
        if s.startswith("=begin") and s.endswith("=end"):
            return s[len("=begin"):-len("=end")].strip()
        return s
    elif lang in parsed_langs:
        res = []
        for line in s.splitlines():
            line = line.strip("/*" + spaces)
            if line:
                res.append(line)
        return "\n".join(res)
    else:
        raise NotImplementedError(lang)


def clean_single_line_comment(s: str, lang: str) -> str:
    """
    Strip comment markers from single-line comments.
    """
    if lang in ["Python", "Julia", "Ruby", "R"]:
        # # foo
        # #foo
        # #####
        # ##foo
        return s.lstrip("#" + spaces)
    elif lang == "Lua":
        # -- foo
        # --foo
        # ----foo
        # ---------
        return s.lstrip("-" + spaces)
    elif lang in parsed_langs:
        # // foo
        # //////
        return s.lstrip("/" + spaces)
    else:
        raise NotImplementedError(lang)


def clean_brief(s: str) -> str:
    for p in ["@brief", r"\brief", "!\n\\brief"]:
        if s.startswith(p):
            return s[len(p):].lstrip()
    return s


def group_single_line_comments(comments, text, min_group_size: int = 1):
    """
    Group consecutive single-line comments
    """
    if len(comments) == 0:
        return []
    if isinstance(text, str):
        text = text.encode()
    # Sanity check: only single-line comments are grouped
    assert not any(x["is_multi_line"] for x in comments)
    comments = sorted(comments, key=lambda x: x["start_byte"])
    group = [comments[0]]
    res = []

    def append():
        res.append({
            "id": group[0]["id"],  # Use the first comment's id as the group id
            "is_multi_line": False,  # To distinguish this synthetic group from original multi-line comments
            "start_byte": group[0]["start_byte"],
            "end_byte": group[-1]["end_byte"],
            "start_point": group[0]["start_point"],
            "end_point": group[-1]["end_point"]
        })

    def is_line_start(c):
        i = c["start_byte"]
        col = c["start_point"][1]
        if text[i-col:i].isspace():
            return True
        return False

    for i in range(1, len(comments)):
        # Comments are consecutive and start at line beginning
        curr = comments[i]
        prev = comments[i - 1]
        if is_line_start(curr) and is_line_start(prev) and (curr["start_point"][0] - prev["start_point"][0] == 1):
            group.append(curr)
        else:
            if len(group) >= min_group_size:
                append()
            group = [curr]
    if len(group) >= min_group_size:
        append()
    return res


def is_function(t: str, lang: str) -> bool:
    """
    Whether the snippet is a function. Determined by the node type returned by tree-sitter.
    """
    if lang == "R":
        return t == "binary_operator"
    return t.startswith("function") or t.startswith("method")


def comment_to_query(comment: str, identifier: str, lang: str) -> str:
    """
    Convert comment to query:
    * Extract semantically meaningful function description: remove input/output descriptions
    * Truncate by length

    comment - Single- or multi-line comment (single-line may be a group of consecutive ones).
    lang - Language

    Returns - Query string
    """
    # 2. Extract semantically meaningful content
    # Idea: walk comment lines until the first "bad" line
    if lang in ["C", "C++", "Java", "JavaScript", "Kotlin", "Lua", "PHP", "Ruby", "TypeScript"]:
        # TODO: C case with explicit "DESCRIPTION:"
        bad = [
            "return:", "returns:",
            "\param", r"\return",
            "-",  # likely argument description
        ]
        res = []
        # No break: description can be at the start or the very end
        for line in comment.splitlines():
            s = line.strip().lower()
            # @ = special words + any args, can't enumerate all; easier to special-case brief
            if s.startswith("-") and "example" in s:
                break
            if identifier in s:
                continue
            if any(s.startswith(p) for p in bad) or (s.startswith("@") and not (s.startswith("@brief" or s.startswith("@description")))):
                continue
            line_words = s.split()
            if (len(line_words) > 0) and line_words[0].endswith(":"):  # variables can be described this way too
                continue
            line = line.strip()
            if line:
                res.append(line)
        return "\n".join(res).strip()
    elif lang == "C#":
        if "<summary>" in comment and "</summary>" in comment:
            i = comment.index("<summary>")
            j = comment.index("</summary>")
            res = comment[i+len("<summary>"):j].strip()
        else:
            res = comment.strip()
        return res.strip()
    elif lang == "Rust":
        return comment.strip()
    elif lang == "R":
        # No clear template structure found here
        return comment.strip()
    elif lang == "Scala":
        return comment.strip()  # TODO: review
    else:
        if lang == "Python":
            bad = [
                "arguments", "args", "param", "example",
                ":param", ":return", ":type", ":rtype", ":raises",
                "todo:", "pylint:",
                "@param", "@return",
                "---", ">>>"
            ]
        elif lang == "Go":
            bad = [
                "@param", "@return",
                "- "  # argument description
            ]
        else:
            raise NotImplementedError(lang)
        res = []
        for line in comment.splitlines():
            s = line.strip().lower()
            if any(s.startswith(p) for p in bad):
                break
            line = line.strip()
            if line:
                res.append(line)
        return "\n".join(res).strip()


def get_pairs_file(x):
    """
    1. Split comments into single- and multi-line; group single-line ones.
       Final comment set: multi-line + grouped single-line.
    2. For each comment, decide if it's worth considering (likely describes a function).
       Rule: comment directly above a function or shortly after the signature is a likely description.
    3. If such a function is found, clean the comment and build a query (semantically meaningful part only).
    4. If something remains after cleaning and there's no word overlap with the function body, add the pair to candidates.
    """
    pairs = []
    text1 = x["content"].encode()
    text2 = x["content_wo_comments"].encode()
    id2snippet = {s["id"]: s for s in x["snippets_wo_comments"]}

    comments_multi = []
    comments_single = []
    for c in x["comments"]:
        if c["is_multi_line"]:
            comments_multi.append(c)
        else:
            comments_single.append(c)
    funcs = [s for s in x["snippets"] if is_function(s["type"], x["language"])]
    for c in comments_multi + group_single_line_comments(comments_single, text1, 1):
        for s in funcs:
            is_previous = s["start_point"][0] - c["end_point"][0] == 1
            is_inside = (s["start_byte"] <= c["start_byte"]) and (c["end_byte"] <= s["end_byte"])
            is_near_after = 0 <= c["start_point"][0] - s["start_point"][0] <= 3
            if is_previous or (is_inside and is_near_after):
                lang = x["language"]
                comment = text1[c["start_byte"]:c["end_byte"]].decode()

                # Strip comment markers
                if c["is_multi_line"]:
                    comment = clean_multi_line_comment(comment, lang)
                else:
                    res = []
                    for line in comment.splitlines():
                        res.append(clean_single_line_comment(line, lang))
                    comment = "\n".join(res)
                comment = clean_brief(comment)

                # Extract semantically meaningful part
                comment = comment_to_query(comment, s["identifier"], lang)

                if len(comment) == 0:
                    continue
                s1 = id2snippet[s["id"]]
                snippet = text2[s1["start_byte"]:s1["end_byte"]].decode()
                pairs.append({
                    "comment": comment,
                    "snippet": snippet,
                    "scope": s["scope"],
                    "comment_id": c["id"],
                    "snippet_id": s["id"],
                    "lang": lang,
                })
    return pairs


def get_pairs_repo(repo):
    """
    1. Build (comment, function) pairs where the function is from .h, .hpp
    2. Filter pairs: keep only those that can be uniquely found by name
    3. Look up the function by name in .c, .cpp files
    """
    pairs = []

    # Parse everything except C, C++; for C/C++ we split into headers and code
    files_c = []
    id2name = {}
    name2count = {}
    pairs_c = []
    for x in repo["files"]:
        # Only process languages that parsed successfully and have functions
        if x["language"] not in parsed_langs:
            continue

        ext = os.path.splitext(x["path"])[1]
        if ext in [".h", ".hpp"]:
            # Build (comment, function) pairs where the function is from .h, .hpp
            pairs_c += get_pairs_file(x)
            for s in x["snippets"]:
                if not is_function(s["type"], x["language"]):
                    continue
                name = s["identifier"]
                id2name[s["id"]] = name
                if name not in name2count:
                    name2count[name] = 0
                name2count[name] += 1
        else:
            pairs += get_pairs_file(x)
            if ext in [".c", ".cpp"]:
                # Also try to get pairs from .c, .cpp files; may not be many
                files_c.append(x)
    good_names = {name for name, count in name2count.items() if count == 1}

    # Index snippets without comments from .c, .cpp
    name2snippet = {}
    name2path = {}
    path2text = {}
    for x in files_c:
        path2text[x["path"]] = x["content_wo_comments"].encode()
        for s in x["snippets_wo_comments"]:
            if not is_function(s["type"], x["language"]):
                continue
            name = s["identifier"]
            if name in good_names:
                name2snippet[name] = s
                name2path[name] = x["path"]

    # Replace signature with function body text and reassign snippet id
    pairs_c_ok = []
    for p in pairs_c:
        name = id2name[p["snippet_id"]]  # Get function name
        if name not in name2snippet:  # Function is in headers but not in code
            continue
        s = name2snippet[name]  # Get function from .c without comments
        text = path2text[name2path[name]]  # Get .c file text without comments
        p["snippet"] = text[s["start_byte"]:s["end_byte"]].decode()  # Get function text from file without comments
        p["snippet_id"] = s["id"]  # Assign id from the .c file function
        pairs_c_ok.append(p)

    return pairs + pairs_c_ok


def main(args):
    os.makedirs(args.output_dir)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))

    def job_fn(job_id):
        pid = os.getpid()
        paths_job = [x for i, x in enumerate(paths) if i % args.n_jobs == job_id]
        n = 0
        for i, path in enumerate(paths_job):
            path_out = os.path.join(args.output_dir, os.path.basename(path))
            with open(path) as fin, open(path_out, "w") as fout:
                for line in tqdm.tqdm(fin, desc=f"[job {job_id}] [pid {pid}] read shard {i + 1} / {len(paths_job)}", position=job_id, leave=False):
                    repo = json.loads(line)
                    pairs = get_pairs_repo(repo)
                    for p in pairs:
                        fout.write(json.dumps(p) + "\n")
                    n += 1
                    if n == args.limit:
                        return

    jobs = []
    for i in range(args.n_jobs):
        job = Process(target=job_fn, args=(i,))
        job.start()
        jobs.append(job)
    for job in jobs:
        job.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--limit", type=int, default=-1)
    main(parser.parse_args())
