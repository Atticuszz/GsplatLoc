"""
Pandoc filter using panflute: scientific 2column mode fixes
This filter does the following:
    1) fixes longtables -> tabular

It's intended use is for scientific papers that require 2 columns in the template layotu.
"""

import re
import sys

import pandas as pd
from panflute import *


def process_image(elem: Image, doc):
    # 获取图片信息
    src = elem.url.replace("\\", "/")  # 替换反斜杠为正斜杠
    src = src.lstrip("./")  # 移除开头的 './'
    title = elem.title or "Untitled"
    print(f"Image's title:{title}", file=sys.stderr)
    caption = get_text(elem.content) or title
    print(f" caption ={caption}", file=sys.stderr)
    # 检查是否是跨列图片
    is_cross_column = "cross-column" in elem.classes
    width = next(
        (
            attr.split("=")[1]
            for attr in elem.attributes
            if attr.startswith("width=")
        ),
        "0.8\\textwidth",
    )
    print(f"width:{width}", file=sys.stderr)
    # 生成 LaTeX 代码
    # if is_cross_column:
    try:

        latex = (
            "\\begin{figure*}[htbp]\n"
            + "    \\centering\n"
            + "    \\includegraphics[width=" + width + "]{" + src + "}\n"
            + "    \\caption{" + caption + "}\n"
            + "    \\label{fig:" + (elem.identifier or 'unnamed') + "}\n"
            + "\\end{figure*}"
        )
        print(f"image:{latex}", file=sys.stderr)
        #     else:
        #         latex = f"""
        # \\begin{{figure}}[htbp]
        #     \\centering
        #     \\includegraphics[width={width}]{{{src}}}
        #     \\caption{{{caption}}}
        #     \\label{{fig:{elem.identifier or 'unnamed'}}}
        # \\end{{figure}}
        # """
        return RawInline(latex, format="latex")
    except Exception as e:
        print(f"Error processing Image: {str(e)}", file=sys.stderr)
        return elem


def color_cell(value, rank):
    colors = {
        1: "\\cellcolor{green!30}\\textbf{",
        2: "\\cellcolor{yellow!30}",
        3: "\\cellcolor{lime!50}",
    }
    # bold no.1
    color_start = colors.get(rank, "")
    color_end = "}" if rank == 1 else ""
    return f"{color_start}{value:.3f}{color_end}"


def process_table_data(data, is_lower_better):
    df = pd.DataFrame(data)

    for col in df.columns[1:]:  # 跳过第一列（方法名）
        col_data = pd.to_numeric(df[col], errors="coerce")
        if is_lower_better:
            ranks = col_data.rank(method="min")
        else:
            ranks = col_data.rank(method="min", ascending=False)
        df[col] = [
            color_cell(float(val), rank) if pd.notnull(val) else val
            for val, rank in zip(col_data, ranks)
        ]
    return df.astype(str)


def get_text(item):
    if isinstance(item, (Str, Math, RawInline)):
        return item.text
    elif isinstance(item, Space):
        return " "
    # elif isinstance(item, ListContainer):
    #     # return "".join([get_text(i) for i in item])
    #     return stringify(item)
    else:
        return stringify(item)


def replace_longtables_with_tabular(elem, doc):
    try:

        def tabular():
            # 使用 l 作为第一列，其余列使用 c（居中对齐）
            return "\\begin{tabular}{l" + "c" * (elem.cols - 1) + "}\n\\toprule\n"

        def headers():
            if elem.head and elem.head.content:
                return (
                    " & ".join(
                        [
                            "\\textbf{" + get_text(cell) + "}"
                            for cell in elem.head.content[0].content
                        ]
                    )
                    + "\\\\\n\\midrule\n"
                )
            return ""

        def items():
            # collect data
            data = []
            for body in elem.content:
                for row in body.content:
                    data.append([get_text(cell) for cell in row.content])
            # color data
            is_lower_better = "↓" in (
                get_text(elem.caption.content[0])
                if elem.caption and elem.caption.content
                else ""
            )
            df = process_table_data(data, is_lower_better)
            # format data as latex table
            rows = []
            for index, row in df.iterrows():
                cells = list(row)

                if index == len(df) - 1:
                    cells[0] = "\\textbf{" + cells[0] + "}"
                rows.append(" & ".join(cells) + "\\\\")
                if index == len(df) - 2 and len(df) > 1:  # 确保至少有两行
                    rows.append("\\midrule")  # 在最后一行之前添加分割线
            return "\n".join(rows) + "\n"

        def caption():
            if elem.caption and elem.caption.content:
                caption_text = get_text(elem.caption.content)
                # set the first sentence to bold
                first_sentence, _, rest = caption_text.partition(".")
                if rest:
                    caption_text = f"\\textbf{{{first_sentence}}}." + rest
                else:
                    caption_text = f"\\textbf{{{caption_text}}}"

                label = "table:" + re.sub(r"\W+", "_", caption_text.lower())[:20]
                return "\\caption{" + caption_text + "}\n" + "\\label{" + label + "}\n"
            return ""

        def table_format():
            """Make sure that the table number (e.g. "Table 1:") is also bold in the generated LaTeX document"""
            return "\\renewcommand{\\thetable}{\\textbf{\\arabic{table}}}\n\\renewcommand{\\tablename}{\\textbf{Table}}\n"

        result = (
            "\\begin{table}[htbp]\n"
            + table_format()
            + "\\centering\n"
            + caption()  # 将 caption 移到这里
            + "\\begin{adjustbox}{max width=\\columnwidth,max height=!,center}\n"
            + tabular()
            + headers()
            + items()
            + "\\bottomrule\n\\end{tabular}\n"
            + "\\end{adjustbox}\n"
            + "\\end{table}"
        )

        print("Table processed successfully", file=sys.stderr)
        return RawBlock(result, "latex")
    except Exception as e:
        print(f"Error processing table: {str(e)}", file=sys.stderr)
        return elem


def prepare(doc):
    pass


def action(elem, doc):
    if doc.format != "latex":
        return None
    if isinstance(elem, Table):
        print("Table found!", file=sys.stderr)
        return replace_longtables_with_tabular(elem, doc)

    # if isinstance(elem, Image):
    #     print("Image found!", file=sys.stderr)
    #     return process_image(elem, doc)
    return None


def finalize(doc):
    pass


def main(doc=None):
    try:
        return run_filter(action, prepare=prepare, finalize=finalize, doc=doc)
    except Exception as e:
        print(f"Error processing run_filter: {str(e)}", file=sys.stderr)
        return elem

if __name__ == "__main__":
    main()
