def render_argument(arg, indent=0):
    """Render an Argument object (and its children) into RST format."""
    ind = "    " * indent
    out = []

    if isinstance(arg, str):
        return f"{ind}{arg}"

    out.append(f"{ind}{arg.name}:")
    out.append(f"{ind}    | type: ``{arg.dtype}``")

    if getattr(arg, "optional", False):
        out.append(f"{ind}    | optional: True")

    if getattr(arg, "default", None) is not None:
        out.append(f"{ind}    | default: ``{arg.default}``")

    if getattr(arg, "doc", ""):
        doc_lines = arg.doc.strip().splitlines()
        for line in doc_lines:
            out.append(f"{ind}    {line.strip()}")

    # 如果有子字段
    for sub in getattr(arg, "sub_fields", []):
        out.append("")  # 空行分隔
        out.append(render_argument(sub, indent + 1))

    # 如果有子变体
    for var in getattr(arg, "sub_variants", []):
        out.append("")
        out.append(f"{ind}    Variant:")
        out.append(render_argument(var, indent + 2))

    return "\n".join(out)


import os
from dpnegf.utils import argcheck

def generate_rst_from_argcheck(output_dir="docs/input_params"):
    os.makedirs(output_dir, exist_ok=True)

    # 这里你可以写死，也可以动态扫描
    modules = {
        "common_options": argcheck.common_options,
        "run_options": argcheck.run_options,
    }

    for name, func in modules.items():
        # arg = func() if callable(func) else func
        # print(f"[DEBUG] {name} returned:", arg, type(arg))
        arg = func()
        rst = f"""
========================================
{name.replace("_", " ").title()}
========================================

.. _`{name}`:

{render_argument(arg)}
"""
        with open(os.path.join(output_dir, f"{name}.rst"), "w", encoding="utf-8") as f:
            f.write(rst)

def main():
    generate_rst_from_argcheck()

if __name__ == "__main__":
    main()
