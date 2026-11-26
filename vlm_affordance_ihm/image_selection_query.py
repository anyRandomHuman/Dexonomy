"""Select goal images for each query in a text file using a vision model."""

import argparse
import base64
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from torch import float4_e2m1fn_x2

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def api_image_content(paths: list[Path]) -> list[dict]:
    content = []
    for index, path in enumerate(paths):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime_type = IMAGE_TYPES[path.suffix.lower()]
        content.extend([
            {"type": "input_text", "text": f"View {index}:"},
            {"type": "input_image",
             "image_url": f"data:{mime_type};base64,{encoded}",
             "detail": "high"},
        ])
    return content


def select_with_codex(prompt: str, paths: list[Path], model: str | None,
                      codex_binary: str) -> str:
    labels = "\n".join(f"{index}: {path.name}" for index, path in enumerate(paths))
    codex_prompt = (
        f"{prompt}\n\nAttached images in index order:\n{labels}\n"
        "Use the attached images to make the selection."
    )
    schema = {
        "type": "object",
        "properties": {
            "selected_indices": {"type": "array", "items": {"type": "integer"}},
            "reasoning": {"type": "string"},
        },
        "required": ["selected_indices", "reasoning"],
        "additionalProperties": False,
    }

    with tempfile.TemporaryDirectory() as temporary_dir:
        schema_path = Path(temporary_dir) / "selection_schema.json"
        response_path = Path(temporary_dir) / "selection_response.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")

        command = [
            codex_binary, "exec", "--ephemeral", "--sandbox", "read-only",
            "--skip-git-repo-check", "--output-schema", str(schema_path),
            "--output-last-message", str(response_path),
        ]
        if model:
            command.extend(["--model", model])
        for path in paths:
            command.extend(["--image", str(path.resolve())])
        # --image accepts multiple values, so terminate option parsing before
        # the positional prompt; otherwise Codex treats it as another image.
        command.extend(["--", codex_prompt])

        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        return response_path.read_text(encoding="utf-8")


def parse_selection_response(response_text: str) -> dict:
    """Find the final selection JSON after any free-form model reasoning."""
    text = response_text.strip()
    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            result, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and {"selected_indices", "reasoning"} <= result.keys():
            return result
    raise ValueError("response contains no selection JSON object")


def select_with_local_model(prompt: str, paths: list[Path], local_pipeline) -> str:
    content = [{"type": "text", "text": prompt}]
    for index, path in enumerate(paths):
        content.extend([
            {"type": "text", "text": f"View {index}: {path.name}"},
            {"type": "image", "image": str(path.resolve())},
        ])
    # Qwen3.5 starts an open <think> block by default. Its chat template
    # accepts enable_thinking directly through the Transformers pipeline.
    template_options = (
        {"enable_thinking": False}
        if local_pipeline.model.config.model_type == "qwen3_5" else {}
    )
    outputs = local_pipeline(
        text=[{"role": "user", "content": content}],
        max_new_tokens=4096,
        return_full_text=False,
        **template_options,
    )
    return outputs[0]["generated_text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True,
                        help="UTF-8 text file with one query per nonempty line")
    parser.add_argument("--images", type=Path, required=True,
                        help="Folder containing candidate goal images")
    parser.add_argument("--backend", choices=("api", "codex", "local_model"), default="api",
                        help="Use the OpenAI API, Codex CLI, or a local Transformers pipeline (default: api)")
    parser.add_argument("--model", help="Model override (API: gpt-4o; local: Qwen/Qwen3.5-9B)")
    parser.add_argument("--codex-bin", help="Path to Codex CLI when it is not on PATH")
    parser.add_argument("--api-key", "--api_key", dest="api_key",
                        help="OpenAI API key; defaults to OPENAI_API_KEY")
    args = parser.parse_args()

    queries = [line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not queries:
        parser.error("the queries file has no nonempty lines")

    paths = sorted((path for path in args.images.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_TYPES),
                   key=lambda path: path.name)
    if not paths:
        parser.error("the images folder has no supported images")

    model_name = args.model or {
        "api": "gpt-4o",
        "codex": "configured",
        "local_model": "Qwen/Qwen3.5-9B",
    }[args.backend]
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.replace("/", "__")).strip("._-")

    if args.backend == "api":
        from openai import OpenAI

        client = OpenAI(api_key=args.api_key) if args.api_key else OpenAI()
        image_content = api_image_content(paths)
    elif args.backend == "codex":
        if args.api_key:
            parser.error("--api-key applies only to --backend api")
        codex_binary = shutil.which(args.codex_bin or "codex")
        if codex_binary is None and args.codex_bin is None:
            codex_binary = shutil.which("/usr/lib/chatgpt/resources/codex")
        if codex_binary is None:
            parser.error("Codex CLI not found; pass --codex-bin /path/to/codex")
    else:
        if args.api_key:
            parser.error("--api-key applies only to --backend api")
        from transformers import pipeline

        local_pipeline = pipeline("image-text-to-text", model=model_name,
                                  device_map="auto", dtype="auto")

    width = max(3, len(str(len(queries))))
    for number, query in enumerate(queries, start=1):
        response_stem = (
            f"{args.queries.stem}_response_{number:0{width}d}_{model_tag}({args.backend})"
        )
        prompt = (
            f"Task: {query}\n"
            f"Choose the best view or ordered sequence of views from indices 0–{len(paths)-1}. "
            'Return only JSON with "selected_indices" (integer list) and '
            '"reasoning" (short string).'
        )
        if args.backend == "api":
            response = client.responses.create(
                model=model_name,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt}, *image_content,
                ]}],
            )
            raw_response = response.output_text
        elif args.backend == "codex":
            raw_response = select_with_codex(prompt, paths, args.model, codex_binary)
        else:
            raw_response = select_with_local_model(prompt, paths, local_pipeline)

        try:
            result = parse_selection_response(raw_response)
            indices = result["selected_indices"]
            if (not isinstance(indices, list) or
                    any(type(index) is not int or not 0 <= index < len(paths)
                        for index in indices)):
                raise ValueError(f"Invalid selected_indices: {indices!r}")
            if not isinstance(result["reasoning"], str):
                raise ValueError(f"Invalid reasoning: {result['reasoning']!r}")
        except ValueError as error:
            raw_path = args.queries.with_name(f"{response_stem}_raw.txt")
            raw_path.write_text(raw_response, encoding="utf-8")
            print(f"[{number}/{len(queries)}] Saved raw response ({error}): {raw_path}")
            continue

        output = {
            "query": query,
            "selected_indices": indices,
            "reasoning": result["reasoning"],
            "selected_images": [str(paths[index].resolve()) for index in indices],
        }
        output_path = args.queries.with_name(f"{response_stem}.json")
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"[{number}/{len(queries)}] {output_path}")


if __name__ == "__main__":
    main()
