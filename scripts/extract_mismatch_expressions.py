import json
import re

INPUT_FILE = "expressions.json"
OUTPUT_FILE = "./mismatched_examples.json"

def normalize(s: str) -> str:
    # 알파벳/숫자만 남기고 모두 제거, 소문자화
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def extract_mismatched_examples(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    result = []

    for item in items:
        expr_norm = normalize(item["expression"])

        for ex in item.get("examples", []):
            text_norm = normalize(ex["text"])

            # 정규화한 expression이 text에 포함되지 않으면 불일치로 간주
            if expr_norm not in text_norm:
                result.append({
                    "expression": item["expression"],
                    "text": ex["text"],
                    "translation": ex.get("translation"),
                    "unit": item.get("unit"),
                    "chapter": item.get("chapter"),
                    "source_chapter": item.get("source_chapter"),
                    "source_section": item.get("source_section")
                })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"추출 완료: {len(result)}개의 예문을 '{output_path}'에 저장했습니다.")

extract_mismatched_examples(INPUT_FILE, OUTPUT_FILE)