import fitz
import json
from collections import defaultdict


def map_text_spans_to_pdf(pdf_path, pii_entities, pii_types):
    doc = fitz.open(pdf_path)
    results = []
    for page_num, page in enumerate(doc):
        for ent, ent_type in zip(pii_entities, pii_types):
            matches = page.search_for(ent)
            for m in matches:
                results.append({
                    "page": page_num,
                    "text": ent,
                    "type": ent_type,
                    "bbox": [m.x0, m.y0, m.x1, m.y1]
                })
    doc.close()
    return results


def get_replacement_text(pii_type, occurrence_count):
    """Return the appropriate replacement text based on PII type with numbering"""
    base_replacements = {
        "Name": "[Name",
        "Address": "[Address",
        "Birthday": "[Birthday",
        "Email": "[Email",
        "Phone": "[Phone",
        "DEFAULT": "[REDACTED"
    }

    base_text = base_replacements.get(pii_type, base_replacements["DEFAULT"])
    return f"{base_text}{occurrence_count}]"


def redact_pdf(pdf_in, pdf_out, pii_spans, pii_policies, log_file):
    doc = fitz.open(pdf_in)
    logs = []

    # Create a mapping of unique (type, text) pairs to occurrence numbers
    type_text_mapping = {}
    type_counters = defaultdict(int)

    # First, identify all unique type+text combinations and assign numbers
    unique_combinations = set()
    for span in pii_spans:
        key = (span["type"], span["text"])
        unique_combinations.add(key)

    # Sort for consistent numbering
    sorted_combinations = sorted(unique_combinations, key=lambda x: (x[0], x[1]))

    # Assign occurrence numbers
    for type_name, text in sorted_combinations:
        type_counters[type_name] += 1
        type_text_mapping[(type_name, text)] = type_counters[type_name]

    # Now process each span with the correct occurrence number
    for span in pii_spans:
        page = doc[span["page"]]
        bbox = fitz.Rect(span["bbox"])
        action = pii_policies.get(span["type"], "redact")

        # Get the correct occurrence number for this type+text combination
        key = (span["type"], span["text"])
        occurrence_number = type_text_mapping.get(key, 1)
        replacement = get_replacement_text(span["type"], occurrence_number)

        if action == "pseudonymize":
            page.add_redact_annot(bbox, text=replacement)
        else:
            page.add_redact_annot(bbox, text=replacement)

        logs.append({
            "page": span["page"] + 1,
            "type": span["type"],
            "occurrence": occurrence_number,
            "original": span["text"],
            "replacement": replacement,
            "bbox": span["bbox"]
        })

    # Apply all redactions
    doc.save(pdf_out, garbage=4, deflate=True, clean=True)
    doc.close()

    with open(log_file, "w") as f:
        json.dump({"redactions": logs}, f, indent=2)

    # Print debug information
    print("=== DEBUG: OCCURRENCE NUMBERING ===")
    for (type_name, text), occ_num in type_text_mapping.items():
        print(f"{type_name}{occ_num}: '{text}'")
    print("===================================")

    print(f"Redaction completed. Log saved to {log_file}")


# Example usage
f = open("found_img//pii_text.json", "r")
data = json.load(f)
print(data)

q = map_text_spans_to_pdf("found_img//text_pdf.pdf", data['pii_entities'], data['pii_types'])
print(f"Found {len(q)} PII spans")

print(q)

poli = {
    "Name": "pseudonymize",
    "Address": "pseudonymize",
    "Birthday": "pseudonymize",
    "Email": "pseudonymize",
    "Phone": "pseudonymize",
}

redact_pdf("found_img//text_pdf.pdf", "testout.pdf", q, poli, "log.json")