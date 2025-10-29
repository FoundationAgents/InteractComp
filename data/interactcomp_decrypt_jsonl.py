import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional


def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def xor_decrypt(ciphertext_b64: str, password: str) -> str:
    if ciphertext_b64 is None:
        return ciphertext_b64
    if not ciphertext_b64:
        return ""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode("utf-8")


def parse_fields(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            json.dump(record, fh, ensure_ascii=True, separators=(",", ":"))
            fh.write("\n")


def decrypt_dataset(
    input_path: Path,
    output_path: Path,
    fields: List[str],
    canary_field: str,
    override_canary: Optional[str],
    drop_canary: bool,
) -> None:
    decrypted_records = []
    for record in iter_jsonl(input_path):
        token = override_canary if override_canary is not None else record.get(canary_field)
        if not token:
            raise ValueError(
                f"Missing canary token for record (id={record.get('id')}). "
                f"Set --canary-value if the column is absent."
            )

        for field in fields:
            if field not in record:
                continue
            value = record[field]
            if value is None:
                continue
            record[field] = xor_decrypt(value, token)
        if drop_canary:
            record.pop(canary_field, None)
        decrypted_records.append(record)

    write_jsonl(output_path, decrypted_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decrypt InteractComp JSONL encrypted with a shared canary token."
    )
    parser.add_argument("--input", required=True, help="Path to the encrypted JSONL.")
    parser.add_argument("--output", required=True, help="Path for the decrypted JSONL.")
    parser.add_argument(
        "--fields",
        default="domain,question,answer,context",
        help="Comma-separated fields to decrypt (default: domain,question,answer,context).",
    )
    parser.add_argument(
        "--canary-field",
        default="canary",
        help="Field name storing the canary token in the encrypted JSONL (default: canary).",
    )
    parser.add_argument(
        "--canary-value",
        help="Optional explicit canary token. Use when the encrypted file omits the canary column.",
    )
    parser.add_argument(
        "--keep-canary",
        action="store_true",
        help="Keep the canary field in the decrypted output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    fields = parse_fields(args.fields) if args.fields else []

    if not fields:
        raise ValueError("At least one field must be specified for decryption.")

    decrypt_dataset(
        input_path,
        output_path,
        fields,
        args.canary_field,
        args.canary_value,
        drop_canary=not args.keep_canary,
    )


if __name__ == "__main__":
    main()
