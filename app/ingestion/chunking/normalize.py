import unicodedata

# Visually identical text can have different Unicode representations. 
# NFC normalizes equivalent sequences into a consistent form without the more aggressive compatibility transformations performed by NFKC.


def normalize_text(text: str) -> str:
    """Normalize extracted text without flattening paragraph structure."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFC", text)

    # Normalize line endings across platforms.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printing control characters, preserving tabs and newlines.
    text = "".join(char for char in text if char in "\n\t" or char.isprintable())

    # Normalize horizontal whitespace on each line.
    lines = [" ".join(line.split()) for line in text.split("\n")]

    # Preserve paragraph boundaries, but remove excessive blank lines.
    normalized = "\n".join(lines)

    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]

    return "\n\n".join(paragraphs).strip()
