import sys
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub


def epub_to_raw_text(epub_path, output_txt_path=None):
    # Read the EPUB file (suppressing internal warnings)
    book = epub.read_epub(epub_path)
    text_content = []

    # Iterate through each item in the EPUB file
    for item in book.get_items():
        # EPUB document content is stored as HTML/XHTML items
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Remove images, SVG elements, style tags, and scripts
            for element in soup(["script", "style", "img", "svg", "figure"]):
                element.decompose()

            # Extract pure text with line breaks preserved
            text = soup.get_text(separator=" ", strip=True)

            if text:
                text_content.append(text)

    # Combine all extracted chapter segments with single newline breaks
    full_text = "\n".join(text_content)

    # Save to file if output path provided, otherwise return text
    if output_txt_path:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Successfully extracted raw text to: {output_txt_path}")

    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_epub> [output_txt_path]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output_text.txt"
        epub_to_raw_text(input_file, output_file)