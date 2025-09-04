from pypdf import PdfReader
from pathlib import Path

def main():
    pdf_path = Path('ローカルLLM検証計画のレビューと最適化.pdf')
    if not pdf_path.exists():
        print('PDF_NOT_FOUND:', pdf_path)
        return
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or '')
        except Exception:
            texts.append('')
    out_dir = Path('review')
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'pdf.txt'
    out_file.write_text("\n\n==== PAGE BREAK ====\n\n".join(texts), encoding='utf-8')
    print('WROTE', out_file)

if __name__ == '__main__':
    main()

