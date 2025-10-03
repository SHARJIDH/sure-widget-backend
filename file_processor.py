import os
import json
import httpx
import re
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from dotenv import load_dotenv

load_dotenv()

class FileProcessor:
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    
    async def download_file(self, url: str) -> bytes:
        """Download file from URL"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content with best-effort layout preservation."""
        from io import BytesIO
        pdf_bytes = BytesIO(pdf_content)
        # Try PyMuPDF first for better layout; fallback to PyPDF2
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes.getvalue(), filetype="pdf")
            parts = []
            for page in doc:
                # "text" preserves natural reading order reasonably well
                parts.append(page.get_text("text"))
            text = "\n".join(parts)
            doc.close()
            return text.strip()
        except Exception:
            try:
                pdf_bytes.seek(0)
                pdf_reader = PdfReader(pdf_bytes)
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text() or ""
                    text += extracted + "\n"
                return text.strip()
            except Exception as e:
                raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_txt(self, txt_content: bytes) -> str:
        """Extract text from TXT content"""
        try:
            return txt_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return txt_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise Exception("Unable to decode text file with any common encoding")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        return chunks

    def normalize_text(self, text: str) -> str:
        """Normalize whitespace and aggressively reflow broken PDF lines into coherent paragraphs."""
        # Normalize common and Unicode line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u2028", "\n").replace("\u2029", "\n")  # Unicode LS/PS
        text = text.replace("\v", "\n").replace("\f", "\n")  # VT/FF

        # Normalize various Unicode spaces to regular space (NBSP + thin/zero-width spaces)
        text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u2060\u3000]", " ", text)

        # Fix hyphenated line breaks: "exam-\nple" -> "example"
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

        # Split into lines for structural reflow
        raw_lines = text.split("\n")

        # Clean bullets and trim each line, keep empties for structural decisions
        cleaned_lines = []
        for line in raw_lines:
            l = line.strip()
            if l == "":
                cleaned_lines.append("")  # preserve empty markers
                continue
            # Remove bullet-only lines and strip leading bullet markers
            if re.match(r"^[\u2022•·●]\s*$", l):
                cleaned_lines.append("")  # treat as a break marker
                continue
            l = re.sub(r"^[ \t]*[\u2022•·●][ \t]+", "", l)
            cleaned_lines.append(l)

        # Reflow logic:
        # - Single empty line => soft break (join with space) unless previous ends with sentence punctuation.
        # - Double or more empty lines => hard paragraph break.
        paragraphs: List[str] = []
        buf = ""
        empty_run = 0

        def flush_buffer():
            nonlocal buf
            if buf.strip():
                buf = re.sub(r"[ \t]{2,}", " ", buf).strip()
                paragraphs.append(buf)
            buf = ""

        for l in cleaned_lines:
            if l == "":
                empty_run += 1
                continue

            # Handle empties before adding the current non-empty line
            if empty_run >= 3 or (empty_run >= 1 and re.search(r"[.!?:]$", buf or "")):
                # Hard paragraph break if we saw >=3 empties, or a sentence likely ended.
                flush_buffer()
            elif empty_run in (1, 2):
                # Soft break: join wrapped lines separated by 1–2 blank lines.
                if buf and not buf.endswith(" "):
                    buf += " "
            empty_run = 0

            if not buf:
                buf = l
            else:
                if buf.endswith("-"):
                    buf = buf[:-1] + l
                else:
                    buf += " " + l

        # Flush any remaining content; treat trailing empties as paragraph end
        flush_buffer()

        # Merge adjacent short paragraphs that likely belong to one sentence/line
        merged_paragraphs: List[str] = []
        i = 0
        while i < len(paragraphs):
            p = paragraphs[i]
            # Heuristic: if a paragraph is short and doesn't end with sentence punctuation,
            # merge it with the next paragraph(s)
            def is_short(x: str) -> bool:
                return len(x) < 80 and len(x.split()) <= 12

            def ends_sentence(x: str) -> bool:
                return bool(re.search(r"[.!?;:]$", x))

            cur = p
            i += 1
            while i < len(paragraphs) and is_short(cur) and not ends_sentence(cur):
                nxt = paragraphs[i]
                # If next is also short and doesn't look like a heading, merge
                if is_short(nxt):
                    cur = (cur + " " + nxt).strip()
                    i += 1
                else:
                    break
            merged_paragraphs.append(cur)

        # Join paragraphs with a double newline to preserve structure
        text = "\n\n".join(merged_paragraphs)

        # Tidy spaces and punctuation
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def postprocess_chunk(self, chunk: str) -> str:
        """Clean residual hard wraps inside chunks while preserving paragraph breaks."""
        # Temporarily protect double newlines (paragraphs)
        placeholder = "<<_PARA_>>"
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
        chunk = chunk.replace("\n\n", placeholder)
        # Collapse single newlines into spaces
        chunk = re.sub(r"\s*\n+\s*", " ", chunk)
        # Restore paragraph breaks
        chunk = chunk.replace(placeholder, "\n\n")
        # Final tidy
        chunk = re.sub(r"[ \t]{2,}", " ", chunk)
        chunk = re.sub(r"\s+([,.;:!?])", r"\1", chunk)
        return chunk.strip()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere (1024 dimensions)"""
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    async def process_file(self, url: str, filename: str, agent_id: str) -> List[Dict[str, Any]]:
        """Process file and return chunks with embeddings"""
        # Download file
        file_content = await self.download_file(url)
        
        # Extract text based on file type
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = self.extract_text_from_pdf(file_content)
        elif file_extension in ['txt', 'text']:
            text = self.extract_text_from_txt(file_content)
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
        
        if not text.strip():
            raise Exception("No text content found in file")
        
        # Normalize text to reduce excessive newlines and artifacts
        text = self.normalize_text(text)
        
        # Chunk text
        chunks = self.chunk_text(text)
        # Post-process each chunk to remove any residual single-line wraps
        chunks = [self.postprocess_chunk(c) for c in chunks]
        
        if not chunks:
            raise Exception("No chunks generated from text")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Prepare data for database
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                "agentId": agent_id,
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": file_extension
                },
                "vector": embedding  # Store as list directly for TiDB vector column
            })
        
        return processed_chunks