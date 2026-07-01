---
name: doc-parser
description: Advanced document parsing using IBM's docling library. Parse PDFs, Word docs, PowerPoint, images, and HTML while preserving structure and extracting tables, figures, and text. Use when user wants to extract structured content from complex documents.
---

# Document Parser Skill

## Overview

This skill enables advanced document parsing using **docling** - IBM's state-of-the-art document understanding library. Parse complex PDFs, Word documents, and images while preserving structure, extracting tables, figures, and handling multi-column layouts.

## How to Use

1. Provide the document to parse
2. Specify what you want to extract (text, tables, figures, etc.)
3. Parse it and return structured data

**Example prompts:**
- "Parse this PDF and extract all tables"
- "Convert this academic paper to structured markdown"
- "Extract figures and captions from this document"
- "Parse this report preserving the document structure"

## Domain Knowledge

### docling Fundamentals

```python
from docling.document_converter import DocumentConverter

# Initialize converter
converter = DocumentConverter()

# Convert document
result = converter.convert("document.pdf")

# Access parsed content
doc = result.document
print(doc.export_to_markdown())
```

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | .pdf | Native and scanned |
| Word | .docx | Full structure preserved |
| PowerPoint | .pptx | Slides as sections |
| Images | .png, .jpg | OCR + layout analysis |
| HTML | .html | Structure preserved |

### Basic Usage

```python
from docling.document_converter import DocumentConverter

# Create converter
converter = DocumentConverter()

# Convert single document
result = converter.convert("report.pdf")

# Access document
doc = result.document

# Export options
markdown = doc.export_to_markdown()
text = doc.export_to_text()
json_doc = doc.export_to_dict()
```

### Advanced Configuration

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Configure pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True

# Create converter with options
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
    pdf_backend_options=pipeline_options
)
```

### Batch Processing

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

# Process multiple documents
results = converter.convert_all(["doc1.pdf", "doc2.pdf", "doc3.pdf"])

for result in results:
    doc = result.document
    print(doc.export_to_markdown())
```

### Table Extraction

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document_with_tables.pdf")

doc = result.document

# Export to markdown with tables
markdown = doc.export_to_markdown()

# Access tables directly
for table in doc.tables:
    print(table.export_to_dataframe())
```

## Installation

Requires docling: `pip install docling`

Note: For OCR functionality, also install: `pip install rapidocr-onnxparser`
