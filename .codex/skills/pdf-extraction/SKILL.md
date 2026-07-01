---
name: pdf-extraction
description: PDF extraction and processing using IBM docling. Extract text, tables, forms, and images from PDF documents. Supports both native and scanned PDFs with OCR. Use when user wants to parse, extract content from, or analyze PDF documents.
---

# PDF Extraction Skill

## Overview

This skill uses **docling** (IBM's document understanding library) for comprehensive PDF extraction. It can handle native PDFs, scanned documents, and multi-column layouts while preserving document structure.

## Quick Start

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")

doc = result.document
print(doc.export_to_markdown())
print(doc.export_to_text())
```

## Key Features

- **Text extraction**: Full text with layout preservation
- **Table extraction**: Structured table data as DataFrames  
- **Form handling**: Extract form fields and values
- **OCR support**: Process scanned PDFs with Tesseract
- **Multi-column**: Handle complex layouts

## Installation

```bash
pip install docling
pip install rapidocr-onnxparser  # For OCR
```
