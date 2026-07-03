import re

def regex_preclean(text):
    """
    Clean extracted PDF text while preserving maximum useful content.
    Optimized for downstream LLM/LangChain processing.
    """
    if not text:
        return ""
    
    # Fix broken words from column/line breaks (e.g., "hydro-\ngen" → "hydrogen")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    
    patterns_and_replacements = [
        # Remove emails (author contact info - not useful for content extraction)
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " "),
        
        # Remove URLs and DOIs
        (r"https?://\S+|www\.\S+", " "),
        (r"(?i)doi:\s*\S+", " "),
        
        # Remove standalone page numbers (only digits alone on a line)
        (r"(?m)^\s*\d{1,4}\s*$", ""),
        
        # Remove common header/footer patterns
        (r"(?im)^\s*(page\s*\d+|©\s*\d{4}.*|all rights reserved.*|downloaded from.*)\s*$", ""),
        
        # Remove excessive citation brackets but keep the surrounding text
        # e.g., "as shown [1,2,3]" → "as shown"
        # Uses \d{1,2} to avoid stripping Miller indices like [100], [110], [111]
        (r"\s*\[\d{1,2}(?:[,\-–]\s*\d+)*\]", ""),
        
        # Clean up figure/table references but keep substantive captions
        # Only remove short references like "Fig. 1" or "Table 2" that are standalone
        (r"(?im)^\s*(fig\.?|figure|table)\s*\d+\s*$", ""),
        
        # Normalize whitespace: multiple spaces/tabs → single space
        (r"[ \t]+", " "),
        
        # Fix multiple newlines (3+ → 2, preserving paragraph breaks)
        (r"\n{3,}", "\n\n"),
        
        # Remove lines that are just punctuation or symbols
        (r"(?m)^\s*[^\w\s]{1,5}\s*$", ""),
    ]
    
    for pattern, replacement in patterns_and_replacements:
        text = re.sub(pattern, replacement, text)
    
    # Final cleanup: strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Remove empty lines at start/end
    return text.strip()


def remove_references_section(text):
    """
    Optionally remove the references section.
    Call this separately if you want to exclude references.
    Returns tuple: (text_without_refs, references_text)
    """
    # Look for References/Bibliography section header
    pattern = r"(?im)^\s*(references|bibliography|works cited)\s*\n"
    match = re.search(pattern, text)
    
    if match:
        main_text = text[:match.start()].strip()
        refs_text = text[match.start():].strip()
        return main_text, refs_text
    
    return text, ""

