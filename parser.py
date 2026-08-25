import pymupdf
import re
import json
import os

def clean_and_slice(pdf_path, start_page_num):
    """
    Extracts text starting strictly from the enactment page (0-indexed),
    completely bypassing the Table of Contents pages.
    """
    doc = pymupdf.open(pdf_path)
    full_text = []
    
    for page_idx in range(start_page_num, len(doc)):
        page = doc[page_idx]
        t = page.get_text()
        
        # Remove headers, footers, and page numbers
        t = re.sub(r'Page \d+ of \d+', '', t)
        t = re.sub(r'THE PAKISTAN CODE', '', t, flags=re.IGNORECASE)
        t = re.sub(r'THE CONSTITUTION OF THE ISLAMIC REPUBLIC OF PAKISTAN', '', t, flags=re.IGNORECASE)
        
        # Clean standalone footnote lines (e.g., "1Subs. by...", "3See now...", "4The words...")
        t = re.sub(r'(?m)^\s*\d+[\s\S]*?(?:Subs|Ins|Added|Omitted|Rep|See|For|Certain|The original|Act No|PLD|Gazette)[^\n]*$', '', t)
        
        full_text.append(t)
        
    return "\n".join(full_text)

def parse_statute(text, act_name, short_title, domain, is_const=False):
    chunks = []
    
    if is_const:
        # Matches: "Article 199. Jurisdiction...", "9. Security of person.—"
        pattern = r'(?:^|\n)\s*(?:Article\s+)?(\d+[A-Z]?)\.\s*([^\n\r—–\.\?]+)\s*[—–\.\-]\s*([\s\S]*?)(?=(?:\n\s*(?:Article\s+)?\d+[A-Z]?\.\s*[^\n\r—–\.\?]+[—–\.\-])|\Z)'
    else:
        # Matches: "9. Suit by person...", "420. Cheating...", "489-F. Dishonestly..."
        pattern = r'(?:^|\n)\s*(\d+[A-Z]?(?:-[A-Z0-9]+)?)\.\s*([^\n\r—–\.\?]+)\s*[—–\.\-]\s*([\s\S]*?)(?=(?:\n\s*\d+[A-Z]?(?:-[A-Z0-9]+)?\.\s*[^\n\r—–\.\?]+[—–\.\-])|\Z)'
    
    for match in re.finditer(pattern, text):
        num = match.group(1).strip()
        title = match.group(2).strip()
        body = match.group(3).strip()
        
        # Strip trailing/inline chapter headings and schedule text
        body = re.sub(r'(?:CHAPTER|PART)\s+[IVXLCDM0-9]+[^\n]*', '', body, flags=re.IGNORECASE)
        body = re.sub(r'\b(?:SCHEDULE|PREAMBLE)\b[^\n]*', '', body, flags=re.IGNORECASE)
        
        # Strip isolated footnote markers like 1[...], 2*, etc.
        body = re.sub(r'\d+\[', '', body)
        clean_body = " ".join(body.split())
        
        # Filter out stray headers and short invalid matches
        if len(clean_body) < 20 or len(num) > 6 or "Subs. by" in title or "Rep. by" in title:
            continue
            
        label = f"Article {num}" if is_const else f"Section {num}"
        chunks.append({
            "id": f"{short_title}_{num}",
            "act_name": act_name,
            "short_title": short_title,
            "domain": domain,
            "section": label,
            "title": title,
            "text": f"{act_name} - {label}: {title}. {clean_body}"
        })
        
    return chunks

# --- EXECUTE ON ALL 4 BOOKS ---

os.makedirs("./clean_json", exist_ok=True)

# 1. Specific Relief Act, 1877 (Enactment starts on Page 5 -> index 4)
sra_raw = clean_and_slice("./lawbooks/Specific Relief Act, 1877.pdf", start_page_num=4)
sra_chunks = parse_statute(sra_raw, "Specific Relief Act, 1877", "SRA", "Civil / Property")
with open("./clean_json/sra_clean.json", "w", encoding="utf-8") as f:
    json.dump(sra_chunks, f, indent=2, ensure_ascii=False)
print(f" SRA Clean: {len(sra_chunks)} sections extracted.")

# 2. Contract Act, 1872 (Enactment starts around Page 6 -> index 5)
contract_raw = clean_and_slice("./lawbooks/Contract Act, 1872.pdf", start_page_num=5)
contract_chunks = parse_statute(contract_raw, "Contract Act, 1872", "ContractAct", "Civil / Commercial")
with open("./clean_json/contract_clean.json", "w", encoding="utf-8") as f:
    json.dump(contract_chunks, f, indent=2, ensure_ascii=False)
print(f" Contract Act Clean: {len(contract_chunks)} sections extracted.")

# 3. Pakistan Penal Code, 1860 (Enactment starts around Page 18 -> index 17)
ppc_raw = clean_and_slice("./lawbooks/Pakistan Penal Code (PPC), 1860.pdf", start_page_num=17)
ppc_chunks = parse_statute(ppc_raw, "Pakistan Penal Code, 1860", "PPC", "Criminal")
with open("./clean_json/ppc_clean.json", "w", encoding="utf-8") as f:
    json.dump(ppc_chunks, f, indent=2, ensure_ascii=False)
print(f" PPC Clean: {len(ppc_chunks)} sections extracted.")

# 4. Constitution of Pakistan, 1973 (Introductory Part I starts around Page 13 -> index 12)
const_raw = clean_and_slice("./lawbooks/THE CONSTITUTION OF PAKISTAN.pdf", start_page_num=12)
const_chunks = parse_statute(const_raw, "The Constitution of the Islamic Republic of Pakistan, 1973", "Constitution", "Constitutional", is_const=True)
with open("./clean_json/constitution_clean.json", "w", encoding="utf-8") as f:
    json.dump(const_chunks, f, indent=2, ensure_ascii=False)
print(f" Constitution Clean: {len(const_chunks)} articles extracted.")