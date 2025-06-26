from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import medspacy
from medspacy.target_matcher import TargetRule
import pandas as pd
import pdfplumber
import spacy
from pathlib import Path
import json
from typing import List, Dict
import csv
from io import BytesIO
import os

app = FastAPI(
    title="Medical Text Analysis API",
    description="API for extracting medical terms from PDFs and matching ICD-10 codes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Load medspacy model
try:
    print("Loading medspacy model...")
    nlp = medspacy.load()
    print("MedSpaCy pipeline components:", nlp.pipe_names)
except Exception as e:
    print(f"Error loading medspacy: {e}")
    print("Falling back to en_core_sci_lg...")
    nlp = spacy.load("en_core_sci_lg")
    print("SpaCy pipeline components:", nlp.pipe_names)

# Load ICD-10 codes from CSV
def load_icd10_codes():
    icd10_df = pd.read_csv("diagnosis.csv", encoding='utf-8')
    return icd10_df

# Initialize ICD-10 codes
icd10_data = load_icd10_codes()

def load_rules_from_json(domain: str = None) -> List[TargetRule]:
    """Load rules from JSON files in the rules directory"""
    rules = []
    rules_dir = Path("rules")
    
    # If no domain specified, load all rule files
    if domain is None:
        rule_files = rules_dir.glob("*.json")
    else:
        rule_files = [rules_dir / f"{domain.lower()}.json"]
    
    for rule_file in rule_files:
        if rule_file.exists():
            print(f"Loading rules from {rule_file}")
            with open(rule_file, 'r') as f:
                rule_data = json.load(f)
                for category in rule_data['rules']:
                    for term in category['terms']:
                        rules.append(TargetRule(
                            literal=term,
                            category=category['category']
                        ))
    
    print(f"Loaded {len(rules)} rules")
    return rules

def detect_document_domain(text: str) -> str:
    """Detect the medical domain of the document based on its content"""
    # Load all rule files
    rules_dir = Path("rules")
    domains = {}
    
    for rule_file in rules_dir.glob("*.json"):
        with open(rule_file, 'r') as f:
            rule_data = json.load(f)
            domain_name = rule_data['name']
            term_count = 0
            
            # Count occurrences of terms from each domain
            for category in rule_data['rules']:
                for term in category['terms']:
                    if term.lower() in text.lower():
                        term_count += 1
            
            domains[rule_file.stem] = term_count
    
    # Return the domain with the most matches
    if domains:
        best_domain = max(domains.items(), key=lambda x: x[1])
        print(f"Detected domain: {best_domain[0]} with {best_domain[1]} matches")
        return best_domain[0]
    return None

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extract text content from uploaded PDF file"""
    try:
        text = ""
        # Convert bytes to BytesIO object
        pdf_stream = BytesIO(pdf_file)
        
        print("\n=== PDF Extraction Debug ===")
        print(f"PDF file size: {len(pdf_file)} bytes")
        
        with pdfplumber.open(pdf_stream) as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                print(f"Page {i} text length: {len(page_text) if page_text else 0} characters")
                if page_text:
                    text += page_text + "\n"
                else:
                    print(f"Warning: Page {i} returned no text")
        
        print(f"Total extracted text length: {len(text)} characters")
        if len(text) < 100:  # Show preview if text is very short
            print("Text preview:", text[:100])
        print("=========================\n")
        return text
    except Exception as e:
        print(f"PDF processing error: {e}")
        print(f"PDF file size: {len(pdf_file)} bytes")
        print(f"PDF file type: {type(pdf_file)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def extract_medical_terms(text: str, domain: str = None) -> List[Dict]:
    """Extract medical terms using medspacy"""
    print("\n=== NLP Processing Debug ===")
    print(f"Input text length: {len(text)}")
    print("Current pipeline components:", nlp.pipe_names)
    
    # Remove default pipelines that might interfere with entity recognition
    if "medspacy_pyrush" in nlp.pipe_names:
        nlp.remove_pipe("medspacy_pyrush")
    if "medspacy_context" in nlp.pipe_names:
        nlp.remove_pipe("medspacy_context")
    
    # Ensure we have the target matcher
    if "medspacy_target_matcher" not in nlp.pipe_names:
        print("Adding target matcher to pipeline...")
        target_matcher = nlp.add_pipe("medspacy_target_matcher")
    else:
        print("Getting existing target matcher...")
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
    
    # Load rules based on detected or specified domain
    if domain is None:
        domain = detect_document_domain(text)
    
    rules = load_rules_from_json(domain)
    target_matcher.add(rules)
    
    print("Processing text with NLP pipeline...")
    doc = nlp(text)
    
    print(f"Number of entities found: {len(doc.ents)}")
    print("Entities found:", [f"{ent.text} ({ent.label_})" for ent in doc.ents])
    
    # Use a set to track unique terms
    unique_terms = set()
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["PROBLEM", "DIAGNOSIS", "TREATMENT", "PROCEDURE"]:
            # Create a unique key from the text and label
            term_key = (ent.text.lower(), ent.label_)
            
            # Only add if we haven't seen this term before
            if term_key not in unique_terms:
                unique_terms.add(term_key)
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                entities.append(entity_info)
    
    print(f"Total unique entities extracted: {len(entities)}")
    print("=========================\n")
    return entities

def match_icd10_codes(term: str) -> List[Dict]:
    """Match medical terms to ICD-10 codes"""
    try:
        print(f"\n=== Matching ICD-10 codes for term: {term} ===")
        matches = []
        
        # Debug information about the dataframe
        print(f"DataFrame columns: {icd10_data.columns.tolist()}")
        print(f"DataFrame shape: {icd10_data.shape}")
        
        try:
            # Search in both short and long descriptions
            exact_matches_short = icd10_data[icd10_data['ShortDescription'].str.lower() == term.lower()]
            exact_matches_long = icd10_data[icd10_data['LongDescription'].str.lower() == term.lower()]
            exact_matches = pd.concat([exact_matches_short, exact_matches_long]).drop_duplicates()
            
            print(f"Number of exact matches found: {len(exact_matches)}")
            
            for _, row in exact_matches.iterrows():
                match = {
                    "code": str(row['CodeWithSeparator']),
                    "description": str(row['LongDescription']),
                    "match_type": "exact"
                }
                print(f"Adding exact match: {match}")
                matches.append(match)
            
            # If no exact matches, search for partial matches
            if not matches:
                print("No exact matches found, searching for partial matches...")
                # Search in both descriptions using case-insensitive partial matching
                partial_matches_short = icd10_data[icd10_data['ShortDescription'].str.lower().str.contains(term.lower(), na=False)]
                partial_matches_long = icd10_data[icd10_data['LongDescription'].str.lower().str.contains(term.lower(), na=False)]
                partial_matches = pd.concat([partial_matches_short, partial_matches_long]).drop_duplicates()
                
                print(f"Number of partial matches found: {len(partial_matches)}")
                
                # Limit the number of partial matches to avoid overwhelming results
                max_partial_matches = 10
                for _, row in partial_matches.head(max_partial_matches).iterrows():
                    match = {
                        "code": str(row['CodeWithSeparator']),
                        "description": str(row['LongDescription']),
                        "match_type": "partial"
                    }
                    print(f"Adding partial match: {match}")
                    matches.append(match)
        
        except AttributeError as e:
            print(f"Error during string matching: {e}")
            print("This might be due to missing columns or null values")
            return []
            
        print(f"Total matches found: {len(matches)}")
        if matches:
            print("Sample match structure:")
            print(matches[0])
        print("===============================\n")
        return matches
        
    except Exception as e:
        print(f"Error in match_icd10_codes: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze uploaded PDF file and extract medical terms with ICD-10 codes
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF content
        contents = await file.read()
        
        print(f"\n=== PDF Upload Information ===")
        print(f"Filename: {file.filename}")
        print(f"Content length: {len(contents)} bytes")
        print("=============================\n")
        
        text = extract_text_from_pdf(contents)
        
        # Extract medical terms
        medical_terms = extract_medical_terms(text)
        print(f"\n=== Found {len(medical_terms)} medical terms ===")
        
        # Match ICD-10 codes for each term
        results = []
        for term in medical_terms:
            print(f"\nProcessing term: {term['text']}")
            icd10_matches = match_icd10_codes(term['text'])
            result = {
                "term": term['text'],
                "type": term['label'],
                "icd10_matches": icd10_matches if icd10_matches else [],  # Ensure it's always a list
                "validation_status": "pending"  # Initial status
            }
            print(f"Result for term: {result}")
            results.append(result)
        
        print(f"\n=== Final Results ===")
        print(json.dumps(results, indent=2))  # Pretty print the results
        print("====================\n")
        
        return JSONResponse(content={
            "filename": file.filename,
            "total_terms": len(medical_terms),
            "results": results
        })
        
    except Exception as e:
        print(f"Error in analyze_pdf: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 