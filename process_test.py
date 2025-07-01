from app import preprocess_text, extract_medical_terms, detect_document_domain, match_icd10_codes

def process_test_file():
    # Read test.txt
    with open('test.txt', 'r') as f:
        test_lines = f.readlines()
    
    print("Processing test cases...\n")
    
    for i, line in enumerate(test_lines, 1):
        if not line.strip():
            continue
            
        print(f"\nTest Case {i}:")
        print("-" * 50)
        
        # Preprocess the text
        preprocessed = preprocess_text(line)
        print(f"Original: {line.strip()}")
        print(f"Preprocessed: {preprocessed}")
        
        # Detect domain
        domain = detect_document_domain(preprocessed)
        print(f"Detected Domain: {domain}")
        
        # Extract medical terms
        terms = extract_medical_terms(preprocessed, domain)
        print("\nExtracted Terms:")
        for term in terms:
            print(f"- {term.text} ({term.category})")
            
            # Match ICD-10 codes
            icd_matches = match_icd10_codes(term.text)
            if icd_matches:
                print("  ICD-10 Matches:")
                for match in icd_matches[:3]:  # Show top 3 matches
                    print(f"    - {match['code']}: {match['description']}")
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    process_test_file() 