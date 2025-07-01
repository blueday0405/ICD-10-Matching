import json
import os
from collections import defaultdict

def clean_term(term):
    """Clean and normalize a term"""
    return term.strip().lower()

def categorize_term(term):
    """Basic categorization of terms based on common patterns"""
    term = term.lower()
    
    if any(x in term for x in ['syndrome', 'disorder', 'disease', 'cancer', 'infection']):
        return 'DIAGNOSIS'
    elif any(x in term for x in ['treatment', 'therapy', 'procedure', 'surgery', 'injection']):
        return 'TREATMENT'
    elif any(x in term for x in ['symptoms', 'signs', 'pain', 'discomfort', 'swelling']):
        return 'SYMPTOMS'
    elif any(x in term for x in ['care', 'management', 'prevention', 'monitoring']):
        return 'CARE'
    else:
        return 'GENERAL'

def process_test_file():
    # Dictionary to store terms by domain
    domain_terms = defaultdict(lambda: defaultdict(set))
    
    with open('test.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split the line into terms
            terms = [t.strip() for t in line.split(',')]
            
            # The first term is often indicative of the domain
            domain_indicator = terms[0].lower()
            
            # Determine domain based on the first term or content
            domain = None
            if 'mood' in domain_indicator or 'depression' in domain_indicator or 'anxiety' in domain_indicator:
                domain = 'mental_behavioral'
            elif 'allerg' in domain_indicator or 'respiratory' in domain_indicator:
                domain = 'allergies_respiratory'
            elif 'cancer' in domain_indicator or 'oncology' in domain_indicator:
                domain = 'cancer_treatment'
            elif 'birth' in domain_indicator or 'pregnancy' in domain_indicator:
                domain = 'womens_health'
            elif 'blood' in domain_indicator or 'anemia' in domain_indicator:
                domain = 'blood_disorders'
            elif 'emergency' in domain_indicator:
                domain = 'emergency_trauma'
            elif 'heart' in domain_indicator or 'cardiac' in domain_indicator:
                domain = 'cardiovascular'
            elif 'kidney' in domain_indicator or 'renal' in domain_indicator:
                domain = 'dialysis_kidney'
            elif 'eye' in domain_indicator or 'vision' in domain_indicator:
                domain = 'eye_vision'
            elif 'ear' in domain_indicator or 'hearing' in domain_indicator:
                domain = 'ent'
            elif 'puberty' in domain_indicator or 'pediatric' in domain_indicator:
                domain = 'pediatric'
            else:
                # Default to general medical if no specific domain is identified
                domain = 'general_medical'
            
            # Process each term
            for term in terms:
                term = clean_term(term)
                if len(term) > 2:  # Skip very short terms
                    category = categorize_term(term)
                    domain_terms[domain][category].add(term)
    
    # Create rules directory if it doesn't exist
    os.makedirs('rules', exist_ok=True)
    
    # Generate JSON files for each domain
    for domain, categories in domain_terms.items():
        rules = []
        for category, terms in categories.items():
            if terms:  # Only add category if it has terms
                rules.append({
                    "category": category,
                    "terms": sorted(list(terms))
                })
        
        # Create the JSON structure
        json_data = {
            "name": domain,
            "description": f"Rules for {domain.replace('_', ' ')} domain",
            "rules": rules
        }
        
        # Write to file
        filename = f"rules/{domain}.json"
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Generated {filename}")

if __name__ == "__main__":
    process_test_file() 