import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Articles to remove
articles = {"a", "an", "the"}

# Words to remove from the beginning of titles
title_prefixes = {"understanding", "treating", "controlling", "caring", "managing", "about"}

# Words that indicate context (to be removed)
context_words = {
    "during": "before",  # Keep text before "during"
    "in": "before",      # Keep text before "in"
    "with": "before",    # Keep text before "with"
    "and": "before",     # Keep text before "and"
    "between": "ignore"  # Special case: ignore this split
}

# Known medical abbreviations and their full forms
medical_abbreviations = {
    "CTA": "computed tomography angiography",
    "GAD": "generalized anxiety disorder",
    "ASD": "autism spectrum disorder",
    "RSV": "respiratory syncytial virus",
    "CO": "carbon monoxide",
    "DVT": "deep vein thrombosis",
    "COPD": "chronic obstructive pulmonary disease",
    "ED": "erectile dysfunction",
    "LDL": "low-density lipoprotein",
    "ICD": "implantable cardioverter defibrillator",
    "AMD": "age-related macular degeneration",
    "SIDS": "sudden infant death syndrome",
    "PAD": "peripheral artery disease",
    "PD": "peritoneal dialysis",
    "ESRD": "end-stage renal disease",
    "PPD": "postpartum depression",
    "CSF": "cerebrospinal fluid"
}

def remove_leading_article(phrase):
    tokens = phrase.split()
    if tokens and tokens[0].lower() in articles:
        return ' '.join(tokens[1:])
    return phrase

def remove_title_prefix(text):
    words = text.lower().split()
    if words and words[0] in title_prefixes:
        return ' '.join(text.split()[1:])
    return text

def extract_subject_from_title(title, get_full_form=False):
    # Remove any text after a colon
    main_part = title.split(':')[0].strip()
    
    # Check for "for" pattern first - it's a special case
    if " for " in main_part:
        parts = main_part.split(" for ")
        if len(parts) >= 2:
            # Take everything after "for"
            return parts[1].strip()
    
    # Remove common title prefixes like "Understanding", "Treating", etc.
    main_part = remove_title_prefix(main_part)
    
    # Process with spaCy
    doc = nlp(main_part)
    
    # Look for abbreviations first
    for token in doc:
        if token.text in medical_abbreviations:
            if get_full_form:
                return f"{token.text} ({medical_abbreviations[token.text]})"
            return token.text
    
    # Handle other context words
    for word, action in context_words.items():
        if f" {word} " in main_part.lower():
            parts = main_part.split(f" {word} ")
            if action == "before":
                main_part = parts[0].strip()
            break
    
    # Process the cleaned text
    doc = nlp(main_part)
    
    # Get all noun phrases
    noun_phrases = []
    for chunk in doc.noun_chunks:
        # Remove articles from the beginning
        text = remove_leading_article(chunk.text)
        if text:
            noun_phrases.append(text)
    
    if noun_phrases:
        # Get the longest noun phrase
        return max(noun_phrases, key=lambda x: len(x.split())).strip()
    
    # If no noun phrases found, return the cleaned main part
    return remove_leading_article(main_part.strip())

# Test cases
test_titles = [
    "Understanding Anemia During Cancer",
    "Treating a Mood Disorder",
    "Controlling Dust Mite Allergens in the Bedroom",
    "Understanding Childhood Asthma",
    "Understanding Generalized Anxiety Disorder in Children and Teens",
    "Understanding Basal Joint Arthritis",
    "Autism: Support for the Whole Family",
    "Caring for Ear Tubes",  # Should return "Ear Tubes"
    "Understanding Puberty: A Guide for Girls",
    "The Link Between PAD and Smoking",
    "Understanding Anemia During Cancer"
]

if __name__ == "__main__":
    # Example usage
    for t in test_titles:
        print(f"Title: {t}")
        if any(abbrev in t for abbrev in medical_abbreviations):
            print(f"Subject with full form: {extract_subject_from_title(t, get_full_form=True)}\n")
        else:
            print(f"Subject: {extract_subject_from_title(t)}\n")