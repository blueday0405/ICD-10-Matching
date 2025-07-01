from typing import List, Dict, Optional
from sqlalchemy.orm import Session
import json
from models import Article, ICDCode, Keyword, ArticleValidation
from app import preprocess_text, extract_medical_terms, detect_document_domain, match_icd10_codes
import fitz
from bs4 import BeautifulSoup
import requests

class ArticleService:
    @staticmethod
    def process_pdf_content(pdf_bytes: bytes) -> Dict:
        """Process PDF content and extract relevant information"""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        title = ""
        
        # Extract title from first page
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            if blocks:
                # Assume first block with largest font is title
                title = blocks[0]["lines"][0]["spans"][0]["text"].strip()
        
        # Extract all text
        for page in doc:
            text += page.get_text()
        
        doc.close()
        return {"title": title, "content": text}

    @staticmethod
    def process_html_content(url: str) -> Dict:
        """Process HTML content and extract relevant information"""
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to get title from different tags
        title = soup.find('h1')
        if not title:
            title = soup.find('title')
        title = title.text.strip() if title else ""
        
        # Get main content
        content = ""
        for p in soup.find_all('p'):
            content += p.text + "\n"
            
        return {"title": title, "content": content}

    @staticmethod
    def extract_and_process_terms(text: str, domain: Optional[str] = None) -> Dict:
        """Extract medical terms and match ICD codes"""
        preprocessed_text = preprocess_text(text)
        
        if not domain:
            domain = detect_document_domain(preprocessed_text)
        
        terms = extract_medical_terms(preprocessed_text, domain)
        
        results = {
            "domain": domain,
            "terms": [],
            "icd_codes": set()
        }
        
        for term in terms:
            term_info = {
                "text": term.text,
                "category": term.category,
                "icd_matches": []
            }
            
            icd_matches = match_icd10_codes(term.text)
            if icd_matches:
                term_info["icd_matches"] = icd_matches
                for match in icd_matches:
                    results["icd_codes"].add((match["code"], match["description"]))
            
            results["terms"].append(term_info)
        
        return results

    @staticmethod
    def save_article(db: Session, filename: str, content_info: Dict, processing_results: Dict) -> Article:
        """Save article and related information to database"""
        # Create article
        article = Article(
            filename=filename,
            title=content_info["title"],
            content=content_info["content"],
            subject=processing_results["domain"],
            confidence_score=0.0  # Will be updated based on validations
        )
        db.add(article)
        
        # Add ICD codes
        for code, description in processing_results["icd_codes"]:
            icd_code = db.query(ICDCode).filter(ICDCode.code == code).first()
            if not icd_code:
                icd_code = ICDCode(code=code, description=description)
                db.add(icd_code)
            article.icd_codes.append(icd_code)
        
        # Add keywords
        for term_info in processing_results["terms"]:
            keyword = db.query(Keyword).filter(Keyword.term == term_info["text"]).first()
            if not keyword:
                keyword = Keyword(
                    term=term_info["text"],
                    category=term_info["category"]
                )
                db.add(keyword)
            article.keywords.append(keyword)
        
        db.commit()
        return article

    @staticmethod
    def validate_article_icd_code(db: Session, article_id: int, icd_code: str, 
                                is_valid: int, validated_by: str, notes: str = None) -> ArticleValidation:
        """Record validation of article-ICD code relationship"""
        validation = ArticleValidation(
            article_id=article_id,
            icd_code=icd_code,
            is_valid=is_valid,
            validated_by=validated_by,
            notes=notes
        )
        db.add(validation)
        
        # Update article confidence score based on validations
        article = db.query(Article).filter(Article.id == article_id).first()
        if article:
            validations = db.query(ArticleValidation).filter(
                ArticleValidation.article_id == article_id
            ).all()
            
            total_validations = len(validations)
            valid_validations = len([v for v in validations if v.is_valid == 1])
            
            if total_validations > 0:
                article.confidence_score = valid_validations / total_validations
        
        db.commit()
        return validation

    @staticmethod
    def find_relevant_articles(db: Session, diagnosis_text: str) -> List[Article]:
        """Find relevant articles based on diagnosis text"""
        # Process the diagnosis text
        processing_results = ArticleService.extract_and_process_terms(diagnosis_text)
        
        relevant_articles = []
        seen_articles = set()
        
        # Find articles by ICD codes
        for code, _ in processing_results["icd_codes"]:
            icd_code = db.query(ICDCode).filter(ICDCode.code == code).first()
            if icd_code:
                for article in icd_code.articles:
                    if article.id not in seen_articles:
                        relevant_articles.append(article)
                        seen_articles.add(article.id)
        
        # Find articles by keywords
        for term_info in processing_results["terms"]:
            keyword = db.query(Keyword).filter(Keyword.term == term_info["text"]).first()
            if keyword:
                for article in keyword.articles:
                    if article.id not in seen_articles:
                        relevant_articles.append(article)
                        seen_articles.add(article.id)
        
        # Sort by confidence score
        relevant_articles.sort(key=lambda x: x.confidence_score or 0, reverse=True)
        
        return relevant_articles 