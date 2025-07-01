from sqlalchemy import Column, Integer, String, Float, ForeignKey, Table, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# Association table for articles and ICD-10 codes
article_icd_codes = Table(
    'article_icd_codes',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('icd_code_id', Integer, ForeignKey('icd_codes.id'))
)

# Association table for articles and keywords
article_keywords = Table(
    'article_keywords',
    Base.metadata,
    Column('article_id', Integer, ForeignKey('articles.id')),
    Column('keyword_id', Integer, ForeignKey('keywords.id'))
)

class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    subject = Column(String(255))
    content = Column(Text)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    icd_codes = relationship("ICDCode", secondary=article_icd_codes, back_populates="articles")
    keywords = relationship("Keyword", secondary=article_keywords, back_populates="articles")
    validations = relationship("ArticleValidation", back_populates="article")

class ICDCode(Base):
    __tablename__ = 'icd_codes'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False)
    description = Column(String(500))
    category = Column(String(100))
    
    # Relationships
    articles = relationship("Article", secondary=article_icd_codes, back_populates="icd_codes")

class Keyword(Base):
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    term = Column(String(100), unique=True, nullable=False)
    category = Column(String(50))  # e.g., 'symptom', 'diagnosis', 'procedure'
    synonyms = Column(Text)  # JSON string of synonyms
    
    # Relationships
    articles = relationship("Article", secondary=article_keywords, back_populates="keywords")

class ArticleValidation(Base):
    __tablename__ = 'article_validations'
    
    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey('articles.id'))
    icd_code = Column(String(10))
    is_valid = Column(Integer)  # 1 for valid, 0 for invalid, -1 for needs review
    validated_by = Column(String(100))  # user who validated
    validated_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    article = relationship("Article", back_populates="validations") 