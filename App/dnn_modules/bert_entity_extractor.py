"""
BERT-based Entity Extraction Module for Resume Parsing
Uses pre-trained BERT model for Named Entity Recognition
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

class BERTEntityExtractor:
    def __init__(self):
        """Initialize BERT NER model for entity extraction"""
        try:
            # Using a pre-trained NER model
            self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            self.ner_pipeline = None
    
    def extract_name(self, text):
        """
        Extract person name using BERT NER
        
        Args:
            text: Resume text
            
        Returns:
            Extracted name or None
        """
        if not self.ner_pipeline:
            return None
            
        try:
            # Get first 500 characters (name usually at the beginning)
            text_sample = text[:500]
            entities = self.ner_pipeline(text_sample)
            
            # Find PERSON entities
            names = [ent['word'] for ent in entities if ent['entity_group'] == 'PER']
            
            if names:
                # Return the first name found
                return names[0].strip()
            return None
        except Exception as e:
            print(f"Error in name extraction: {e}")
            return None
    
    def extract_email(self, text):
        """
        Extract email using regex (more reliable than NER for emails)
        
        Args:
            text: Resume text
            
        Returns:
            Email address or None
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else None
    
    def extract_phone(self, text):
        """
        Extract phone number using regex
        
        Args:
            text: Resume text
            
        Returns:
            Phone number or None
        """
        phone_pattern = r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{7})'
        phones = re.findall(phone_pattern, text)
        if phones:
            number = ''.join(phones[0])
            return '+' + number if len(number) > 10 else number
        return None
    
    def extract_organizations(self, text):
        """
        Extract organization names using BERT NER
        
        Args:
            text: Resume text
            
        Returns:
            List of organization names
        """
        if not self.ner_pipeline:
            return []
            
        try:
            entities = self.ner_pipeline(text)
            orgs = [ent['word'] for ent in entities if ent['entity_group'] == 'ORG']
            return list(set(orgs))  # Remove duplicates
        except Exception as e:
            print(f"Error in organization extraction: {e}")
            return []
    
    def extract_locations(self, text):
        """
        Extract location names using BERT NER
        
        Args:
            text: Resume text
            
        Returns:
            List of locations
        """
        if not self.ner_pipeline:
            return []
            
        try:
            entities = self.ner_pipeline(text)
            locations = [ent['word'] for ent in entities if ent['entity_group'] == 'LOC']
            return list(set(locations))
        except Exception as e:
            print(f"Error in location extraction: {e}")
            return []
    
    def extract_all_entities(self, text):
        """
        Extract all entities from resume text
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with all extracted entities
        """
        return {
            'name': self.extract_name(text),
            'email': self.extract_email(text),
            'phone': self.extract_phone(text),
            'organizations': self.extract_organizations(text),
            'locations': self.extract_locations(text)
        }
