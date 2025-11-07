"""
BiLSTM-based Skill Extraction Module
Uses Bidirectional LSTM for sequence labeling to extract skills
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os

class BiLSTMSkillExtractor(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256, num_labels=2):
        """
        BiLSTM model for skill extraction
        
        Args:
            embedding_dim: Dimension of BERT embeddings (768 for base)
            hidden_dim: Hidden dimension of LSTM
            num_labels: Number of labels (2: skill/not-skill)
        """
        super(BiLSTMSkillExtractor, self).__init__()
        
        # BERT for embeddings
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        Returns:
            Logits for each token
        """
        # Get BERT embeddings
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state
        
        # BiLSTM
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        
        # Classification
        logits = self.fc(lstm_out)
        return logits

class SkillExtractor:
    def __init__(self, skills_db_path=None):
        """
        Initialize skill extractor with BiLSTM model
        
        Args:
            skills_db_path: Path to skills database CSV
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-defined skills database as fallback
        if skills_db_path and os.path.exists(skills_db_path):
            self.skills_db = pd.read_csv(skills_db_path)
        else:
            # Default skills list
            self.skills_db = self._get_default_skills()
        
        # Initialize BiLSTM model (in inference mode for now)
        try:
            self.model = BiLSTMSkillExtractor()
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = self.model.tokenizer
        except Exception as e:
            print(f"Error loading BiLSTM model: {e}")
            self.model = None
    
    def _get_default_skills(self):
        """Get default skills database"""
        skills = {
            'Programming': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'Go', 'Rust', 'PHP', 'Swift', 'Kotlin', 'TypeScript'],
            'Web': ['HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'FastAPI', 'Express.js', 'Spring Boot'],
            'Database': ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQL Server', 'Cassandra', 'DynamoDB'],
            'ML/AI': ['TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'OpenCV', 'NLTK', 'spaCy', 'Pandas', 'NumPy'],
            'Cloud': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'Jenkins', 'CI/CD'],
            'Tools': ['Git', 'GitHub', 'GitLab', 'Jira', 'Confluence', 'Postman', 'VS Code', 'IntelliJ']
        }
        
        all_skills = []
        for category, skill_list in skills.items():
            for skill in skill_list:
                all_skills.append({'skill': skill, 'category': category})
        
        return pd.DataFrame(all_skills)
    
    def extract_skills_hybrid(self, text):
        """
        Extract skills using hybrid approach: BiLSTM + keyword matching
        
        Args:
            text: Resume text
            
        Returns:
            List of extracted skills with categories
        """
        extracted_skills = []
        text_lower = text.lower()
        
        # Keyword-based extraction (fallback)
        for _, row in self.skills_db.iterrows():
            skill = row['skill']
            if skill.lower() in text_lower:
                extracted_skills.append({
                    'skill': skill,
                    'category': row.get('category', 'General'),
                    'confidence': 0.9
                })
        
        # BiLSTM-based extraction (if model available)
        if self.model:
            try:
                lstm_skills = self._extract_with_bilstm(text)
                extracted_skills.extend(lstm_skills)
            except Exception as e:
                print(f"BiLSTM extraction failed: {e}")
        
        # Remove duplicates
        unique_skills = {}
        for skill_info in extracted_skills:
            skill_name = skill_info['skill']
            if skill_name not in unique_skills:
                unique_skills[skill_name] = skill_info
        
        return list(unique_skills.values())
    
    def _extract_with_bilstm(self, text):
        """
        Extract skills using BiLSTM model
        
        Args:
            text: Resume text
            
        Returns:
            List of skills extracted by BiLSTM
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.model(inputs['input_ids'], inputs['attention_mask'])
            predictions = torch.argmax(logits, dim=-1)
        
        # Extract skill tokens (label 1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        skill_tokens = [tokens[i] for i in range(len(tokens)) if predictions[0][i] == 1]
        
        # Combine tokens into skills
        skills = []
        current_skill = []
        for token in skill_tokens:
            if token.startswith('##'):
                current_skill.append(token[2:])
            else:
                if current_skill:
                    skills.append({
                        'skill': ''.join(current_skill),
                        'category': 'BiLSTM-Detected',
                        'confidence': 0.85
                    })
                current_skill = [token]
        
        if current_skill:
            skills.append({
                'skill': ''.join(current_skill),
                'category': 'BiLSTM-Detected',
                'confidence': 0.85
            })
        
        return skills
    
    def get_skills_by_category(self, skills):
        """
        Group skills by category
        
        Args:
            skills: List of skill dictionaries
            
        Returns:
            Dictionary with skills grouped by category
        """
        categorized = {}
        for skill_info in skills:
            category = skill_info.get('category', 'General')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(skill_info['skill'])
        
        return categorized
