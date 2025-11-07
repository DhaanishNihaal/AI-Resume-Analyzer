"""
CNN-based Resume Classification Module
Classifies resumes into job categories using Convolutional Neural Networks
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class CNNResumeClassifier(nn.Module):
    def __init__(self, embedding_dim=768, num_filters=128, filter_sizes=[3, 4, 5], num_classes=10, dropout=0.5):
        """
        CNN model for resume classification
        
        Args:
            embedding_dim: Dimension of embeddings
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
            num_classes: Number of job categories
            dropout: Dropout rate
        """
        super(CNNResumeClassifier, self).__init__()
        
        # BERT for embeddings
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout and fully connected layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        Returns:
            Class logits
        """
        # Get BERT embeddings
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state  # [batch_size, seq_len, embedding_dim]
        
        # Transpose for Conv1d: [batch_size, embedding_dim, seq_len]
        embeddings = embeddings.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embeddings))  # [batch_size, num_filters, seq_len - kernel_size + 1]
            pooled = torch.max_pool1d(conv_out, conv_out.shape[2])  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # Concatenate all conv outputs
        concat = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        # Dropout and classification
        concat = self.dropout(concat)
        logits = self.fc(concat)
        
        return logits

class ResumeClassifier:
    def __init__(self):
        """Initialize resume classifier"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Job categories
        self.categories = [
            'Software Developer',
            'Data Scientist',
            'Web Developer',
            'DevOps Engineer',
            'Machine Learning Engineer',
            'Business Analyst',
            'Product Manager',
            'UI/UX Designer',
            'Database Administrator',
            'Network Engineer'
        ]
        
        try:
            # Initialize CNN model
            self.model = CNNResumeClassifier(num_classes=len(self.categories))
            self.model.to(self.device)
            self.model.eval()
            self.tokenizer = self.model.tokenizer
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            self.model = None
    
    def classify_resume(self, text):
        """
        Classify resume into job category
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with predicted category and confidence scores
        """
        if not self.model:
            return self._rule_based_classification(text)
        
        try:
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
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get predictions
            probs = probabilities[0].cpu().numpy()
            predicted_idx = np.argmax(probs)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(probs)[-3:][::-1]
            
            return {
                'predicted_category': self.categories[predicted_idx],
                'confidence': float(probs[predicted_idx]),
                'top_3_predictions': [
                    {
                        'category': self.categories[idx],
                        'confidence': float(probs[idx])
                    }
                    for idx in top_3_idx
                ]
            }
        except Exception as e:
            print(f"CNN classification failed: {e}")
            return self._rule_based_classification(text)
    
    def _rule_based_classification(self, text):
        """
        Fallback rule-based classification
        
        Args:
            text: Resume text
            
        Returns:
            Classification result
        """
        text_lower = text.lower()
        
        # Keywords for each category
        keywords = {
            'Software Developer': ['java', 'python', 'c++', 'software development', 'programming', 'coding'],
            'Data Scientist': ['data science', 'machine learning', 'statistics', 'python', 'r', 'data analysis'],
            'Web Developer': ['html', 'css', 'javascript', 'react', 'angular', 'web development', 'frontend'],
            'DevOps Engineer': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'aws', 'devops', 'terraform'],
            'Machine Learning Engineer': ['tensorflow', 'pytorch', 'deep learning', 'neural networks', 'ml', 'ai'],
            'Business Analyst': ['business analysis', 'requirements', 'stakeholder', 'sql', 'tableau'],
            'Product Manager': ['product management', 'roadmap', 'agile', 'scrum', 'product strategy'],
            'UI/UX Designer': ['ui', 'ux', 'figma', 'sketch', 'design', 'wireframe', 'prototype'],
            'Database Administrator': ['database', 'sql', 'oracle', 'mysql', 'postgresql', 'dba'],
            'Network Engineer': ['network', 'cisco', 'routing', 'switching', 'firewall', 'tcp/ip']
        }
        
        # Count keyword matches
        scores = {}
        for category, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            scores[category] = score
        
        # Get top category
        if max(scores.values()) > 0:
            predicted_category = max(scores, key=scores.get)
            total_keywords = sum(scores.values())
            confidence = scores[predicted_category] / total_keywords if total_keywords > 0 else 0
        else:
            predicted_category = 'General'
            confidence = 0.5
        
        # Get top 3
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'predicted_category': predicted_category,
            'confidence': confidence,
            'top_3_predictions': [
                {
                    'category': cat,
                    'confidence': score / max(scores.values()) if max(scores.values()) > 0 else 0
                }
                for cat, score in sorted_categories
            ]
        }
    
    def get_recommended_skills(self, category):
        """
        Get recommended skills for a job category
        
        Args:
            category: Job category
            
        Returns:
            List of recommended skills
        """
        skill_recommendations = {
            'Software Developer': ['Design Patterns', 'Algorithms', 'Data Structures', 'Git', 'Testing', 'Agile'],
            'Data Scientist': ['Statistics', 'Machine Learning', 'Python', 'R', 'SQL', 'Data Visualization'],
            'Web Developer': ['JavaScript', 'React', 'Node.js', 'REST APIs', 'Responsive Design', 'Git'],
            'DevOps Engineer': ['Docker', 'Kubernetes', 'CI/CD', 'AWS', 'Terraform', 'Monitoring'],
            'Machine Learning Engineer': ['TensorFlow', 'PyTorch', 'Deep Learning', 'MLOps', 'Python', 'Statistics'],
            'Business Analyst': ['SQL', 'Excel', 'Tableau', 'Requirements Analysis', 'Agile', 'Communication'],
            'Product Manager': ['Product Strategy', 'Agile', 'User Research', 'Analytics', 'Roadmapping', 'Communication'],
            'UI/UX Designer': ['Figma', 'Sketch', 'Prototyping', 'User Research', 'Wireframing', 'Design Systems'],
            'Database Administrator': ['SQL', 'Database Design', 'Performance Tuning', 'Backup/Recovery', 'Security'],
            'Network Engineer': ['Cisco', 'TCP/IP', 'Routing', 'Switching', 'Security', 'Troubleshooting']
        }
        
        return skill_recommendations.get(category, ['Communication', 'Problem Solving', 'Teamwork'])
