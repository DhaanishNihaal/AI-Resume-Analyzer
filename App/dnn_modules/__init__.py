"""
Deep Neural Network Modules for AI Resume Analyzer
Includes BERT-based entity extraction, BiLSTM skill extraction, 
CNN resume classification, and neural recommendation system
"""

from .bert_entity_extractor import BERTEntityExtractor
from .bilstm_skill_extractor import SkillExtractor
from .cnn_resume_classifier import ResumeClassifier
from .neural_recommender import NeuralRecommender

__all__ = [
    'BERTEntityExtractor',
    'SkillExtractor',
    'ResumeClassifier',
    'NeuralRecommender'
]
