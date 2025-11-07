"""
Neural Embedding-based Recommendation System
Uses sentence transformers for semantic similarity-based recommendations
"""

import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

class NeuralRecommender:
    def __init__(self):
        """Initialize neural recommender with sentence transformer"""
        try:
            # Load pre-trained sentence transformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            self.model = None
        
        # Job descriptions database
        self.job_database = self._get_job_database()
        
        # Course recommendations database
        self.course_database = self._get_course_database()
        
        # Pre-compute embeddings for jobs and courses
        if self.model:
            self._precompute_embeddings()
    
    def _get_job_database(self):
        """Get job descriptions database"""
        return {
            'Software Developer': {
                'description': 'Develop and maintain software applications using programming languages like Python, Java, C++. Work on algorithms, data structures, and software design patterns.',
                'skills': ['Python', 'Java', 'C++', 'Git', 'Algorithms', 'Data Structures'],
                'experience': 'Entry to Senior level'
            },
            'Data Scientist': {
                'description': 'Analyze complex data sets using statistical methods and machine learning. Build predictive models and provide data-driven insights.',
                'skills': ['Python', 'R', 'Machine Learning', 'Statistics', 'SQL', 'Data Visualization'],
                'experience': 'Mid to Senior level'
            },
            'Web Developer': {
                'description': 'Design and develop web applications using HTML, CSS, JavaScript, and modern frameworks like React, Angular, or Vue.js.',
                'skills': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js', 'REST APIs'],
                'experience': 'Entry to Mid level'
            },
            'Machine Learning Engineer': {
                'description': 'Design and implement machine learning models and systems. Work with TensorFlow, PyTorch, and deploy ML models to production.',
                'skills': ['TensorFlow', 'PyTorch', 'Python', 'Deep Learning', 'MLOps', 'Cloud'],
                'experience': 'Mid to Senior level'
            },
            'DevOps Engineer': {
                'description': 'Manage CI/CD pipelines, containerization, and cloud infrastructure. Automate deployment and monitoring processes.',
                'skills': ['Docker', 'Kubernetes', 'AWS', 'Jenkins', 'Terraform', 'Linux'],
                'experience': 'Mid to Senior level'
            }
        }
    
    def _get_course_database(self):
        """Get course recommendations database"""
        return {
            'Python Programming': {
                'description': 'Learn Python programming from basics to advanced concepts including OOP, data structures, and algorithms.',
                'platform': 'Coursera, Udemy',
                'duration': '4-6 weeks',
                'level': 'Beginner to Intermediate'
            },
            'Machine Learning': {
                'description': 'Comprehensive machine learning course covering supervised and unsupervised learning, neural networks, and deep learning.',
                'platform': 'Coursera, edX',
                'duration': '8-12 weeks',
                'level': 'Intermediate to Advanced'
            },
            'Web Development': {
                'description': 'Full-stack web development covering HTML, CSS, JavaScript, React, Node.js, and databases.',
                'platform': 'Udemy, freeCodeCamp',
                'duration': '12-16 weeks',
                'level': 'Beginner to Intermediate'
            },
            'Data Science': {
                'description': 'Complete data science bootcamp covering statistics, Python, machine learning, and data visualization.',
                'platform': 'Coursera, DataCamp',
                'duration': '10-14 weeks',
                'level': 'Intermediate'
            },
            'Cloud Computing': {
                'description': 'Learn AWS, Azure, or GCP cloud platforms, including compute, storage, networking, and serverless architectures.',
                'platform': 'AWS Training, Coursera',
                'duration': '6-8 weeks',
                'level': 'Intermediate'
            },
            'DevOps': {
                'description': 'Master DevOps practices including CI/CD, Docker, Kubernetes, infrastructure as code, and monitoring.',
                'platform': 'Udemy, Linux Academy',
                'duration': '8-10 weeks',
                'level': 'Intermediate to Advanced'
            },
            'Deep Learning': {
                'description': 'Advanced deep learning course covering CNNs, RNNs, Transformers, and modern architectures.',
                'platform': 'Coursera, Fast.ai',
                'duration': '10-12 weeks',
                'level': 'Advanced'
            }
        }
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for jobs and courses"""
        # Job embeddings
        job_texts = [f"{job}: {info['description']}" for job, info in self.job_database.items()]
        self.job_embeddings = self.model.encode(job_texts, convert_to_tensor=True)
        self.job_names = list(self.job_database.keys())
        
        # Course embeddings
        course_texts = [f"{course}: {info['description']}" for course, info in self.course_database.items()]
        self.course_embeddings = self.model.encode(course_texts, convert_to_tensor=True)
        self.course_names = list(self.course_database.keys())
    
    def recommend_jobs(self, resume_text, top_k=5):
        """
        Recommend jobs based on resume content using semantic similarity
        
        Args:
            resume_text: Resume text
            top_k: Number of recommendations
            
        Returns:
            List of recommended jobs with similarity scores
        """
        if not self.model:
            return []
        
        try:
            # Encode resume
            resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarities = util.cos_sim(resume_embedding, self.job_embeddings)[0]
            
            # Get top-k jobs
            top_indices = torch.topk(similarities, k=min(top_k, len(self.job_names))).indices
            
            recommendations = []
            for idx in top_indices:
                job_name = self.job_names[idx]
                similarity = similarities[idx].item()
                
                recommendations.append({
                    'job_title': job_name,
                    'similarity_score': similarity,
                    'match_percentage': round(similarity * 100, 2),
                    'description': self.job_database[job_name]['description'],
                    'required_skills': self.job_database[job_name]['skills'],
                    'experience_level': self.job_database[job_name]['experience']
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in job recommendation: {e}")
            return []
    
    def recommend_courses(self, resume_text, skills_gap=None, top_k=5):
        """
        Recommend courses based on resume and skills gap
        
        Args:
            resume_text: Resume text
            skills_gap: List of missing skills
            top_k: Number of recommendations
            
        Returns:
            List of recommended courses
        """
        if not self.model:
            return []
        
        try:
            # Create query text
            if skills_gap:
                query_text = f"{resume_text} Need to learn: {', '.join(skills_gap)}"
            else:
                query_text = resume_text
            
            # Encode query
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            
            # Calculate similarity
            similarities = util.cos_sim(query_embedding, self.course_embeddings)[0]
            
            # Get top-k courses
            top_indices = torch.topk(similarities, k=min(top_k, len(self.course_names))).indices
            
            recommendations = []
            for idx in top_indices:
                course_name = self.course_names[idx]
                similarity = similarities[idx].item()
                
                recommendations.append({
                    'course_name': course_name,
                    'relevance_score': similarity,
                    'match_percentage': round(similarity * 100, 2),
                    'description': self.course_database[course_name]['description'],
                    'platform': self.course_database[course_name]['platform'],
                    'duration': self.course_database[course_name]['duration'],
                    'level': self.course_database[course_name]['level']
                })
            
            return recommendations
        except Exception as e:
            print(f"Error in course recommendation: {e}")
            return []
    
    def find_skill_gaps(self, current_skills, target_job):
        """
        Find skill gaps between current skills and target job requirements
        
        Args:
            current_skills: List of current skills
            target_job: Target job title
            
        Returns:
            List of missing skills
        """
        if target_job not in self.job_database:
            return []
        
        required_skills = self.job_database[target_job]['skills']
        current_skills_lower = [skill.lower() for skill in current_skills]
        
        missing_skills = [
            skill for skill in required_skills
            if skill.lower() not in current_skills_lower
        ]
        
        return missing_skills
    
    def calculate_job_match_score(self, resume_text, job_title):
        """
        Calculate detailed match score for a specific job
        
        Args:
            resume_text: Resume text
            job_title: Job title to match against
            
        Returns:
            Dictionary with detailed match information
        """
        if not self.model or job_title not in self.job_database:
            return None
        
        try:
            # Encode texts
            resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
            job_idx = self.job_names.index(job_title)
            job_embedding = self.job_embeddings[job_idx]
            
            # Calculate similarity
            similarity = util.cos_sim(resume_embedding, job_embedding).item()
            
            return {
                'job_title': job_title,
                'overall_match': round(similarity * 100, 2),
                'description': self.job_database[job_title]['description'],
                'required_skills': self.job_database[job_title]['skills'],
                'recommendation': self._get_match_recommendation(similarity)
            }
        except Exception as e:
            print(f"Error calculating match score: {e}")
            return None
    
    def _get_match_recommendation(self, similarity):
        """Get recommendation based on similarity score"""
        if similarity >= 0.8:
            return "Excellent match! You should definitely apply."
        elif similarity >= 0.6:
            return "Good match! Consider applying after reviewing requirements."
        elif similarity >= 0.4:
            return "Moderate match. You may need to upskill in some areas."
        else:
            return "Low match. Consider gaining more relevant experience and skills."
