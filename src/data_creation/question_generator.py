"""
Question Generator Module
==========================

PURPOSE:
    Generate clinical case questions from extracted medical guidelines.
    Creates 100 MCQ questions with different types and difficulty levels.

TECHNICAL DETAILS:
    - Generates questions directly from guideline content
    - Ensures distribution: 40 diagnosis, 30 treatment, 30 management
    - Difficulty levels: 30 easy, 40 medium, 30 hard
    - Each question includes reasoning and ground truth evidence

RESEARCH BASIS:
    - Questions derived from source material improve evaluation validity
    - Varied difficulty ensures comprehensive system testing
"""

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates clinical case questions from medical guidelines.
    
    Creates multiple-choice questions with varying difficulty and types,
    ensuring each question is grounded in the source guidelines.
    
    Attributes:
        ollama_model: Name of the Ollama model to use
        output_dir: Directory to save generated questions
    """
    
    # Question type distribution
    QUESTION_TYPES = {
        'diagnosis': 40,      # Questions about identifying conditions
        'treatment': 30,      # Questions about treatment choices
        'management': 30      # Questions about patient management
    }
    
    # Difficulty distribution
    DIFFICULTY_LEVELS = {
        'easy': 30,
        'medium': 40,
        'hard': 30
    }
    
    def __init__(self,
                 ollama_model: str = "llama3.1:8b",
                 output_dir: str = "data/questions",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the QuestionGenerator.
        
        Args:
            ollama_model: Ollama model name
            output_dir: Directory to save questions
            ollama_host: Ollama API host URL
        """
        self.ollama_model = ollama_model
        self.output_dir = Path(output_dir)
        self.ollama_host = ollama_host
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Call Ollama API for text generation."""
        try:
            import ollama
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "num_predict": 1500,
                }
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def _create_question_distribution(self, total_questions: int = 100) -> List[Dict]:
        """
        Create the distribution of questions across types and difficulties.
        
        Returns:
            List of question specifications with type and difficulty
        """
        distribution = []
        
        # Calculate questions per type
        for q_type, type_count in self.QUESTION_TYPES.items():
            # Distribute difficulties within each type
            type_questions = []
            remaining = type_count
            
            for difficulty, diff_ratio in self.DIFFICULTY_LEVELS.items():
                # Proportional distribution
                count = int(type_count * (diff_ratio / 100))
                type_questions.extend([
                    {'type': q_type, 'difficulty': difficulty}
                    for _ in range(count)
                ])
                remaining -= count
            
            # Add any remaining to medium difficulty
            type_questions.extend([
                {'type': q_type, 'difficulty': 'medium'}
                for _ in range(remaining)
            ])
            
            distribution.extend(type_questions)
        
        # Shuffle to mix types and difficulties
        random.shuffle(distribution)
        
        return distribution[:total_questions]
    
    def generate_question(self, 
                          guideline: Dict, 
                          question_spec: Dict,
                          question_id: int) -> Dict:
        """
        Generate a single clinical case question from a guideline.
        
        Args:
            guideline: Guideline dictionary with 'name' and 'content'
            question_spec: Specification with 'type' and 'difficulty'
            question_id: Unique question ID
            
        Returns:
            Dictionary containing the complete question
        """
        q_type = question_spec['type']
        difficulty = question_spec['difficulty']
        guideline_name = guideline.get('name', 'Medical Guideline')
        guideline_content = guideline.get('content', '')
        
        # Difficulty-specific instructions
        difficulty_instructions = {
            'easy': """Create a straightforward case with classic presentation. 
The correct answer should be clearly supported by the guidelines.
Use a typical patient with no complicating factors.""",
            
            'medium': """Create a case with some complexity. 
Include one or two atypical features or comorbidities.
The correct answer requires integrating multiple pieces of information.""",
            
            'hard': """Create a challenging case with subtle findings or complications.
Include distractors that could mislead if guidelines aren't carefully applied.
May involve contraindications, drug interactions, or special populations."""
        }
        
        # Question type instructions
        type_instructions = {
            'diagnosis': """Focus on: identifying the correct diagnosis based on presentation.
Include relevant history, symptoms, vital signs, and/or test results.
Distractors should be plausible differential diagnoses.""",
            
            'treatment': """Focus on: selecting the most appropriate treatment.
Present a patient with an established diagnosis.
Include specific medication choices, dosages, or treatment modalities.
Distractors should be treatments for similar conditions or contraindicated options.""",
            
            'management': """Focus on: appropriate next steps in patient management.
May include: monitoring, follow-up, lifestyle modifications, or escalation criteria.
Test understanding of the overall management approach, not just treatment."""
        }
        
        system_prompt = """You are a medical educator creating clinical case questions.
Create realistic patient scenarios based on the provided guidelines.
Ensure questions test understanding of the guidelines and are clinically relevant.
All information in the question must be derivable from the guidelines."""

        prompt = f"""Based on the following medical guideline, create a clinical case question.

GUIDELINE: {guideline_name}
---
{guideline_content[:3000]}  
---

QUESTION REQUIREMENTS:
- Type: {q_type.upper()} question
- Difficulty: {difficulty.upper()}

{difficulty_instructions[difficulty]}

{type_instructions[q_type]}

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS (use this exact structure):

CASE:
[Write a detailed patient scenario including demographics, chief complaint, history, 
vital signs, and relevant findings. 2-4 sentences.]

QUESTION:
[Write a clear, specific question about {q_type}]

OPTIONS:
A. [First option]
B. [Second option]
C. [Third option]
D. [Fourth option]

CORRECT_ANSWER: [Single letter A, B, C, or D]

REASONING:
[Explain why the correct answer is right and why others are wrong. 
Reference specific guideline recommendations. 2-3 sentences.]

EVIDENCE:
[Quote or paraphrase the specific guideline section that supports the answer]
"""

        try:
            response = self._call_ollama(prompt, system_prompt)
            
            # Parse the response
            parsed = self._parse_question_response(response, question_id)
            
            if parsed:
                parsed['id'] = question_id
                parsed['type'] = q_type
                parsed['difficulty'] = difficulty
                parsed['source_guideline'] = guideline_name
                parsed['source_guideline_id'] = guideline.get('id', 0)
                return parsed
            else:
                logger.warning(f"Failed to parse question {question_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating question {question_id}: {e}")
            return None
    
    def _parse_question_response(self, response: str, question_id: int) -> Optional[Dict]:
        """
        Parse the LLM response into a structured question dictionary.
        
        Args:
            response: Raw LLM response text
            question_id: Question ID for error reporting
            
        Returns:
            Parsed question dictionary or None if parsing fails
        """
        try:
            # Extract each section using regex
            case_match = re.search(r'CASE:\s*\n?(.*?)(?=\n\s*QUESTION:)', response, re.DOTALL | re.IGNORECASE)
            question_match = re.search(r'QUESTION:\s*\n?(.*?)(?=\n\s*OPTIONS:)', response, re.DOTALL | re.IGNORECASE)
            options_match = re.search(r'OPTIONS:\s*\n?(.*?)(?=\n\s*CORRECT_ANSWER:)', response, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r'CORRECT_ANSWER:\s*\n?\s*([A-Da-d])', response, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*\n?(.*?)(?=\n\s*EVIDENCE:)', response, re.DOTALL | re.IGNORECASE)
            evidence_match = re.search(r'EVIDENCE:\s*\n?(.*?)$', response, re.DOTALL | re.IGNORECASE)
            
            if not all([case_match, question_match, options_match, answer_match]):
                logger.warning(f"Question {question_id}: Missing required sections")
                return None
            
            # Parse options
            options_text = options_match.group(1).strip()
            options = {}
            for letter in ['A', 'B', 'C', 'D']:
                opt_match = re.search(rf'{letter}[\.\)]\s*(.*?)(?=[B-D][\.\)]|$)', options_text, re.DOTALL | re.IGNORECASE)
                if opt_match:
                    options[letter] = opt_match.group(1).strip()
            
            if len(options) < 4:
                # Try alternative parsing
                opt_lines = [l.strip() for l in options_text.split('\n') if l.strip()]
                for line in opt_lines:
                    for letter in ['A', 'B', 'C', 'D']:
                        if line.upper().startswith(letter):
                            options[letter] = re.sub(r'^[A-D][\.\)]\s*', '', line, flags=re.IGNORECASE)
            
            return {
                'case': case_match.group(1).strip(),
                'question': question_match.group(1).strip(),
                'options': options,
                'correct_answer': answer_match.group(1).upper(),
                'reasoning': reasoning_match.group(1).strip() if reasoning_match else '',
                'evidence': evidence_match.group(1).strip() if evidence_match else ''
            }
            
        except Exception as e:
            logger.error(f"Error parsing question {question_id}: {e}")
            return None
    
    def generate_questions(self, 
                           guidelines: List[Dict], 
                           total_questions: int = 100,
                           save: bool = True) -> List[Dict]:
        """
        Generate all questions from guidelines.
        
        Args:
            guidelines: List of guideline dictionaries
            total_questions: Total number of questions to generate
            save: Whether to save questions to files
            
        Returns:
            List of generated question dictionaries
        """
        if not guidelines:
            raise ValueError("No guidelines provided")
        
        # Create question distribution
        distribution = self._create_question_distribution(total_questions)
        
        # Distribute questions across guidelines
        questions_per_guideline = total_questions // len(guidelines)
        extra_questions = total_questions % len(guidelines)
        
        questions = []
        question_id = 1
        
        logger.info(f"Generating {total_questions} questions from {len(guidelines)} guidelines...")
        
        # Assign questions to guidelines
        guideline_assignments = []
        for i, guideline in enumerate(guidelines):
            count = questions_per_guideline + (1 if i < extra_questions else 0)
            guideline_assignments.extend([(guideline, distribution[len(guideline_assignments) + j]) 
                                           for j in range(count)])
        
        # Shuffle to mix guidelines
        random.shuffle(guideline_assignments)
        
        # Generate questions
        for guideline, question_spec in tqdm(guideline_assignments, desc="Generating questions"):
            question = self.generate_question(guideline, question_spec, question_id)
            
            if question:
                questions.append(question)
                question_id += 1
            
            # Stop if we have enough
            if len(questions) >= total_questions:
                break
        
        # If we don't have enough, generate more from random guidelines
        attempts = 0
        while len(questions) < total_questions and attempts < 20:
            guideline = random.choice(guidelines)
            spec = random.choice(distribution)
            question = self.generate_question(guideline, spec, question_id)
            if question:
                questions.append(question)
                question_id += 1
            attempts += 1
        
        logger.info(f"Generated {len(questions)} questions")
        
        if save:
            self._save_questions(questions)
            self._create_train_test_split(questions)
        
        return questions
    
    def _save_questions(self, questions: List[Dict]):
        """Save all questions to a JSON file."""
        # Save complete questions
        questions_path = self.output_dir / 'all_questions.json'
        with open(questions_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(questions)} questions to: {questions_path}")
        
        # Save statistics
        stats = self._compute_statistics(questions)
        stats_path = self.output_dir / 'questions_statistics.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to: {stats_path}")
    
    def _compute_statistics(self, questions: List[Dict]) -> Dict:
        """Compute statistics about generated questions."""
        stats = {
            'total_questions': len(questions),
            'by_type': {},
            'by_difficulty': {},
            'by_guideline': {},
            'by_correct_answer': {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        }
        
        for q in questions:
            # By type
            q_type = q.get('type', 'unknown')
            stats['by_type'][q_type] = stats['by_type'].get(q_type, 0) + 1
            
            # By difficulty
            diff = q.get('difficulty', 'unknown')
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1
            
            # By guideline
            guideline = q.get('source_guideline', 'unknown')
            stats['by_guideline'][guideline] = stats['by_guideline'].get(guideline, 0) + 1
            
            # By answer
            answer = q.get('correct_answer', 'X')
            if answer in stats['by_correct_answer']:
                stats['by_correct_answer'][answer] += 1
        
        return stats
    
    def _create_train_test_split(self, questions: List[Dict], test_ratio: float = 0.2):
        """
        Create train/test split of questions (80/20).
        
        Args:
            questions: List of all questions
            test_ratio: Ratio of questions for test set
        """
        # Shuffle questions
        shuffled = questions.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * (1 - test_ratio))
        train_questions = shuffled[:split_idx]
        test_questions = shuffled[split_idx:]
        
        # Create splits directory
        splits_dir = self.output_dir.parent / 'splits'
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train set
        train_path = splits_dir / 'train.json'
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_questions, f, indent=2, ensure_ascii=False)
        
        # Save test set
        test_path = splits_dir / 'test.json'
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_questions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created train/test split: {len(train_questions)} train, {len(test_questions)} test")
        
        # Save split info
        split_info = {
            'train_count': len(train_questions),
            'test_count': len(test_questions),
            'test_ratio': test_ratio,
            'train_file': str(train_path),
            'test_file': str(test_path)
        }
        
        info_path = splits_dir / 'split_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2)
    
    def load_questions(self, split: str = 'all') -> List[Dict]:
        """
        Load previously generated questions.
        
        Args:
            split: 'all', 'train', or 'test'
            
        Returns:
            List of question dictionaries
        """
        if split == 'all':
            filepath = self.output_dir / 'all_questions.json'
        else:
            filepath = self.output_dir.parent / 'splits' / f'{split}.json'
        
        if not filepath.exists():
            raise FileNotFoundError(f"Questions file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def generate_questions_from_guidelines(guidelines_dir: str, 
                                        output_dir: str = "data/questions",
                                        total_questions: int = 100) -> List[Dict]:
    """
    Convenience function to generate questions from guideline files.
    
    Args:
        guidelines_dir: Directory containing guideline text files
        output_dir: Directory to save questions
        total_questions: Number of questions to generate
        
    Returns:
        List of generated question dictionaries
    """
    from .guideline_generator import GuidelineGenerator
    
    # Load guidelines
    gg = GuidelineGenerator(output_dir=guidelines_dir)
    guidelines = gg.load_guidelines()
    
    if not guidelines:
        raise ValueError(f"No guidelines found in {guidelines_dir}")
    
    # Generate questions
    qg = QuestionGenerator(output_dir=output_dir)
    questions = qg.generate_questions(guidelines, total_questions=total_questions)
    
    return questions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        guidelines_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/questions"
        
        questions = generate_questions_from_guidelines(guidelines_dir, output_dir)
        print(f"Generated {len(questions)} questions")
    else:
        print("Usage: python question_generator.py <guidelines_dir> [output_dir]")

