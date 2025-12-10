"""
Tree-of-Thought (ToT) Reasoning for Medical MCQs

Implements structured multi-branch reasoning for complex medical questions:
1. Identify relevant symptoms
2. Identify potential guidelines
3. Evaluate each option
4. Select the final answer

This structured reasoning often outperforms vanilla CoT for tricky multi-step questions.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.document_processor import Document


@dataclass
class ThoughtBranch:
    """A branch in the tree of thought reasoning."""
    branch_name: str
    reasoning: str
    evidence: List[str]
    confidence: float
    conclusion: Optional[str] = None


@dataclass
class TreeOfThoughtResult:
    """Complete tree-of-thought reasoning result."""
    selected_answer: str
    confidence_score: float
    branches: List[ThoughtBranch]
    reasoning_path: List[str]  # Path through the tree
    final_reasoning: str


class TreeOfThoughtReasoner:
    """
    Tree-of-Thought reasoning engine for complex medical questions.
    
    Breaks down reasoning into structured branches:
    1. Symptom identification branch
    2. Guideline identification branch
    3. Option evaluation branch (one per option)
    4. Final selection branch
    """
    
    def __init__(self, llm_model=None):
        """Initialize ToT reasoner."""
        self.llm_model = llm_model
    
    def reason(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        retrieved_contexts: List[Document],
        num_snippets: int = 5
    ) -> TreeOfThoughtResult:
        """
        Perform tree-of-thought reasoning.
        
        Args:
            question: The question being asked
            case_description: Patient case description
            options: Answer options {A: "text", ...}
            retrieved_contexts: Retrieved medical documents
            num_snippets: Number of context snippets to use
            
        Returns:
            TreeOfThoughtResult with answer and reasoning
        """
        if not self.llm_model:
            # Fallback to simple reasoning if no LLM
            return self._fallback_reasoning(question, case_description, options, retrieved_contexts)
        
        # Prepare context snippets
        context_docs = retrieved_contexts[:num_snippets]
        context_snippets = []
        for idx, doc in enumerate(context_docs, 1):
            snippet = f"""--- SNIPPET {idx} ---
Guideline: {doc.metadata.get('title', doc.metadata.get('guideline_title', 'Unknown'))}
Category: {doc.metadata.get('category', 'Unknown')}
Content: {doc.content[:800]}"""
            context_snippets.append(snippet)
        
        context_text = "\n\n".join(context_snippets)
        
        # Build ToT prompt
        tot_prompt = self._build_tot_prompt(
            question, case_description, options, context_text, num_snippets
        )
        
        # Generate ToT reasoning
        response = self.llm_model.generate(
            prompt=tot_prompt,
            system_prompt=self._get_tot_system_prompt(),
            temperature=0.1,
            max_tokens=1024  # More tokens for ToT reasoning
        )
        
        # Parse ToT response
        return self._parse_tot_response(response, options)
    
    def _build_tot_prompt(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        context_text: str,
        num_snippets: int
    ) -> str:
        """Build Tree-of-Thought reasoning prompt."""
        return f"""Clinical Case:
{case_description}

Question: {question}

Answer Options:
{chr(10).join([f"{label}: {text}" for label, text in options.items()])}

Medical Guidelines Context ({num_snippets} snippets):
{context_text}

TREE-OF-THOUGHT REASONING - Complete each branch systematically:

BRANCH 1: Identify Relevant Symptoms
Analyze the case description and identify:
- Key symptoms mentioned
- Vital signs or lab values
- Patient demographics (age, gender)
- Relevant medical history

Your analysis:
[Analyze symptoms here]

BRANCH 2: Identify Potential Guidelines
Based on the symptoms and question, identify which guidelines are most relevant:
- Review each context snippet
- Identify which guidelines match the symptoms
- Note any guideline names or categories mentioned

Your analysis:
[Identify relevant guidelines here]

BRANCH 3: Evaluate Each Option
For EACH option (A, B, C, D), evaluate systematically:

Option A: "{options.get('A', 'N/A')}"
- Search through ALL snippets for this option
- Check if it's explicitly mentioned
- Check if it's supported by treatment protocols
- Note which snippet(s) support or contradict it
Your evaluation: [Evaluate Option A here]

Option B: "{options.get('B', 'N/A')}"
- Search through ALL snippets for this option
- Check if it's explicitly mentioned
- Check if it's supported by treatment protocols
- Note which snippet(s) support or contradict it
Your evaluation: [Evaluate Option B here]

Option C: "{options.get('C', 'N/A')}"
- Search through ALL snippets for this option
- Check if it's explicitly mentioned
- Check if it's supported by treatment protocols
- Note which snippet(s) support or contradict it
Your evaluation: [Evaluate Option C here]

Option D: "{options.get('D', 'N/A')}"
- Search through ALL snippets for this option
- Check if it's explicitly mentioned
- Check if it's supported by treatment protocols
- Note which snippet(s) support or contradict it
Your evaluation: [Evaluate Option D here]

BRANCH 4: Synthesize and Select
Based on your analysis in Branches 1-3:
- Which symptoms are most relevant to the question?
- Which guidelines provide the best answer?
- Which option has the strongest support across all snippets?
- Which option best matches the treatment protocol for these symptoms?

Your synthesis:
[Synthesize and select here]

FINAL ANSWER:
Based on your tree-of-thought reasoning above, provide your final answer:
- If you can answer: "ANSWER: [A/B/C/D] CONFIDENCE: [0.0-1.0]"
- If you cannot answer: "ANSWER: Cannot answer from the provided context. CONFIDENCE: 0.0"

Provide your reasoning path showing how you moved through the branches:
REASONING_PATH: [Brief summary of your reasoning path through the branches]
"""
    
    def _get_tot_system_prompt(self) -> str:
        """Get system prompt for ToT reasoning."""
        return """You are a highly accurate medical reasoning assistant using Tree-of-Thought (ToT) reasoning. Your task is to answer multiple-choice clinical questions by systematically reasoning through structured branches.

TREE-OF-THOUGHT METHODOLOGY:
1. Break down the problem into structured branches
2. Reason through each branch systematically
3. Synthesize information from all branches
4. Select the best answer based on the complete reasoning tree

CRITICAL RULES:
- ONLY use information from the provided context snippets
- Complete EACH branch before moving to the next
- Be explicit about what you find (or don't find) in each branch
- Synthesize information from all branches before selecting an answer
- If context doesn't support any option clearly, say "Cannot answer from the provided context."

Your reasoning should be systematic, explicit, and based solely on the provided context."""
    
    def _parse_tot_response(self, response: str, options: Dict[str, str]) -> TreeOfThoughtResult:
        """Parse ToT response into structured result."""
        import re
        
        # Extract final answer
        cannot_answer = "cannot answer from the provided context" in response.lower()
        if cannot_answer:
            return TreeOfThoughtResult(
                selected_answer="Cannot answer from the provided context.",
                confidence_score=0.0,
                branches=[],
                reasoning_path=[],
                final_reasoning=response
            )
        
        # Extract answer
        answer_match = re.search(r'ANSWER:\s*([A-D])', response, re.IGNORECASE)
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response, re.IGNORECASE)
        
        selected_answer = answer_match.group(1).upper() if answer_match else None
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        
        if not selected_answer or selected_answer not in options:
            selected_answer = "Cannot answer from the provided context."
            confidence = 0.0
        
        # Extract reasoning path
        reasoning_path_match = re.search(r'REASONING_PATH:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        reasoning_path = []
        if reasoning_path_match:
            path_text = reasoning_path_match.group(1).strip()
            reasoning_path = [line.strip() for line in path_text.split('\n') if line.strip()]
        
        # Create branches from response (simplified parsing)
        branches = []
        branch_patterns = [
            (r'BRANCH 1[:\s]+(.+?)(?=BRANCH 2|BRANCH 3|BRANCH 4|FINAL)', 'Symptom Identification'),
            (r'BRANCH 2[:\s]+(.+?)(?=BRANCH 3|BRANCH 4|FINAL)', 'Guideline Identification'),
            (r'BRANCH 3[:\s]+(.+?)(?=BRANCH 4|FINAL)', 'Option Evaluation'),
            (r'BRANCH 4[:\s]+(.+?)(?=FINAL|ANSWER)', 'Synthesis and Selection')
        ]
        
        for pattern, name in branch_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                branch_text = match.group(1).strip()
                branches.append(ThoughtBranch(
                    branch_name=name,
                    reasoning=branch_text[:500],  # Limit length
                    evidence=[],
                    confidence=0.5
                ))
        
        return TreeOfThoughtResult(
            selected_answer=selected_answer,
            confidence_score=confidence,
            branches=branches,
            reasoning_path=reasoning_path,
            final_reasoning=response
        )
    
    def _fallback_reasoning(
        self,
        question: str,
        case_description: str,
        options: Dict[str, str],
        retrieved_contexts: List[Document]
    ) -> TreeOfThoughtResult:
        """Fallback reasoning when LLM is not available."""
        # Simple rule-based fallback
        return TreeOfThoughtResult(
            selected_answer="Cannot answer from the provided context.",
            confidence_score=0.0,
            branches=[],
            reasoning_path=["LLM not available for ToT reasoning"],
            final_reasoning="Tree-of-Thought reasoning requires LLM model."
        )


def main():
    """Demo: Test Tree-of-Thought reasoning."""
    print("="*70)
    print("TREE-OF-THOUGHT REASONING DEMO")
    print("="*70)
    print("\n[INFO] Tree-of-Thought reasoner initialized")
    print("[INFO] Ready to process complex multi-step medical questions")


if __name__ == "__main__":
    main()

