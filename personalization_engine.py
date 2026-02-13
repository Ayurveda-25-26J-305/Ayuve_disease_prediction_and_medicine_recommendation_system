"""
Personalization Engine for Answer Refinement
Implements Option 2: Personalized Answer Refinement based on Prakriti

Features:
- Prakriti (Dosha constitution) assessment
- User profile management
- Seasonal adaptation (Ritu)
- Geographic context (Sri Lankan Ayurveda)
- Answer personalization based on individual constitution
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Dosha characteristics and balancing guidelines
DOSHA_GUIDELINES = {
    'vata': {
        'characteristics': ['dry', 'cold', 'light', 'mobile', 'rough'],
        'imbalance_signs': ['anxiety', 'dry skin', 'constipation', 'irregular digestion', 'insomnia'],
        'balance_foods': ['warm', 'moist', 'grounding', 'oily', 'sweet'],
        'avoid_foods': ['raw', 'cold', 'dry', 'bitter', 'astringent'],
        'favor_tastes': ['sweet', 'sour', 'salty'],
        'lifestyle': ['regular routine', 'warm environments', 'calming activities'],
    },
    'pitta': {
        'characteristics': ['hot', 'sharp', 'oily', 'light', 'spreading'],
        'imbalance_signs': ['irritability', 'inflammation', 'acidity', 'skin rashes', 'burning sensation'],
        'balance_foods': ['cool', 'mild', 'grounding', 'sweet', 'bitter'],
        'avoid_foods': ['spicy', 'sour', 'salty', 'hot', 'oily'],
        'favor_tastes': ['sweet', 'bitter', 'astringent'],
        'lifestyle': ['cooling environments', 'moderate exercise', 'relaxation'],
    },
    'kapha': {
        'characteristics': ['heavy', 'slow', 'cool', 'oily', 'stable'],
        'imbalance_signs': ['lethargy', 'weight gain', 'congestion', 'slow digestion', 'depression'],
        'balance_foods': ['light', 'warm', 'dry', 'stimulating', 'pungent'],
        'avoid_foods': ['heavy', 'oily', 'sweet', 'cold', 'dairy'],
        'favor_tastes': ['pungent', 'bitter', 'astringent'],
        'lifestyle': ['vigorous exercise', 'stimulation', 'variety'],
    }
}


# Seasonal guidelines for Sri Lanka
SEASONAL_GUIDELINES = {
    'winter': {
        'months': [12, 1, 2],
        'predominant_dosha': 'vata',
        'general_advice': 'Favor warm, nourishing foods and stay warm',
        'sri_lanka_context': 'Cool season in Sri Lanka, but still mild compared to other regions'
    },
    'spring': {
        'months': [3, 4, 5],
        'predominant_dosha': 'kapha',
        'general_advice': 'Favor light, dry, stimulating foods to counter spring moisture',
        'sri_lanka_context': 'Inter-monsoon period with occasional showers'
    },
    'monsoon': {
        'months': [6, 7, 8],
        'predominant_dosha': 'vata',
        'general_advice': 'Support digestion with warm, easily digestible foods',
        'sri_lanka_context': 'Southwest monsoon brings heavy rains, affecting digestion'
    },
    'fall': {
        'months': [9, 10, 11],
        'predominant_dosha': 'pitta',
        'general_advice': 'Favor cooling foods to balance accumulated summer heat',
        'sri_lanka_context': 'Inter-monsoon period, warm and humid'
    }
}


class PersonalizationEngine:
    """
    Personalization engine for tailoring Ayurvedic answers to individual constitution
    """
    
    def __init__(self, llm_generator=None, profiles_dir: str = "./user_profiles"):
        """
        Initialize personalization engine
        
        Args:
            llm_generator: LLM instance for generating personalized text
            profiles_dir: Directory to store user profiles
        """
        self.llm_generator = llm_generator
        self.profiles_dir = profiles_dir
        
        # Create profiles directory if it doesn't exist
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        logger.info("PersonalizationEngine initialized")
    
    def calculate_prakriti_from_responses(self, responses: Dict[str, str]) -> Dict[str, int]:
        """
        Calculate Prakriti scores from questionnaire responses
        
        Args:
            responses: Dictionary mapping question_id to answer choice ('A', 'B', or 'C')
                      A = Vata, B = Pitta, C = Kapha
        
        Returns:
            Dictionary with dosha scores
        """
        vata_score = 0
        pitta_score = 0
        kapha_score = 0
        
        for question_id, answer in responses.items():
            if answer == 'A':
                vata_score += 1
            elif answer == 'B':
                pitta_score += 1
            elif answer == 'C':
                kapha_score += 1
        
        return {
            'vata': vata_score,
            'pitta': pitta_score,
            'kapha': kapha_score
        }
    
    def get_dominant_dosha(self, prakriti_scores: Dict[str, int]) -> str:
        """
        Determine dominant dosha from scores
        
        Args:
            prakriti_scores: Dictionary with vata, pitta, kapha scores
            
        Returns:
            Name of dominant dosha or combined type (e.g., 'vata-pitta')
        """
        sorted_doshas = sorted(
            prakriti_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        highest = sorted_doshas[0]
        second_highest = sorted_doshas[1]
        
        # If two doshas are very close (within 1 point), it's a dual constitution
        if abs(highest[1] - second_highest[1]) <= 1:
            return f"{highest[0]}-{second_highest[0]}"
        else:
            return highest[0]
    
    def get_current_season(self) -> str:
        """
        Determine current season based on month
        
        Returns:
            Season name
        """
        current_month = datetime.now().month
        
        for season, info in SEASONAL_GUIDELINES.items():
            if current_month in info['months']:
                return season
        
        return 'spring'  # Default fallback
    
    def create_user_profile(
        self, 
        user_id: str,
        prakriti_responses: Dict[str, str],
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create or update a user profile
        
        Args:
            user_id: Unique identifier for user
            prakriti_responses: Questionnaire responses
            preferences: Optional dietary/health preferences
            
        Returns:
            User profile dictionary
        """
        # Calculate Prakriti
        prakriti_scores = self.calculate_prakriti_from_responses(prakriti_responses)
        dominant_dosha = self.get_dominant_dosha(prakriti_scores)
        
        # Get current season
        current_season = self.get_current_season()
        
        # Create profile
        profile = {
            'user_id': user_id,
            'prakriti': prakriti_scores,
            'dominant_dosha': dominant_dosha,
            'current_season': current_season,
            'location': 'sri_lanka',
            'preferences': preferences or {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Save profile
        self.save_user_profile(user_id, profile)
        
        logger.info(f"Created profile for user {user_id}: {dominant_dosha} constitution")
        
        return profile
    
    def save_user_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """
        Save user profile to file
        
        Args:
            user_id: User identifier
            profile: Profile dictionary
        """
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Saved profile for user {user_id}")
    
    def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load user profile from file
        
        Args:
            user_id: User identifier
            
        Returns:
            Profile dictionary or None if not found
        """
        profile_path = os.path.join(self.profiles_dir, f"{user_id}.json")
        
        if not os.path.exists(profile_path):
            return None
        
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        logger.info(f"Loaded profile for user {user_id}")
        
        return profile
    
    def build_personalization_context(self, user_profile: Dict[str, Any]) -> str:
        """
        Build context string describing user's constitution for prompting
        
        Args:
            user_profile: User profile dictionary
            
        Returns:
            Formatted context string
        """
        dominant_dosha = user_profile['dominant_dosha']
        season = user_profile.get('current_season', 'spring')
        
        # Get dosha guidelines
        if '-' in dominant_dosha:
            # Dual constitution
            doshas = dominant_dosha.split('-')
            dosha_info = f"dual {doshas[0]}-{doshas[1]} constitution"
        else:
            dosha_info = f"{dominant_dosha} constitution"
        
        context = f"""
User Profile:
- Dominant Dosha: {dominant_dosha} ({dosha_info})
- Current Season: {season.title()}
- Location: Sri Lanka (tropical climate)
""".strip()
        
        return context
    
    def personalize_answer(
        self, 
        base_answer: str, 
        user_profile: Dict[str, Any],
        question: str = ""
    ) -> str:
        """
        Personalize answer with LLM-generated seasonal tips
        """
        dominant_dosha = user_profile['dominant_dosha']
        season = user_profile.get('current_season', 'spring')
        
        # Generate dynamic personalization using LLM
        print("✨ Generating personalized tips...")
        return self._llm_based_personalization(base_answer, user_profile, question)
    
    def _llm_based_personalization(
        self,
        base_answer: str,
        user_profile: Dict[str, Any],
        question: str
    ) -> str:
        """
        LLM-generated personalization based on question context
        """
        dominant_dosha = user_profile['dominant_dosha']
        season = user_profile.get('current_season', 'spring')
        
        # Generate context-aware personalization prompt
        personalization_prompt = f"""<|system|>You are an Ayurvedic expert. Add brief personalized tips for this specific question.<|end|>
<|user|>Base Answer: {base_answer[:500]}

Question Context: {question}
User Constitution: {dominant_dosha.capitalize()}
Current Season: {season.capitalize()}

Add 2-3 sentence personalized tip specifically for {dominant_dosha.capitalize()} constitution related to this question. Be concise and relevant to the question asked.<|end|>
<|assistant|>"""
        
        try:
            personalized_tip = self.llm_generator.generate(personalization_prompt, max_new_tokens=100)
            # Clean up
            personalized_tip = personalized_tip.strip()
            for token in ['<|system|>', '<|user|>', '<|assistant|>', '<|end|>']:
                personalized_tip = personalized_tip.replace(token, '')
            
            return f"{base_answer}\n\n**Personalized for {dominant_dosha.capitalize()} Constitution:**\n{personalized_tip}"
        except:
            # Fallback to template if LLM fails
            return self._fallback_personalization(base_answer, user_profile)
    
    def _fallback_personalization(
        self, 
        base_answer: str, 
        user_profile: Dict[str, Any]
    ) -> str:
        """
        Template-based personalization fallback (when LLM not available)
        
        Args:
            base_answer: Original answer
            user_profile: User profile
            
        Returns:
            Personalized answer with templates
        """
        dominant_dosha = user_profile['dominant_dosha']
        season = user_profile.get('current_season', 'spring')
        
        # Get guidelines
        if '-' in dominant_dosha:
            primary_dosha = dominant_dosha.split('-')[0]
        else:
            primary_dosha = dominant_dosha
        
        dosha_guide = DOSHA_GUIDELINES.get(primary_dosha, DOSHA_GUIDELINES['vata'])
        season_guide = SEASONAL_GUIDELINES.get(season, SEASONAL_GUIDELINES['spring'])
        
        # Build personalized sections
        constitution_tip = f"""**Personalized for Your {dominant_dosha.title().replace('-', ' & ')} Constitution:**
For {dominant_dosha.title()} individuals, emphasize {', '.join(dosha_guide['balance_foods'][:3])} foods and approaches. Avoid {', '.join(dosha_guide['avoid_foods'][:2])} qualities that can aggravate your constitution."""
        
        seasonal_tip = f"""**Seasonal Tip ({season.title()}):**
During {season}, which naturally affects {season_guide['predominant_dosha'].title()} dosha, {season_guide['general_advice'].lower()}. {season_guide['sri_lanka_context']}."""
        
        # Combine
        personalized = f"""{base_answer}

{constitution_tip}

{seasonal_tip}"""
        
        return personalized
    
    def get_dosha_guidelines(self, dosha_name: str) -> Dict[str, Any]:
        """
        Get balancing guidelines for a specific dosha
        
        Args:
            dosha_name: Name of dosha ('vata', 'pitta', or 'kapha')
            
        Returns:
            Guidelines dictionary
        """
        return DOSHA_GUIDELINES.get(dosha_name, {})


# ============================================================
# Prakriti Assessment Questions (10-question version)
# ============================================================

PRAKRITI_QUESTIONS = [
    {
        'id': 'q1',
        'question': 'What is your body frame and build?',
        'options': {
            'A': 'Thin, light frame, hard to gain weight',
            'B': 'Medium build, muscular, athletic',
            'C': 'Heavy, sturdy frame, easy to gain weight'
        }
    },
    {
        'id': 'q2',
        'question': 'How is your digestion typically?',
        'options': {
            'A': 'Irregular, often gas or bloating, variable appetite',
            'B': 'Strong, feel hungry often, can\'t skip meals',
            'C': 'Slow but steady, can skip meals easily'
        }
    },
    {
        'id': 'q3',
        'question': 'What is your skin type?',
        'options': {
            'A': 'Dry, rough, thin, gets dry patches',
            'B': 'Warm, oily, prone to rashes or acne',
            'C': 'Thick, moist, smooth, oily'
        }
    },
    {
        'id': 'q4',
        'question': 'How do you handle stress or pressure?',
        'options': {
            'A': 'Anxious, worried, mind races',
            'B': 'Irritable, angry, impatient',
            'C': 'Calm, withdrawn, may avoid confrontation'
        }
    },
    {
        'id': 'q5',
        'question': 'What is your sleep pattern like?',
        'options': {
            'A': 'Light sleeper, difficulty falling asleep, wake easily',
            'B': 'Moderate sleep, wake refreshed, sleep well',
            'C': 'Heavy sleeper, need lots of sleep, hard to wake up'
        }
    },
    {
        'id': 'q6',
        'question': 'How is your energy level throughout the day?',
        'options': {
            'A': 'Comes in bursts, get tired easily, need stimulation',
            'B': 'Consistent, strong, can push through fatigue',
            'C': 'Steady and enduring, slow to start but consistent'
        }
    },
    {
        'id': 'q7',
        'question': 'What is your body temperature preference?',
        'options': {
            'A': 'Prefer warm weather, dislike cold and wind',
            'B': 'Prefer cool weather, dislike heat',
            'C': 'Comfortable in most weather, dislike damp/cold'
        }
    },
    {
        'id': 'q8',
        'question': 'How do you learn and remember information?',
        'options': {
            'A': 'Learn quickly, forget quickly, creative',
            'B': 'Sharp intellect, good memory, focused',
            'C': 'Learn slowly but retain well, methodical'
        }
    },
    {
        'id': 'q9',
        'question': 'What is your speaking style?',
        'options': {
            'A': 'Fast talker, talkative, scattered thoughts',
            'B': 'Precise, articulate, argumentative',
            'C': 'Slow, melodious, measured speech'
        }
    },
    {
        'id': 'q10',
        'question': 'How do you approach new activities or changes?',
        'options': {
            'A': 'Enthusiastic but may not finish, love variety',
            'B': 'Focused and determined, competitive',
            'C': 'Resistant to change, prefer routine and stability'
        }
    }
]


# ============================================================
# Usage Example
# ============================================================

def example_usage():
    """
    Example of how to use the PersonalizationEngine
    """
    # Initialize engine
    engine = PersonalizationEngine()
    
    # Example questionnaire responses
    responses = {
        'q1': 'A', 'q2': 'A', 'q3': 'A', 'q4': 'A', 'q5': 'A',
        'q6': 'B', 'q7': 'A', 'q8': 'A', 'q9': 'A', 'q10': 'A'
    }  # Mostly Vata
    
    # Create user profile
    profile = engine.create_user_profile(
        user_id='user123',
        prakriti_responses=responses
    )
    
    print(f"Dominant Dosha: {profile['dominant_dosha']}")
    print(f"Season: {profile['current_season']}")
    
    # Example answer personalization
    base_answer = "Triphala supports digestive health and detoxification."
    
    personalized = engine.personalize_answer(base_answer, profile)
    print("\nPersonalized Answer:")
    print(personalized)


if __name__ == "__main__":
    example_usage()
