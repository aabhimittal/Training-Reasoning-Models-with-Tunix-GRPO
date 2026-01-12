#!/usr/bin/env python3
"""
Training Data Generator for Tunix Reasoning Model
==================================================

Automatically generates diverse reasoning examples across multiple domains.
Outputs to reasoning_training_data.json for use in training pipeline.

Usage:
    python generate_training_data.py --count 1000 --output reasoning_training_data.json
"""

import json
import random
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class ExampleTemplate:
    """Template for generating training examples."""
    type: str
    difficulty: str
    domain: str
    question_template: str
    answer_template: str
    metadata: Dict[str, Any]


# ==============================================================================
# MATH PROBLEM GENERATORS
# ==============================================================================

class MathGenerator:
    """Generate mathematical reasoning problems."""
    
    @staticmethod
    def generate_percentage_problem():
        """Generate percentage calculation problems."""
        percentage = random.choice([10, 15, 20, 25, 30, 40, 50, 75])
        number = random.choice([80, 120, 160, 200, 240, 360, 480, 600])
        answer = (percentage / 100) * number
        
        return {
            "question": f"What is {percentage}% of {number}? Let's think step by step.",
            "answer": str(int(answer)) if answer.is_integer() else f"{answer:.2f}",
            "type": "math",
            "difficulty": "easy",
            "domain": "arithmetic",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["percentage", "multiplication"]
            }
        }
    
    @staticmethod
    def generate_linear_equation():
        """Generate linear equation problems."""
        a = random.randint(2, 9)
        b = random.randint(5, 20)
        result = random.randint(15, 50)
        x = (result - b) / a
        
        if x.is_integer():
            return {
                "question": f"Solve for x: {a}x + {b} = {result}. Show your work step by step.",
                "answer": f"x = {int(x)}",
                "type": "math",
                "difficulty": "easy",
                "domain": "algebra",
                "metadata": {
                    "source": "auto_generated",
                    "verification": "verifiable",
                    "keywords": ["linear equation", "algebra", "solving"]
                }
            }
        return None
    
    @staticmethod
    def generate_word_problem():
        """Generate word problems."""
        scenarios = [
            {
                "context": "A car travels",
                "rate": random.choice([40, 50, 60, 70, 80]),
                "time": random.choice([1.5, 2, 2.5, 3, 3.5]),
                "unit": "km"
            },
            {
                "context": "A cyclist rides",
                "rate": random.choice([15, 20, 25, 30]),
                "time": random.choice([2, 3, 4, 5]),
                "unit": "miles"
            },
            {
                "context": "A plane flies",
                "rate": random.choice([400, 500, 600, 700]),
                "time": random.choice([1, 1.5, 2, 2.5]),
                "unit": "km"
            }
        ]
        
        scenario = random.choice(scenarios)
        distance = scenario['rate'] * scenario['time']
        
        return {
            "question": f"{scenario['context']} at {scenario['rate']} {scenario['unit']}/hour for {scenario['time']} hours. How far does it travel? Explain your reasoning.",
            "answer": f"{distance} {scenario['unit']}",
            "type": "math",
            "difficulty": "medium",
            "domain": "word_problems",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["distance", "rate", "time", "word problem"]
            }
        }
    
    @staticmethod
    def generate_geometry_problem():
        """Generate geometry problems."""
        shapes = [
            {
                "name": "rectangle",
                "property": "perimeter",
                "length": random.randint(5, 20),
                "width": random.randint(5, 15),
                "formula": lambda l, w: 2 * (l + w)
            },
            {
                "name": "rectangle",
                "property": "area",
                "length": random.randint(5, 20),
                "width": random.randint(5, 15),
                "formula": lambda l, w: l * w
            },
            {
                "name": "square",
                "property": "perimeter",
                "length": random.randint(5, 15),
                "width": None,
                "formula": lambda l, w: 4 * l
            }
        ]
        
        shape = random.choice(shapes)
        if shape['width']:
            answer = shape['formula'](shape['length'], shape['width'])
            question = f"A {shape['name']} has length {shape['length']} meters and width {shape['width']} meters. What is its {shape['property']}? Show your calculation."
        else:
            answer = shape['formula'](shape['length'], None)
            question = f"A {shape['name']} has side length {shape['length']} meters. What is its {shape['property']}? Show your calculation."
        
        return {
            "question": question,
            "answer": f"{answer} square meters" if shape['property'] == 'area' else f"{answer} meters",
            "type": "math",
            "difficulty": "easy",
            "domain": "geometry",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["geometry", shape['property'], shape['name']]
            }
        }
    
    @staticmethod
    def generate_proportion_problem():
        """Generate proportion problems."""
        items = ["apples", "books", "pencils", "tickets", "cookies"]
        item = random.choice(items)
        
        quantity1 = random.randint(3, 8)
        price1 = round(random.uniform(2.0, 8.0), 2)
        quantity2 = random.randint(10, 20)
        
        unit_price = price1 / quantity1
        price2 = round(unit_price * quantity2, 2)
        
        return {
            "question": f"If {quantity1} {item} cost ${price1:.2f}, how much do {quantity2} {item} cost at the same rate? Explain your method.",
            "answer": f"${price2:.2f}",
            "type": "math",
            "difficulty": "medium",
            "domain": "proportions",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["proportion", "unit rate", "word problem"]
            }
        }
    
    @staticmethod
    def generate_examples(count: int) -> List[Dict[str, Any]]:
        """Generate multiple math examples."""
        generators = [
            MathGenerator.generate_percentage_problem,
            MathGenerator.generate_linear_equation,
            MathGenerator.generate_word_problem,
            MathGenerator.generate_geometry_problem,
            MathGenerator.generate_proportion_problem
        ]
        
        examples = []
        attempts = 0
        max_attempts = count * 3
        
        while len(examples) < count and attempts < max_attempts:
            generator = random.choice(generators)
            example = generator()
            if example:
                examples.append(example)
            attempts += 1
        
        return examples


# ==============================================================================
# CODE REASONING GENERATORS
# ==============================================================================

class CodeGenerator:
    """Generate code reasoning problems."""
    
    @staticmethod
    def generate_python_output():
        """Generate Python code output problems."""
        templates = [
            {
                "code": "[x**2 for x in range({n})]",
                "answer": lambda n: str([x**2 for x in range(n)]),
                "n": random.randint(4, 7)
            },
            {
                "code": "[i for i in range({n}) if i % 2 == 0]",
                "answer": lambda n: str([i for i in range(n) if i % 2 == 0]),
                "n": random.randint(8, 12)
            },
            {
                "code": "sum([x for x in range({n})])",
                "answer": lambda n: str(sum(range(n))),
                "n": random.randint(5, 10)
            }
        ]
        
        template = random.choice(templates)
        n = template['n']
        code = template['code'].format(n=n)
        answer = template['answer'](n)
        
        return {
            "question": f"What does this Python code output: `print({code})`? Trace through the execution.",
            "answer": answer,
            "type": "code",
            "difficulty": "easy",
            "domain": "python",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["python", "output", "list comprehension"]
            }
        }
    
    @staticmethod
    def generate_debugging_problem():
        """Generate code debugging problems."""
        bugs = [
            {
                "code": "for i in range(10): if i % 2 = 0: print(i)",
                "fix": "Use == instead of = for comparison",
                "corrected": "for i in range(10): if i % 2 == 0: print(i)"
            },
            {
                "code": "def factorial(n): return n * factorial(n-1)",
                "fix": "Missing base case for recursion",
                "corrected": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
            },
            {
                "code": "list = [1, 2, 3]; list.append[4]",
                "fix": "Use parentheses () not brackets [] for method call",
                "corrected": "list = [1, 2, 3]; list.append(4)"
            }
        ]
        
        bug = random.choice(bugs)
        
        return {
            "question": f"Debug this code: `{bug['code']}`. What's wrong and what's the fix?",
            "answer": f"{bug['fix']}. Corrected: {bug['corrected']}",
            "type": "code",
            "difficulty": "easy",
            "domain": "debugging",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["debugging", "syntax error", "python"]
            }
        }
    
    @staticmethod
    def generate_complexity_problem():
        """Generate algorithm complexity problems."""
        algorithms = [
            {
                "name": "linear search",
                "complexity": "O(n)",
                "reason": "must check each element once in worst case"
            },
            {
                "name": "binary search",
                "complexity": "O(log n)",
                "reason": "halves the search space with each step"
            },
            {
                "name": "bubble sort",
                "complexity": "O(n²)",
                "reason": "nested loops comparing adjacent elements"
            },
            {
                "name": "merge sort",
                "complexity": "O(n log n)",
                "reason": "divides array recursively (log n) and merges (n)"
            }
        ]
        
        algo = random.choice(algorithms)
        
        return {
            "question": f"What's the time complexity of {algo['name']}? Explain why.",
            "answer": f"{algo['complexity']} because it {algo['reason']}",
            "type": "code",
            "difficulty": "medium",
            "domain": "algorithms",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["time complexity", algo['name'], "algorithms"]
            }
        }
    
    @staticmethod
    def generate_examples(count: int) -> List[Dict[str, Any]]:
        """Generate multiple code examples."""
        generators = [
            CodeGenerator.generate_python_output,
            CodeGenerator.generate_debugging_problem,
            CodeGenerator.generate_complexity_problem
        ]
        
        examples = []
        for _ in range(count):
            generator = random.choice(generators)
            examples.append(generator())
        
        return examples


# ==============================================================================
# SCIENCE REASONING GENERATORS
# ==============================================================================

class ScienceGenerator:
    """Generate science reasoning problems."""
    
    @staticmethod
    def generate_physics_problem():
        """Generate physics problems."""
        problems = [
            {
                "question": "Why do objects fall at the same rate in a vacuum regardless of mass?",
                "answer": "Gravitational acceleration is constant (9.8 m/s²) regardless of mass, and without air resistance all objects accelerate equally",
                "domain": "physics"
            },
            {
                "question": "Explain why a helium balloon floats in air using principles of buoyancy.",
                "answer": "Helium is less dense than air, so the buoyant force (weight of displaced air) exceeds the weight of the balloon, causing it to float",
                "domain": "physics"
            },
            {
                "question": "Why does a spinning figure skater speed up when pulling their arms in?",
                "answer": "Conservation of angular momentum: L = Iω. Reducing moment of inertia (I) by pulling arms in requires angular velocity (ω) to increase",
                "domain": "physics"
            }
        ]
        
        problem = random.choice(problems)
        
        return {
            "question": problem['question'] + " Explain the reasoning.",
            "answer": problem['answer'],
            "type": "science",
            "difficulty": "medium",
            "domain": problem['domain'],
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["physics", "explanation", "principles"]
            }
        }
    
    @staticmethod
    def generate_chemistry_problem():
        """Generate chemistry problems."""
        problems = [
            {
                "question": "Why does salt dissolve in water but not in oil?",
                "answer": "Salt is ionic (polar) and water is polar, so 'like dissolves like'. Oil is nonpolar and cannot break ionic bonds in salt",
                "domain": "chemistry"
            },
            {
                "question": "Explain why iron rusts but gold doesn't.",
                "answer": "Iron is more reactive and readily oxidizes when exposed to oxygen and water. Gold is noble metal with very low reactivity",
                "domain": "chemistry"
            },
            {
                "question": "Why does adding salt to water raise its boiling point?",
                "answer": "Boiling point elevation: salt ions disrupt water molecules, requiring more energy (higher temperature) to transition to gas phase",
                "domain": "chemistry"
            }
        ]
        
        problem = random.choice(problems)
        
        return {
            "question": problem['question'] + " Explain at the molecular level.",
            "answer": problem['answer'],
            "type": "science",
            "difficulty": "medium",
            "domain": problem['domain'],
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["chemistry", "molecular", "explanation"]
            }
        }
    
    @staticmethod
    def generate_biology_problem():
        """Generate biology problems."""
        problems = [
            {
                "question": "Why do plants appear green?",
                "answer": "Chlorophyll absorbs red and blue light for photosynthesis but reflects green light, which we see",
                "domain": "biology"
            },
            {
                "question": "Explain why antibiotics don't work on viruses.",
                "answer": "Antibiotics target bacterial structures like cell walls. Viruses lack these structures and rely on host cells to reproduce",
                "domain": "biology"
            },
            {
                "question": "Why do we need to breathe oxygen?",
                "answer": "Oxygen is the final electron acceptor in cellular respiration, enabling ATP production for cellular energy",
                "domain": "biology"
            }
        ]
        
        problem = random.choice(problems)
        
        return {
            "question": problem['question'] + " Explain the biological reasoning.",
            "answer": problem['answer'],
            "type": "science",
            "difficulty": "medium",
            "domain": problem['domain'],
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["biology", "explanation", "process"]
            }
        }
    
    @staticmethod
    def generate_examples(count: int) -> List[Dict[str, Any]]:
        """Generate multiple science examples."""
        generators = [
            ScienceGenerator.generate_physics_problem,
            ScienceGenerator.generate_chemistry_problem,
            ScienceGenerator.generate_biology_problem
        ]
        
        examples = []
        for _ in range(count):
            generator = random.choice(generators)
            examples.append(generator())
        
        return examples


# ==============================================================================
# LOGIC PUZZLE GENERATORS
# ==============================================================================

class LogicGenerator:
    """Generate logic puzzles."""
    
    @staticmethod
    def generate_weighing_problem():
        """Generate balance scale problems."""
        n_balls = random.choice([8, 9, 12])
        
        if n_balls == 8:
            answer = "2 weighings"
            explanation = "Divide into 3-3-2. Weigh first two groups. If balanced, heavy is in remaining 2 (1 weighing). If unbalanced, divide heavier group of 3 and weigh any two"
        elif n_balls == 9:
            answer = "2 weighings"
            explanation = "Divide into 3-3-3. Weigh two groups. If balanced, heavy is in third group. Then weigh any two from heavy group"
        else:  # 12
            answer = "3 weighings"
            explanation = "Divide into 4-4-4. First weighing identifies heavy group. Second weighing narrows to 2 balls. Third weighing identifies the heavy one"
        
        return {
            "question": f"You have {n_balls} balls, one slightly heavier. You have a balance scale. What's the minimum weighings to find the heavy ball? Explain your strategy.",
            "answer": f"{answer}. {explanation}",
            "type": "logic_puzzle",
            "difficulty": "hard",
            "domain": "optimization",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["weighing", "optimization", "strategy"]
            }
        }
    
    @staticmethod
    def generate_syllogism():
        """Generate syllogism problems."""
        creatures = [
            ("bloops", "razzies", "lazzies"),
            ("flibbers", "glorps", "zorks"),
            ("widgets", "gadgets", "doodads"),
            ("snorfs", "morfles", "borfles")
        ]
        
        a, b, c = random.choice(creatures)
        
        return {
            "question": f"If all {a} are {b} and all {b} are {c}, are all {a} definitely {c}? Explain your logical reasoning.",
            "answer": f"Yes, by transitive property: {a} → {b} → {c}, therefore {a} → {c}",
            "type": "logic_puzzle",
            "difficulty": "medium",
            "domain": "logic",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["syllogism", "logic", "transitivity"]
            }
        }
    
    @staticmethod
    def generate_truth_teller():
        """Generate truth-teller/liar problems."""
        scenarios = [
            {
                "question": "You meet two people. One always tells truth, one always lies. Person A says 'Person B is a liar.' What can you conclude?",
                "answer": "If A is truth-teller, then B is liar (consistent). If A is liar, then B is truth-teller, but then A's statement would be true (contradiction). Therefore A is truth-teller, B is liar.",
                "difficulty": "medium"
            },
            {
                "question": "Three people: one always tells truth, one always lies, one answers randomly. You can ask one yes/no question to one person. How do you identify the truth-teller?",
                "answer": "Ask person A: 'If I asked you if B is the truth-teller, would you say yes?' Truth-teller and liar will give consistent answers about same person, random won't help, so ask two people same question",
                "difficulty": "hard"
            }
        ]
        
        scenario = random.choice(scenarios)
        
        return {
            "question": scenario['question'] + " Show your reasoning.",
            "answer": scenario['answer'],
            "type": "logic_puzzle",
            "difficulty": scenario['difficulty'],
            "domain": "logic",
            "metadata": {
                "source": "auto_generated",
                "verification": "verifiable",
                "keywords": ["logic", "truth", "deduction"]
            }
        }
    
    @staticmethod
    def generate_examples(count: int) -> List[Dict[str, Any]]:
        """Generate multiple logic examples."""
        generators = [
            LogicGenerator.generate_weighing_problem,
            LogicGenerator.generate_syllogism,
            LogicGenerator.generate_truth_teller
        ]
        
        examples = []
        for _ in range(count):
            generator = random.choice(generators)
            examples.append(generator())
        
        return examples


# ==============================================================================
# CREATIVE TASK GENERATORS
# ==============================================================================

class CreativeGenerator:
    """Generate creative writing and ideation tasks."""
    
    @staticmethod
    def generate_story_opening():
        """Generate story opening prompts."""
        scenarios = [
            "a world where emotions are visible as colors",
            "a society where memories can be bought and sold",
            "a city that exists in multiple dimensions simultaneously",
            "a world where everyone hears a unique soundtrack to their life",
            "a planet where time flows differently in each region"
        ]
        
        scenario = random.choice(scenarios)
        
        return {
            "question": f"Write a creative opening paragraph for a story about {scenario}. Show your creative reasoning process.",
            "answer": "[Creative opening paragraph that establishes the unique world mechanic, introduces a character in a moment of tension, and creates immediate stakes]",
            "type": "creative_writing",
            "difficulty": "medium",
            "domain": "fiction",
            "metadata": {
                "source": "auto_generated",
                "verification": "non_verifiable",
                "keywords": ["creative writing", "world-building", "opening"]
            }
        }
    
    @staticmethod
    def generate_metaphor():
        """Generate metaphor creation tasks."""
        concepts = [
            ("artificial intelligence", "child learning"),
            ("climate change", "blanket fort"),
            ("internet", "highway system"),
            ("democracy", "potluck dinner"),
            ("education", "gardening")
        ]
        
        concept, comparison = random.choice(concepts)
        
        return {
            "question": f"Create a metaphor that explains {concept} to someone unfamiliar with it. Explain your creative choices.",
            "answer": f"[Metaphor comparing {concept} to {comparison}, with explanation of why this comparison illuminates key aspects]",
            "type": "creative_writing",
            "difficulty": "medium",
            "domain": "metaphor",
            "metadata": {
                "source": "auto_generated",
                "verification": "non_verifiable",
                "keywords": ["metaphor", "explanation", "creativity"]
            }
        }
    
    @staticmethod
    def generate_invention():
        """Generate invention/innovation tasks."""
        problems = [
            "reduce food waste in households",
            "help elderly people stay connected with family",
            "make public transportation more efficient",
            "encourage children to exercise",
            "reduce plastic usage in daily life"
        ]
        
        problem = random.choice(problems)
        
        return {
            "question": f"Design an innovative solution to {problem}. Explain your key features and the reasoning behind them.",
            "answer": f"[Solution description with 3-4 key features, each with reasoning for how it addresses the problem]",
            "type": "creative_ideation",
            "difficulty": "hard",
            "domain": "innovation",
            "metadata": {
                "source": "auto_generated",
                "verification": "non_verifiable",
                "keywords": ["innovation", "problem-solving", "design"]
            }
        }
    
    @staticmethod
    def generate_examples(count: int) -> List[Dict[str, Any]]:
        """Generate multiple creative examples."""
        generators = [
            CreativeGenerator.generate_story_opening,
            CreativeGenerator.generate_metaphor,
            CreativeGenerator.generate_invention
        ]
        
        examples = []
        for _ in range(count):
            generator = random.choice(generators)
            examples.append(generator())
        
        return examples


# ==============================================================================
# SUMMARIZATION GENERATORS
# ==============================================================================

class SummarizationGenerator:
    """Generate summarization tasks."""
    
    @staticmethod
    def generate_technical_summary():
        """Generate technical text summarization."""
        texts = [
            {
                "text": "Machine learning algorithms use statistical techniques to enable computer systems to improve their performance on tasks through experience. Neural networks, inspired by biological neurons, consist of interconnected layers that process information. Deep learning uses multiple layers to extract increasingly complex features from raw input.",
                "answer": "Machine learning enables computers to learn from experience using statistical methods, with neural networks and deep learning being key approaches that process information through layered structures."
            },
            {
                "text": "Blockchain technology creates a distributed ledger that records transactions across multiple computers. Each block contains transaction data and is cryptographically linked to previous blocks. This structure makes the ledger tamper-resistant and transparent to all participants.",
                "answer": "Blockchain is a distributed, tamper-resistant ledger system where transaction data is stored in cryptographically linked blocks across multiple computers."
            }
        ]
        
        item = random.choice(texts)
        
        return {
            "question": f"Summarize the following in simple terms: '{item['text']}' Show your reasoning for what to emphasize.",
            "answer": item['answer'],
            "type": "summarization",
            "difficulty": "medium",
            "domain": "technical",
            "metadata": {
                "source": "auto_generated",
                "verification": "non_verifiable",
                "keywords": ["summarization", "technical", "simplification"]
            }
        }
    
    @staticmethod
    def generate_examples(count: int) -> List[Dict[str, Any]]:
        """Generate multiple summarization examples."""
        examples = []
        for _ in range(count):
            examples.append(SummarizationGenerator.generate_technical_summary())
        
        return examples


# ==============================================================================
# MAIN GENERATOR
# ==============================================================================

class TrainingDataGenerator:
    """Main class for generating complete training dataset."""
    
    def __init__(self, target_distribution: Dict[str, float] = None):
        """
        Initialize generator with target distribution.
        
        Args:
            target_distribution: Dict mapping type to percentage (e.g., {'math': 0.4})
        """
        self.target_distribution = target_distribution or {
            'math': 0.40,
            'code': 0.20,
            'logic_puzzle': 0.15,
            'science': 0.10,
            'creative': 0.10,
            'summarization': 0.05
        }
    
    def generate(self, total_count: int) -> List[Dict[str, Any]]:
        """
        Generate complete training dataset.
        
        Args:
            total_count: Total number of examples to generate
            
        Returns:
            List of training examples
        """
        examples = []
        
        # Calculate counts per type
        type_counts = {
            type_name: int(total_count * percentage)
            for type_name, percentage in self.target_distribution.items()
        }
        
        print(f"Generating {total_count} training examples...")
        print(f"Distribution: {type_counts}")
        
        # Generate math examples
        print(f"Generating {type_counts['math']} math examples...")
        examples.extend(MathGenerator.generate_examples(type_counts['math']))
        
        # Generate code examples
        print(f"Generating {type_counts['code']} code examples...")
        examples.extend(CodeGenerator.generate_examples(type_counts['code']))
        
        # Generate logic puzzles
        print(f"Generating {type_counts['logic_puzzle']} logic puzzles...")
        examples.extend(LogicGenerator.generate_examples(type_counts['logic_puzzle']))
        
        # Generate science examples
        print(f"Generating {type_counts['science']} science examples...")
        examples.extend(ScienceGenerator.generate_examples(type_counts['science']))
        
        # Generate creative examples
        print(f"Generating {type_counts['creative']} creative examples...")
        examples.extend(CreativeGenerator.generate_examples(type_counts['creative']))
        
        # Generate summarization examples
        print(f"Generating {type_counts['summarization']} summarization examples...")
        examples.extend(SummarizationGenerator.generate_examples(type_counts['summarization']))
        
        # Shuffle examples
        random.shuffle(examples)
        
        print(f"✅ Generated {len(examples)} total examples")
        
        return examples
    
    def save_to_file(self, examples: List[Dict[str, Any]], filename: str):
        """Save examples to JSON file."""
        with open(filename, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"✅ Saved {len(examples)} examples to {filename}")
    
    def get_statistics(self, examples: List[Dict[str, Any]]):
        """Print dataset statistics."""
        from collections import Counter
        
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        
        print(f"\nTotal examples: {len(examples)}")
        
        # Type distribution
        types = Counter(ex['type'] for ex in examples)
        print("\nBy type:")
        for type_name, count in types.most_common():
            percentage = count / len(examples) * 100
            print(f"  {type_name}: {count} ({percentage:.1f}%)")
        
        # Difficulty distribution
        difficulties = Counter(ex['difficulty'] for ex in examples)
        print("\nBy difficulty:")
        for diff, count in difficulties.most_common():
            percentage = count / len(examples) * 100
            print(f"  {diff}: {count} ({percentage:.1f}%)")
        
        # Verification status
        verifiable = sum(1 for ex in examples if ex['metadata'].get('verification') == 'verifiable')
        print(f"\nVerifiable: {verifiable} ({verifiable/len(examples)*100:.1f}%)")
        print(f"Non-verifiable: {len(examples) - verifiable} ({(len(examples)-verifiable)/len(examples)*100:.1f}%)")
        
        print("="*80 + "\n")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(
        description="Generate training data for Tunix reasoning model"
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1000,
        help='Number of examples to generate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reasoning_training_data.json',
        help='Output filename (default: reasoning_training_data.json)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate data
    generator = TrainingDataGenerator()
    examples = generator.generate(args.count)
    
    # Save to file
    generator.save_to_file(examples, args.output)
    
    # Print statistics
    generator.get_statistics(examples)
    
    print(f"✨ Done! Dataset saved to {args.output}")
    print(f"Upload this file to Kaggle and use in your notebook.")


if __name__ == '__main__':
    main()
