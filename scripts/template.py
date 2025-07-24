#!/usr/bin/env python3
"""
Template definitions for entity text generation.
Templates use [entity] as a placeholder for entity names.
"""

# Basic templates
TEMPLATES = {
    # Original format - just the entity name
    "entity_only": "[entity]",
    
    # Occupation
    "occupation_question": "What is the occupation of [entity]",
    "occupation_simple": "The occupation of [entity] is",
    "profession_query": "What profession does [entity] have?",
    "professional_intro": "In their professional capacity, [entity]",

    # Description templates
    "who_is": "Who is [entity]",
    "about": "Tell me about [entity]",
    "describe": "Describe [entity]",

    # Biographical templates
    "biography": "Write a biography of [entity]",
    "life_story": "Tell me the life story of [entity]",
    
    # Classification templates
    "classify": "Classify [entity] by profession:",
    "category": "What category does [entity] belong to?",

    # Basic information questions
    "what_is": "What is [entity]",
    "who_or_what": "Who or what is [entity]",
    "about": "What do you know about [entity]",
    "tell_me": "Can you tell me about [entity]",
    "information": "What information do you have about [entity]",
    "describe": "How would you describe [entity]",
    
    # Characteristics and qualities
    "characteristics": "What are the characteristics of [entity]",
    "unique": "What is unique about [entity]",
    "special": "What is special about [entity]",
    "notable": "What is notable about [entity]",
    "remarkable": "What is remarkable about [entity]",
    "interesting": "What is interesting about [entity]",
    
    # Purpose and function
    "purpose": "What is the purpose of [entity]",
    "function": "What is the function of [entity]",
    "role": "What is the role of [entity]",
    "significance": "What is the significance of [entity]",
    "importance": "What is the importance of [entity]",
    
    # Background and context
    "background": "What is the background of [entity]",
    "history": "What is the history of [entity]",
    "origin": "What is the origin of [entity]",
    "story": "What is the story behind [entity]",
    
    # Detailed inquiries
    "details": "What details can you share about [entity]",
    "facts": "What facts do you know about [entity]",
    "key_points": "What are the key points about [entity]",
    "main_features": "What are the main features of [entity]",
    "reputation": "What is the reputation of [entity]",

    # Irrelevant context
    "look" : "Everyone in the room looked at [entity]",
    "gift" : "The gift was given to [entity]",
    "weather" : "The weather was quite pleasant for [entity]",

}

def get_template(template_name: str) -> str:
    """Get template by name."""
    if template_name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Template '{template_name}' not found. Available templates: {available}")
    return TEMPLATES[template_name]

def apply_template(template: str, entity_name: str) -> str:
    """Apply template by replacing [entity] placeholder with entity name."""
    if "[entity]" not in template:
        raise ValueError(f"Template must contain [entity] placeholder: {template}")
    return template.replace("[entity]", entity_name)

def list_templates() -> list[str]:
    """List all available template names."""
    return list(TEMPLATES.keys())

def show_template_examples(entity_name: str = "Barack Obama") -> None:
    """Show examples of all templates with a sample entity name."""
    print(f"Template examples with '{entity_name}':")
    print("=" * 50)
    for name, template in TEMPLATES.items():
        result = apply_template(template, entity_name)
        print(f"{name:20} : {result}")
    print("=" * 50)

if __name__ == "__main__":
    # Show examples when run directly
    show_template_examples()