import re
from typing import List

# Example list of states and districts
STATES = [
    "Andhra Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh"
]

DISTRICTS = [
    "Bangalore", "Chennai", "Mumbai", "Hyderabad", "Lucknow"
]

def extract_location_from_text(text: str) -> List[str]:
    """
    Extract possible state or district names from input text.

    Args:
        text (str): Input text to search for locations.

    Returns:
        List[str]: List of matched locations (states or districts).
    """
    text_lower = text.lower()
    matched_locations = []

    # Check for states
    for state in STATES:
        if state.lower() in text_lower:
            matched_locations.append(state)

    # Check for districts
    for district in DISTRICTS:
        if district.lower() in text_lower:
            matched_locations.append(district)

    return matched_locations


# Example usage
if __name__ == "__main__":
    sample_text = "We have job openings in Bangalore and Maharashtra."
    locations = extract_location_from_text(sample_text)
    print(locations)  # Output: ['Maharashtra', 'Bangalore']
