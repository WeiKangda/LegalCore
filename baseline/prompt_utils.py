import re


def extract_event_triggers_with_spans(text):
    """
    Extract event triggers, their indices, and spans from the input text.

    Args:
        text (str): Input text containing event triggers in the format {EXX trigger_word}.

    Returns:
        list of dict: Each dict contains 'Span' and 'Trigger' for each event.
    """
    matches = re.finditer(r"\{E\d+\s+([^\}]+)\}", text)
    events = []

    # Tokenize text into words and keep track of their positions
    words = text.replace("{", "").replace("}", "").split()
    current_index = 0

    for word in words:
        cleaned_word = re.sub(r"[^\w]", "", word)  # Remove punctuation for clean matching
        # Check if the current word is part of an event trigger
        for match in matches:
            trigger_word = match.group(1)
            if trigger_word.startswith(cleaned_word):
                # Calculate span range
                start_index = max(0, current_index - 2)
                end_index = min(len(words) - 1, current_index + 2)
                span = f"{start_index}-{end_index}"
                events.append({
                    "Span": span,
                    "Trigger": trigger_word
                })
                break
        current_index += 1

    return events

if __name__=="__main__":
    # Example usage
    text = ("The Parties will diligently {E22 perform} their respective {E23 activities} set forth in the Research Plan "
            "(such {E24 activities} , collectively, the \"Research {E25 Program} \") in accordance with the timelines "
            "set forth therein, with the objective of {E26 identifying} Hit Compounds and Lead Scaffolds that "
            "{E27 modulate} the applicable Target.")

    events = extract_event_triggers_with_spans(text)

    # Output the formatted results
    for event in events:
        print(f"Span: {event['Span']}")
        print(f"Trigger: {event['Trigger']}")
