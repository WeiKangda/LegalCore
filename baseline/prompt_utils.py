import re


def extract_event_triggers_with_spans_fixed(text):
    """
    Extract event triggers, their indices, and spans from the input text.
    Properly handle punctuation, single quotes, and numeric formats (e.g., "1.4").

    Args:
        text (str): Input text containing event triggers in the format {EXX trigger_word}.

    Returns:
        list of dict: Each dict contains 'Span' and 'Trigger' for each event.
    """
    # Replace {EXX trigger_word} with trigger_word
    cleaned_text = re.sub(r"\{E\d+\s+([^\}]+)\}", r"\1", text)

    # Use regex to split words while treating numeric formats as single tokens
    words = re.findall(r'\b\w+(?:\.\w+)?\b', cleaned_text)

    matches = list(re.finditer(r"\{E\d+\s+([^\}]+)\}", text))

    # Extract events and their spans
    events = []
    current_index = 0
    print(words)

    for word in words:

        for match in matches:
            trigger_word = match.group(1)
            if trigger_word==word:
                # Calculate span range
                start_index = max(0, current_index - 2)
                end_index = min(len(words) - 1, current_index + 2)
                span = f"{start_index}-{end_index}"
                events.append({
                    "Span": span,
                    "Trigger": trigger_word
                })
                break
        print(current_index)

        current_index += 1

    return events


# Example usage
# text = ('1.4 " {E16 Invention} " means any {E17 invention} , know-how, data, '
#         '{E18 discovery} or proprietary information, whether or not patentable, '
#         'that is made or generated solely by the Representatives of Anixa or OntoChem '
#         'or jointly by the Representatives of Anixa and OntoChem in performing the Research Plan, '
#         'including all intellectual property rights in the foregoing.')
text="""1. PURCHASE OF EQUIPMENT. BNL at its expense shall  {E13 obtain}  ,  {E14 install}  ,  {E15 maintain}  and  {E16 upgrade}  as necessary any and all hardware, software, data and telephone lines, other communications equipment and any other equipment (hereinafter collectively  referred  to as the "Equipment") which it  {E17 determines}  is necessary to  {E18 allow}  it to  {E19 use}  and  {E20 access}  the VIP System pursuant to the terms of this Agreement."""
events = extract_event_triggers_with_spans_fixed(text)

# Output the formatted results
for event in events:
    print(f"Span: {event['Span']}")
    print(f"Trigger: {event['Trigger']}")
