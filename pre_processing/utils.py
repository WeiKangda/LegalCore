import re
from collections import defaultdict
import json

def load_jsonl(file_path):
    """
    Load a JSON Lines (jsonl) file and return its contents as a list of dictionaries.

    Args:
        file_path (str): Path to the jsonl file.

    Returns:
        list: A list of dictionaries, each representing a line in the jsonl file.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on line {len(data)+1}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return data
    
def get_words_by_index(text, index):
    words = re.split(r'\s+', text)
    return words[index]

def replace_elements(lst, i, j, singleton_index):
    """
    Replaces the element at index i with the element at index j, 
    and the element at index j with the element at index i.

    Parameters:
        lst (list): The list of elements.
        i (int): The index of the first element to replace.
        j (int): The index of the second element to replace.

    Returns:
        list: The modified list with swapped elements.
    """
    if not (0 <= i < len(lst) and 0 <= j < len(lst)):
        raise IndexError("Indices i and j must be within the range of the list.")
    
    # Swap elements at indices i and j
    if i != j:
        lst[i], lst[j] = "{E"+str(singleton_index)+" "+lst[i], lst[j]+"}"
        #print(lst[i], lst[j])
    else:
        lst[i] = "{E"+str(singleton_index)+" "+lst[i]+"}"
    return lst

def remove_color_and_tokenization(input_text):
    # Regex pattern to match #COLOR and #TOKENIZATION-TYPE lines
    pattern = r"^#COLOR:.*|^#TOKENIZATION-TYPE:.*"
    # Remove the matching lines
    cleaned_text = re.sub(pattern, "", input_text, flags=re.MULTILINE)
    # Remove any extra blank lines
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text).strip()
    return cleaned_text

def read_text_file(file_path):
    """
    Reads the content of a text file and returns it as a string.

    :param file_path: Path to the text file
    :return: Content of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except IOError:
        print(f"Error: Could not read the file at {file_path}.")

def process_text(text):
    # Find all occurrences of {Exx word}
    pattern = r"\{(E\d+)\s+(.*?)\}"
    matches = list(re.finditer(pattern, text))
    #print(len(matches))

    processed_text = text
    ex_indices = {}

    # Replace each {Exx word} with the word and save its absolute index
    for match in matches:
        ex_tag = match.group(1)  # E.g., "E1"
        word = match.group(2)    # Word within the curly braces

        # Replace the pattern in the text with the word only
        #print(match.group(0), "{"+ex_tag+"_"+word+"}")
        processed_text = processed_text.replace(match.group(0), " "+word+" ")
    # Create word list for processed text
    processed_text_words = re.split(r'\s+', processed_text.strip()) 
    singleton_text_words = re.split(r'\s+', processed_text.strip())

    word_book_keep = {} # Used for keeping the index of last processed word with the same trigger word
    # Find indices of each replaced word in the final processed text
    singleton_index = 0
    global_index = 0
    for match in matches:
        #print(singleton_index, match)
        ex_tag = match.group(1)  # E.g., "E1"
        word = match.group(2)    # Word within the curly braces
        words_split = re.split(r'\s+', word)
         
        #print(len(processed_text_words), len(words_split))
        # Find the starting index of the first word in the processed text
        for i in range(len(processed_text_words)):
            if processed_text_words[i:i + len(words_split)] == words_split:
                if ex_tag not in ex_indices:
                    if word in word_book_keep:
                        if i > word_book_keep[word] and i > global_index:
                            word_book_keep[word] = i
                            global_index = i
                            ex_indices[ex_tag] = [[word, i, "E"+str(singleton_index)]]
                            singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                            singleton_index += 1
                        else:
                            continue
                    else:
                        if i > global_index:
                            global_index = i
                            word_book_keep[word] = i
                            ex_indices[ex_tag] = [[word, i, "E"+str(singleton_index)]]
                            singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                            singleton_index += 1
                        else:
                            continue
                else:
                    already_in = False
                    for mention in ex_indices[ex_tag]:
                        if mention[1] == i:
                            already_in = True
                            break
                    if already_in:
                        continue
                    else:
                        if word in word_book_keep:
                            if i > word_book_keep[word] and i > global_index:
                                global_index = i
                                word_book_keep[word] = i
                                ex_indices[ex_tag].append([word, i, "E"+str(singleton_index)])
                                singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                                singleton_index += 1
                            else:
                                continue
                        else:
                            if i > global_index:
                                global_index = i
                                word_book_keep[word] = i
                                ex_indices[ex_tag].append([word, i, "E"+str(singleton_index)])
                                singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                                singleton_index += 1
                            else:
                                continue
                break
        #print(global_index, singleton_index, match)
    #print(len(matches))
    # Create sentence list for processed text
    processed_text_sentences = re.split(r'(?<=[.!?])\s+', processed_text.strip())
    singleton_text = " ".join(singleton_text_words)

    return processed_text, processed_text_words, processed_text_sentences, ex_indices, singleton_text

def find_sent_id(index, data_dict):
    """
    Finds the key in a dictionary where the index falls within the range specified by the value list.

    Args:
        index (int): The index to check.
        data_dict (dict): A dictionary where keys are integers and values are lists with two integers [start, end].

    Returns:
        int or None: The key if the index is within the range, otherwise None.
    """
    for key, value_range in data_dict.items():
        if len(value_range) == 2 and value_range[0] <= index < value_range[1]:
            return key, value_range[0]
    return None, None

def process_text_maven_ere(text):
    # Find all occurrences of {Exx word}
    pattern = r"\{(E\d+)\s+(.*?)\}"
    matches = list(re.finditer(pattern, text))
    #print(len(matches))

    processed_text = text
    ex_indices = {}

    # Replace each {Exx word} with the word and save its absolute index
    for match in matches:
        ex_tag = match.group(1)  # E.g., "E1"
        word = match.group(2)    # Word within the curly braces

        # Replace the pattern in the text with the word only
        #print(match.group(0), "{"+ex_tag+"_"+word+"}")
        processed_text = processed_text.replace(match.group(0), " "+word+" ")
    # Create sentence list for processed text
    processed_text_sentences = re.split(r'(?<=[.!?])\s+', processed_text.strip())
    # Create token list for processed text
    processed_text_tokens = [re.split(r'\s+', processed_text_sentence.strip()) for processed_text_sentence in processed_text_sentences]
    sentence_map = {}
    cumulative_length = 0
    for i, processed_text_token in enumerate(processed_text_tokens):
        sentence_map[i] = [cumulative_length, cumulative_length + len(processed_text_token)]
        cumulative_length += len(processed_text_token)
    # Create word list for processed text
    processed_text_words = re.split(r'\s+', processed_text.strip()) 
    singleton_text_words = re.split(r'\s+', processed_text.strip())

    word_book_keep = {} # Used for keeping the index of last processed word with the same trigger word
    # Find indices of each replaced word in the final processed text
    singleton_index = 0
    global_index = 0
    for match in matches:
        #print(singleton_index, match)
        ex_tag = match.group(1)  # E.g., "E1"
        word = match.group(2)    # Word within the curly braces
        words_split = re.split(r'\s+', word)
         
        #print(len(processed_text_words), len(words_split))
        # Find the starting index of the first word in the processed text
        for i in range(len(processed_text_words)):
            if processed_text_words[i:i + len(words_split)] == words_split:
                if ex_tag not in ex_indices:
                    if word in word_book_keep:
                        if i > word_book_keep[word] and i > global_index:
                            word_book_keep[word] = i
                            global_index = i
                            sent_id, cumulative_offset = find_sent_id(i, sentence_map)
                            ex_indices[ex_tag] = [[word, [i-cumulative_offset, i+len(words_split)-cumulative_offset], "E"+str(singleton_index), sent_id, i]]
                            singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                            singleton_index += 1
                        else:
                            continue
                    else:
                        if i > global_index:
                            global_index = i
                            word_book_keep[word] = i
                            sent_id, cumulative_offset = find_sent_id(i, sentence_map)
                            ex_indices[ex_tag] = [[word, [i-cumulative_offset, i+len(words_split)-cumulative_offset], "E"+str(singleton_index), sent_id, i]]
                            singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                            singleton_index += 1
                        else:
                            continue
                else:
                    already_in = False
                    for mention in ex_indices[ex_tag]:
                        if mention[1] == i:
                            already_in = True
                            break
                    if already_in:
                        continue
                    else:
                        if word in word_book_keep:
                            if i > word_book_keep[word] and i > global_index:
                                global_index = i
                                word_book_keep[word] = i
                                sent_id, cumulative_offset = find_sent_id(i, sentence_map)
                                ex_indices[ex_tag].append([word, [i-cumulative_offset, i+len(words_split)-cumulative_offset], "E"+str(singleton_index), sent_id, i])
                                singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                                singleton_index += 1
                            else:
                                continue
                        else:
                            if i > global_index:
                                global_index = i
                                word_book_keep[word] = i
                                sent_id, cumulative_offset = find_sent_id(i, sentence_map)
                                ex_indices[ex_tag].append([word, [i-cumulative_offset, i+len(words_split)-cumulative_offset], "E"+str(singleton_index), sent_id, i])
                                singleton_text_words = replace_elements(singleton_text_words, i, i + len(words_split) - 1, singleton_index)
                                singleton_index += 1
                            else:
                                continue
                break
        #print(global_index, singleton_index, match)
    #print(len(matches))
    singleton_text = " ".join(singleton_text_words)

    return processed_text, processed_text_words, processed_text_sentences, ex_indices, singleton_text, processed_text_tokens

def convert_to_llm_style(file_path):
    input_text = remove_color_and_tokenization(read_text_file(file_path))
    converted_data = {}
    processed_text, processed_words, processed_sentences, indices_dict, singleton_text = process_text(input_text)

    converted_data["id"] = file_path.split("/")[-1].replace(".txt", "")
    converted_data["tokens"] = processed_words
    converted_data["sentences"] = processed_sentences
    converted_data["text"] = processed_text
    converted_data["text_with_events"] = input_text
    converted_data["singleton_text"] = singleton_text
    converted_data["events"] = []

    for event_id, event_mentions in indices_dict.items():
        processed_event = {"id": event_id, "mention": [{"trigger_word": event_mention[0], "offset": event_mention[1], "singleton_id": event_mention[2]} for event_mention in event_mentions]}
        converted_data["events"].append(processed_event)

    return converted_data

def convert_to_maven_ere_style(file_path):
    input_text = remove_color_and_tokenization(read_text_file(file_path))
    converted_data = {}
    processed_text, processed_words, processed_sentences, indices_dict, singleton_text, processed_text_tokens = process_text_maven_ere(input_text)

    converted_data["id"] = file_path.split("/")[-1].replace(".txt", "")
    converted_data["tokens"] = processed_text_tokens
    converted_data["words"] = processed_words
    converted_data["sentences"] = processed_sentences
    converted_data["text"] = processed_text
    converted_data["text_with_events"] = input_text
    converted_data["singleton_text"] = singleton_text
    converted_data["events"] = []

    for event_id, event_mentions in indices_dict.items():
        processed_event = {"id": event_id, "mention": [{"trigger_word": event_mention[0], "offset": event_mention[1], "singleton_id": event_mention[2], "sent_id": event_mention[3], "globle_offset": event_mention[4]} for event_mention in event_mentions]}
        converted_data["events"].append(processed_event)

    return converted_data

def split_by_sections(text):
    # Define a regular expression pattern to match section numbers (e.g., 1., 2., 3. Payment)
    pattern = r'(\d+\.)\s+([A-Za-z0-9\s]+)'
    
    # Find all matches of sections in the text
    sections = re.findall(pattern, text)

    # Initialize an empty list to hold the sections
    split_sections = []
    
    # Add sections to the list
    for i, (section_number, section_title) in enumerate(sections):
        # Find the content after each section header and append to the split_sections list
        start_index = text.find(f"{section_number} {section_title}") + len(f"{section_number} {section_title}")
        end_index = text.find(f"{sections[i + 1][0]} {sections[i + 1][1]}") if i + 1 < len(sections) else len(text)
        split_sections.append(text[start_index:end_index].strip())

    return split_sections

if __name__ == "__main__":
    # Example usage
    
    converted_data = convert_to_maven_ere_style("./data/52.txt")
    #print(converted_data["events"])
    #print(converted_data["singleton_text"])
    sections = split_by_sections(converted_data["text"])
    print(sections)