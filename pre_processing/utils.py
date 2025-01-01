import os
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

def convert_to_maven_ere_style(file_path):
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


def sanitize_model_name(model_name):
    """Sanitize model name by replacing special characters with underscores."""
    return re.sub(r'[^\w]', '_', model_name)


def generate_paths(base_dir, task_name, model_name, prompt_setting):
    """
    Generate output and result file paths dynamically based on task name, model, and prompt setting.

    Args:
        base_dir (str): Base directory for the annotation results.
        task_name (str): The name of the task (e.g., 'event_detection').
        model_name (str): Full model name.
        prompt_setting (str): Prompt setting (e.g., 'zero_shot').

    Returns:
        tuple: output_file, final_result_file
    """
    sanitized_model_name = sanitize_model_name(model_name)
    task_dir = os.path.join(base_dir, task_name, sanitized_model_name, prompt_setting)
    os.makedirs(task_dir, exist_ok=True)  # Ensure directory exists
    output_file = os.path.join(task_dir, f"{task_name}_output.jsonl")
    final_result_file = os.path.join(task_dir, f"{task_name}_result.txt")
    return output_file, final_result_file

if __name__ == "__main__":
    # Example usage
    
    converted_data = convert_to_maven_ere_style("./annotation_validation/jonathan_annotations/52.txt")
    #print(converted_data["events"])
    #print(converted_data["singleton_text"])
    sections = split_by_sections(converted_data["text"])
    print(sections)
    #print(get_words_by_index(converted_data["text"], 2490))
    '''input_text = read_text_file("/Users/kangdawei/Desktop/Research/Legal-Coreference/annotation_validation/jonathan_annotations/52.txt")

    processed_text, processed_words, processed_sentences, indices_dict = process_text(input_text)

    print("Processed Text:")
    print(processed_text)
    print("\nProcessed Text Words:")
    print(processed_words)
    print("\nProcessed Text Sentences:")
    print(processed_sentences)
    print("\nIndices Dictionary:")
    print(indices_dict)'''