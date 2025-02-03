import re
import json
import os

def append_to_jsonl(file_path, data):
    """
    Append a dictionary to a JSONL file. 
    If the file doesn't exist, create a new one.
    
    :param file_path: Path to the JSONL file.
    :param data: Dictionary to append.
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    
    # Open the file in append mode, creating it if it doesn't exist
    with open(file_path, 'a') as file:
        # Write the dictionary as a JSON string followed by a newline
        file.write(json.dumps(data) + '\n')
        
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

def extract_spans_and_triggers(raw_trigger_words):
    # Regular expression pattern to match the Span and Trigger format
    pattern = r"Span:\s(\d+-\d+)\s+Trigger:\s(\w+)"
    matches = re.findall(pattern, raw_trigger_words)
    
    # Convert matches to a list of dictionaries
    result = [{"offset": match[0], "trigger_word": match[1]} for match in matches]
    
    return result

def update_offsets(processed_trigger_words, text):
    # Tokenize the text into words
    words = text.split()

    # Update the offsets with integer indices based on proximity to the original offset
    for word in processed_trigger_words:
        trigger_word = word.get("trigger_word")
        original_offset = word.get("offset")

        # Convert original offset to integer range if it exists
        if isinstance(original_offset, str) and "-" in original_offset:
            start, end = map(int, original_offset.split("-"))
        else:
            start, end = -1, -1

        # Find all occurrences of the trigger word and their positions
        occurrences = [i for i, w in enumerate(words) if w.strip(',') == trigger_word]

        # Select the occurrence closest to the original offset start
        if occurrences:
            closest_index = min(occurrences, key=lambda x: abs(x - start) if start != -1 else x)
            word['offset'] = closest_index
        else:
            # If the word is not found, set offset to -1
            word['offset'] = -1

    return processed_trigger_words

def extract_mentions(data):
    # Initialize an empty list to store the tuples
    mention_list = []
    
    # Iterate over each dictionary in the input list
    for item in data:
        # Extract the 'mention' list and process each dictionary within it
        for mention in item.get('mention', []):
            # Append a tuple of (offset, trigger_word) to the list
            mention_list.append((mention['offset'], mention['trigger_word']))
    
    return mention_list

def process_coreference(text):
    """
    Processes a coreference text into a list of tuples.

    Args:
        text (str): The input text containing coreference relationships.

    Returns:
        list: A list of tuples representing the coreference relationships.
    """
    print(text)
    # Split the text into lines
    lines = text.strip().split("\n")

    # Initialize an empty list to store the tuples
    coreference_tuples = []

    # Process each line
    # for line in lines:
    #     # Split the line by spaces and extract the elements
    #     elements = line.split()
    #     if len(elements) == 3 and elements[1] == "COREFERENCE":
    #         coreference_tuples.append((elements[0], elements[2]))
    #
    # return coreference_tuples
    for line in lines:
        # Split the line by spaces and extract the elements
        elements = line.split("COREFERENCE")
        if len(elements) == 2:
            # Extract numeric index using regex
            num1 = re.search(r'\d+', elements[0])  # Find number in first element
            num2 = re.search(r'\d+', elements[1])  # Find number in second element

            if num1 and num2:
                # Convert extracted numbers back to EXX format
                exx1 = f"E{int(num1.group())}"  # E0, E1, E25, etc.
                exx2 = f"E{int(num2.group())}"

                coreference_tuples.append((exx1, exx2))

    return coreference_tuples

def create_coreference_clusters(coreference_tuples):
    """
    Creates coreference clusters from a list of coreference tuples.

    Args:
        coreference_tuples (list): A list of tuples representing coreference relationships.

    Returns:
        list: A list of clusters, where each cluster is a set of coreferent elements.
    """
    clusters = []
    element_to_cluster = {}

    for e1, e2 in coreference_tuples:
        if e1 in element_to_cluster and e2 in element_to_cluster:
            # Merge two clusters if both elements already have clusters
            if element_to_cluster[e1] != element_to_cluster[e2]:
                cluster1 = element_to_cluster[e1]
                cluster2 = element_to_cluster[e2]
                cluster1.update(cluster2)
                for elem in cluster2:
                    element_to_cluster[elem] = cluster1
                clusters.remove(cluster2)
        elif e1 in element_to_cluster:
            # Add e2 to e1's cluster
            cluster = element_to_cluster[e1]
            cluster.add(e2)
            element_to_cluster[e2] = cluster
        elif e2 in element_to_cluster:
            # Add e1 to e2's cluster
            cluster = element_to_cluster[e2]
            cluster.add(e1)
            element_to_cluster[e1] = cluster
        else:
            # Create a new cluster for e1 and e2
            new_cluster = {e1, e2}
            clusters.append(new_cluster)
            element_to_cluster[e1] = new_cluster
            element_to_cluster[e2] = new_cluster

    return [list(cluster) for cluster in clusters]


def replace_elements_with_mentions(clusters, mention_list):
    """
    Replaces elements in clusters with their corresponding (offset, trigger_word) tuples.

    Args:
        clusters (list): A list of clusters, where each cluster is a list of elements.
        mention_list (list): A list of dictionaries containing element mentions with offsets and trigger words.

    Returns:
        list: A list of clusters with elements replaced by their (offset, trigger_word) tuples.
    """
    # Create a mapping from singleton_id to (offset, trigger_word)
    id_to_mention = {
        mention["singleton_id"]: (mention["offset"], mention["trigger_word"])
        for entry in mention_list
        for mention in entry["mention"]
    }

    # Replace elements in clusters
    # replaced_clusters = []
    # for cluster in clusters:
    #     replaced_cluster = [id_to_mention.get(element, element) for element in cluster]
    #     replaced_clusters.append(replaced_cluster)
    replaced_clusters = []
    for cluster in clusters:
        replaced_cluster = [
            id_to_mention[element] for element in cluster if element in id_to_mention
        ]
        if replaced_cluster:
            replaced_clusters.append(replaced_cluster)

    return replaced_clusters

def mentions_to_clusters(mentions):
    """
    Converts a mention list into clusters of tuples in the form of (offset, trigger_word).

    Args:
        mentions (list): List of dictionaries containing mentions with id and trigger information.

    Returns:
        list: List of clusters, where each cluster is a list of (offset, trigger_word) tuples.
    """
    clusters = []
    for mention_group in mentions:
        cluster = []
        for mention in mention_group["mention"]:
            cluster.append((mention["offset"], mention["trigger_word"]))
        clusters.append(cluster)
    return clusters

if __name__ == "__main__":
    text = """
        E0 COREFERENCE E11
        E2 COREFERENCE E7
        E3 COREFERENCE E7
        E4 COREFERENCE E10
        E5 COREFERENCE E10
        E7 COREFERENCE E8
        E8 COREFERENCE E6
        E9 COREFERENCE E8
        E9 COREFERENCE E11
        """
    mention_list = [{"id": "E158", "mention": [{"trigger_word": "Redactions", "offset": 2, "singleton_id": "E0"}, {"trigger_word": "gg", "offset": 99, "singleton_id": "E11"}]}, {"id": "E1", "mention": [{"trigger_word": "denoted", "offset": 9, "singleton_id": "E1"}]}, {"id": "E2", "mention": [{"trigger_word": "made", "offset": 20, "singleton_id": "E2"}]}, {"id": "E3", "mention": [{"trigger_word": "located", "offset": 39, "singleton_id": "E3"}]}, {"id": "E4", "mention": [{"trigger_word": "located", "offset": 60, "singleton_id": "E4"}]}, {"id": "E5", "mention": [{"trigger_word": "referred", "offset": 73, "singleton_id": "E5"}]}, {"id": "E6", "mention": [{"trigger_word": "wish", "offset": 88, "singleton_id": "E6"}]}, {"id": "E7", "mention": [{"trigger_word": "collaborate", "offset": 90, "singleton_id": "E7"}]}, {"id": "E159", "mention": [{"trigger_word": "discovery", "offset": 93, "singleton_id": "E8"}, {"trigger_word": "discovery", "offset": 328, "singleton_id": "E18"}]}, {"id": "E160", "mention": [{"trigger_word": "development", "offset": 95, "singleton_id": "E9"}]}, {"id": "E161", "mention": [{"trigger_word": "treatment", "offset": 102, "singleton_id": "E10"}]}]

    coreference_tuples = process_coreference(text)
    print(coreference_tuples)
    clusters = create_coreference_clusters(coreference_tuples)
    print(clusters)
    clusters = replace_elements_with_mentions(clusters, mention_list)
    print(clusters)

    gold_clusters = mentions_to_clusters(mention_list)
    print(gold_clusters)
    '''# Example usage
    raw_trigger_words = """
    Span: 1-4 
    Trigger: acknowledges 

    Span: 5-14 
    Trigger: cooperate 

    Span: 15-24 
    Trigger: logistics 

    Span: 25-31 
    Trigger: management 

    Span: 32-37 
    Trigger: storage 

    Span: 38-46 
    Trigger: dispensation
    """

    text = "Sponsor  acknowledges  that Sponsor shall  cooperate  with the Concessionaire  regarding  logistics and management of the Sponsor's food products, and appropriate storage and dispensation of the food products."
    spans_and_triggers = extract_spans_and_triggers(raw_trigger_words)
    print(spans_and_triggers)

    processed_spans_and_triggers = update_offsets(spans_and_triggers, text)
    print(processed_spans_and_triggers)'''

    '''all_data = load_jsonl("/scratch/user/kangda/Legal-Coreference/annotation_validation/jonathan_annotations/data.jsonl")
    data = all_data[0]
    data = data["events"]
    data = extract_mentions(data)
    print(data)'''