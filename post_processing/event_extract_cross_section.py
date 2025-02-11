import re
from collections import defaultdict
import json
from bisect import bisect_left



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
        print(f"Error decoding JSON on line {len(data) + 1}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return data


def find_section_boundaries(token_list):
    """
    Identify section boundaries (start and end indices) in a token list.

    Args:
        token_list (list): List of tokens.

    Returns:
        list: List of tuples (start_index, end_index) for each section.
    """
    section_pattern = re.compile(r'^(\d+)\.(\d*)$')  # Matches "1.", "1.0", "1.01"

    current_main_section = None  # Track the latest main section
    section_dict = defaultdict(list)

    section_start = None  # Track the start index of the current section
    section_list=[]
    for i, token in enumerate(token_list):
        match = section_pattern.match(token)
        if match:
            main_section = match.group(1)  # Extract main section number

            # 如果是新的主编号（如 1 → 2），存储上一个 section 的结束位置
            if current_main_section is not None and main_section != current_main_section:
                section_dict[current_main_section].append((section_start, i - 1))  # 上一个 section 结束
                section_list.append(section_start)
            # 只有当是新的主编号时，才更新 current_main_section 和 section_start
            if main_section != current_main_section:
                current_main_section = main_section
                section_start = i  # 更新起始位置

        # 处理最后一个 section
    if current_main_section is not None:
        section_dict[current_main_section].append((section_start, len(token_list) - 1))
        section_list.append(section_start)

    #table of content
    section_dict["TOC"]=(0,section_dict["1"][0][0])
    return section_dict,section_list

def is_local(offset1,offset2,section_list):
    return bisect_left(section_list,offset1)==bisect_left(section_list,offset2)

def is_local_cluster(cluster,section_list):
    n=len(cluster)

    for i in range(n):
        offset1, keyword1=cluster[i]
        for j in range(i+1,n):
            offset2,keyword2=cluster[j]
            if not is_local(offset1,offset2,section_list):
                return False
    return True


if __name__ == "__main__":
    # Example usage
    data_path="../annotation_validation/completed_data.jsonl"
    # converted_data = convert_to_maven_ere_style(data_path)
    data=load_jsonl(data_path)
    # Split token list into sections
    token_list=data[0]["tokens"]
    cluster=[[9, "omitted"], [37, "filed"], [1288, "filed"], [1279, "omitted"], [500, "omitted"]]
    # print(token_list[958:])
    section_dict,section_list=find_section_boundaries(token_list)
    # Print sections
    print(section_dict)
    print(section_list)
    print(is_local_cluster(cluster,section_list))
