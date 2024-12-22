from collections import defaultdict
import numpy as np

def calculate_f1_score(predicted_list, gold_list):
    # Convert lists to sets of tuples for easier comparison
    predicted_set = set(predicted_list)
    gold_set = set(gold_list)
    
    print(predicted_set)
    print(gold_set)
    # Calculate True Positives, False Positives, and False Negatives
    true_positives = predicted_set & gold_set
    false_positives = predicted_set - gold_set
    false_negatives = gold_set - predicted_set
    
    # Precision, Recall, and F1 Score
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if predicted_set else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if gold_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives)
    }

def calculate_micro_macro_f1(all_predicted, all_gold):
    # Validate input sizes
    assert len(all_predicted) == len(all_gold), "Mismatch in the number of list pairs!"
    
    # Metrics for micro F1
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    # Metrics for macro F1
    macro_f1_scores = []
    macro_precision_scores = []
    macro_recall_scores = []
    
    for predicted, gold in zip(all_predicted, all_gold):
        result = calculate_f1_score(predicted, gold)
        total_true_positives += result['true_positives']
        total_false_positives += result['false_positives']
        total_false_negatives += result['false_negatives']
        macro_f1_scores.append(result['f1'])
        macro_precision_scores.append(result['precision'])
        macro_recall_scores.append(result['recall'])
    
    # Calculate Micro Precision, Recall, and F1
    micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Calculate Macro F1
    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0
    macro_precision = sum(macro_precision_scores) / len(macro_precision_scores) if macro_precision_scores else 0
    macro_recall = sum(macro_recall_scores) / len(macro_recall_scores) if macro_recall_scores else 0
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
    }

def calculate_muc_score(true_clusters, predicted_clusters):
    """
    Calculate the MUC score between two sets of clusters.

    Parameters:
        true_clusters (list of list of tuples):
            The ground truth clusters, where each cluster is a list of (offset, trigger_word) tuples.
        predicted_clusters (list of list of tuples):
            The predicted clusters, where each cluster is a list of (offset, trigger_word) tuples.

    Returns:
        float: The MUC score, calculated as the F1 score of precision and recall.
    """
    def count_links(clusters):
        """Count the number of links in the given clusters."""
        links = 0
        for cluster in clusters:
            size = len(cluster)
            if size > 1:
                links += size - 1
        return links

    def count_shared_links(true_clusters, predicted_clusters):
        """Count the shared links between true and predicted clusters."""
        shared_links = 0
        for true_cluster in true_clusters:
            true_set = set(true_cluster)
            for predicted_cluster in predicted_clusters:
                predicted_set = set(predicted_cluster)
                overlap = true_set & predicted_set
                if len(overlap) > 1:
                    shared_links += len(overlap) - 1
        return shared_links

    # Count the total links in true and predicted clusters
    total_true_links = count_links(true_clusters)
    total_predicted_links = count_links(predicted_clusters)

    # Count the shared links between true and predicted clusters
    shared_links = count_shared_links(true_clusters, predicted_clusters)

    # Calculate precision and recall
    precision = shared_links / total_predicted_links if total_predicted_links > 0 else 0.0
    recall = shared_links / total_true_links if total_true_links > 0 else 0.0

    # Calculate F1 score
    if precision + recall > 0:
        muc_score = 2 * (precision * recall) / (precision + recall)
    else:
        muc_score = 0.0

    return precision, recall, muc_score

def calculate_micro_macro_muc(all_true_clusters, all_predicted_clusters):
    """
    Calculate the micro and macro MUC scores across multiple cluster sets.

    Parameters:
        all_true_clusters (list of list of list of tuples):
            A list of ground truth cluster sets, where each set contains clusters.
        all_predicted_clusters (list of list of list of tuples):
            A list of predicted cluster sets, where each set contains clusters.

    Returns:
        tuple: A tuple containing micro and macro MUC precision, recall, and F1 scores.
    """
    total_shared_links = 0
    total_true_links = 0
    total_predicted_links = 0
    individual_scores = []
    individual_precisions = []
    individual_recalls = []

    for true_clusters, predicted_clusters in zip(all_true_clusters, all_predicted_clusters):
        # Count shared, true, and predicted links for each set
        def count_links(clusters):
            links = 0
            for cluster in clusters:
                size = len(cluster)
                if size > 1:
                    links += size - 1
            return links

        def count_shared_links(true_clusters, predicted_clusters):
            shared_links = 0
            for true_cluster in true_clusters:
                true_set = set(true_cluster)
                for predicted_cluster in predicted_clusters:
                    predicted_set = set(predicted_cluster)
                    overlap = true_set & predicted_set
                    if len(overlap) > 1:
                        shared_links += len(overlap) - 1
            return shared_links

        shared_links = count_shared_links(true_clusters, predicted_clusters)
        true_links = count_links(true_clusters)
        predicted_links = count_links(predicted_clusters)

        # Update totals for micro averaging
        total_shared_links += shared_links
        total_true_links += true_links
        total_predicted_links += predicted_links

        # Calculate individual precision, recall, and F1 score for macro averaging
        precision = shared_links / predicted_links if predicted_links > 0 else 0.0
        recall = shared_links / true_links if true_links > 0 else 0.0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        individual_scores.append(f1_score)
        individual_precisions.append(precision)
        individual_recalls.append(recall)

    # Calculate micro precision, recall, and F1 score
    if total_predicted_links > 0 and total_true_links > 0:
        micro_precision = total_shared_links / total_predicted_links
        micro_recall = total_shared_links / total_true_links
        if micro_precision + micro_recall > 0:
            micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        else:
            micro_f1_score = 0.0
    else:
        micro_precision = 0.0
        micro_recall = 0.0
        micro_f1_score = 0.0

    # Calculate macro precision, recall, and F1 score
    macro_precision = sum(individual_precisions) / len(individual_precisions) if individual_precisions else 0.0
    macro_recall = sum(individual_recalls) / len(individual_recalls) if individual_recalls else 0.0
    macro_f1_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0

    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1_score
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1_score
        }
    }

def calculate_b3_score(reference_clusters, predicted_clusters):
    """
    Calculate B^3 precision, recall, and F1 score between reference and predicted clusters.

    Args:
        reference_clusters (list): List of reference clusters, where each cluster is a list of (offset, trigger_word) tuples.
        predicted_clusters (list): List of predicted clusters, where each cluster is a list of (offset, trigger_word) tuples.

    Returns:
        tuple: precision, recall, and F1 score.
    """
    # Create mappings of elements to their respective clusters for reference and predicted
    ref_map = {}
    for cluster in reference_clusters:
        for element in cluster:
            ref_map[element] = cluster

    pred_map = {}
    for cluster in predicted_clusters:
        for element in cluster:
            pred_map[element] = cluster

    # Initialize precision and recall sums
    precision_sum = 0
    recall_sum = 0

    # Calculate precision and recall for each element
    all_elements = set(ref_map.keys()) | set(pred_map.keys())
    for element in all_elements:
        ref_cluster = ref_map.get(element, set())
        pred_cluster = pred_map.get(element, set())

        intersection_size = len(set(ref_cluster) & set(pred_cluster))
        if len(pred_cluster) > 0:
            precision_sum += intersection_size / len(pred_cluster)
        if len(ref_cluster) > 0:
            recall_sum += intersection_size / len(ref_cluster)

    # Total number of elements
    n_elements = len(all_elements)

    # Calculate precision, recall, and F1
    precision = precision_sum / n_elements
    recall = recall_sum / n_elements
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def calculate_micro_macro_b3(reference_clusters_list, predicted_clusters_list):
    """
    Calculate micro and macro B^3 precision, recall, and F1 scores.

    Args:
        reference_clusters_list (list): List of reference cluster sets (each a list of clusters).
        predicted_clusters_list (list): List of predicted cluster sets (each a list of clusters).

    Returns:
        dict: Micro and macro precision, recall, and F1 scores.
    """
    # Micro-level accumulators
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_items = 0

    # Macro-level accumulators
    macro_precision_sum = 0
    macro_recall_sum = 0
    macro_f1_sum = 0
    num_documents = len(reference_clusters_list)

    for ref_clusters, pred_clusters in zip(reference_clusters_list, predicted_clusters_list):
        precision, recall, f1 = calculate_b3_score(ref_clusters, pred_clusters)

        # Update micro-level sums
        total_precision += precision * len(ref_clusters)
        total_recall += recall * len(ref_clusters)
        total_f1 += f1 * len(ref_clusters)
        total_items += len(ref_clusters)

        # Update macro-level sums
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1

    # Calculate micro scores
    micro_precision = total_precision / total_items if total_items > 0 else 0.0
    micro_recall = total_recall / total_items if total_items > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Calculate macro scores
    macro_precision = macro_precision_sum / num_documents if num_documents > 0 else 0.0
    macro_recall = macro_recall_sum / num_documents if num_documents > 0 else 0.0
    macro_f1 = macro_f1_sum / num_documents if num_documents > 0 else 0.0

    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
    }

def calculate_ceaf_e_score(reference_clusters, predicted_clusters):
    """
    Calculate CEAF_e precision, recall, and F1 score between reference and predicted clusters.

    Args:
        reference_clusters (list): List of reference clusters, where each cluster is a list of (offset, trigger_word) tuples.
        predicted_clusters (list): List of predicted clusters, where each cluster is a list of (offset, trigger_word) tuples.

    Returns:
        tuple: precision, recall, and F1 score.
    """
    # Helper function to calculate similarity
    def phi4(cluster1, cluster2):
        return len(set(cluster1) & set(cluster2))

    # Create similarity matrix
    num_ref = len(reference_clusters)
    num_pred = len(predicted_clusters)
    similarity_matrix = np.zeros((num_ref, num_pred))

    for i, ref_cluster in enumerate(reference_clusters):
        for j, pred_cluster in enumerate(predicted_clusters):
            similarity_matrix[i][j] = phi4(ref_cluster, pred_cluster)

    # Solve assignment problem (max bipartite matching)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    # Calculate total similarity
    total_similarity = similarity_matrix[row_ind, col_ind].sum()

    # Calculate precision, recall, and F1
    total_predicted = sum(len(cluster) for cluster in predicted_clusters)
    total_reference = sum(len(cluster) for cluster in reference_clusters)

    precision = total_similarity / total_predicted if total_predicted > 0 else 0.0
    recall = total_similarity / total_reference if total_reference > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def calculate_micro_macro_ceaf_e(reference_clusters_list, predicted_clusters_list):
    """
    Calculate micro and macro CEAF_e precision, recall, and F1 scores.

    Args:
        reference_clusters_list (list): List of reference cluster sets (each a list of clusters).
        predicted_clusters_list (list): List of predicted cluster sets (each a list of clusters).

    Returns:
        dict: Micro and macro precision, recall, and F1 scores.
    """
    # Micro-level accumulators
    total_similarity = 0
    total_predicted = 0
    total_reference = 0

    # Macro-level accumulators
    macro_precision_sum = 0
    macro_recall_sum = 0
    macro_f1_sum = 0
    num_documents = len(reference_clusters_list)

    for ref_clusters, pred_clusters in zip(reference_clusters_list, predicted_clusters_list):
        precision, recall, f1 = calculate_ceaf_e_score(ref_clusters, pred_clusters)

        # Update micro-level sums
        total_similarity += sum(len(set(cluster) & set(pred_clusters[i])) for i, cluster in enumerate(ref_clusters))
        total_predicted += sum(len(cluster) for cluster in pred_clusters)
        total_reference += sum(len(cluster) for cluster in ref_clusters)

        # Update macro-level sums
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1

    # Calculate micro scores
    micro_precision = total_similarity / total_predicted if total_predicted > 0 else 0.0
    micro_recall = total_similarity / total_reference if total_reference > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Calculate macro scores
    macro_precision = macro_precision_sum / num_documents if num_documents > 0 else 0.0
    macro_recall = macro_recall_sum / num_documents if num_documents > 0 else 0.0
    macro_f1 = macro_f1_sum / num_documents if num_documents > 0 else 0.0

    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
    }

def calculate_blanc_score(reference_clusters, predicted_clusters):
    """
    Calculate BLANC precision, recall, and F1 score between reference and predicted clusters.

    Args:
        reference_clusters (list): List of reference clusters, where each cluster is a list of (offset, trigger_word) tuples.
        predicted_clusters (list): List of predicted clusters, where each cluster is a list of (offset, trigger_word) tuples.

    Returns:
        tuple: precision, recall, and F1 score.
    """
    # Create sets of pairs for reference and predicted clusters
    def generate_pairs(clusters):
        pairs = set()
        for cluster in clusters:
            pairs.update({tuple(sorted((a, b))) for a in cluster for b in cluster if a != b})
        return pairs

    ref_pairs = generate_pairs(reference_clusters)
    pred_pairs = generate_pairs(predicted_clusters)

    # True positives, false positives, and false negatives
    true_positives = len(ref_pairs & pred_pairs)
    false_positives = len(pred_pairs - ref_pairs)
    false_negatives = len(ref_pairs - pred_pairs)

    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def calculate_micro_macro_blanc(reference_clusters_list, predicted_clusters_list):
    """
    Calculate micro and macro BLANC precision, recall, and F1 scores.

    Args:
        reference_clusters_list (list): List of reference cluster sets (each a list of clusters).
        predicted_clusters_list (list): List of predicted cluster sets (each a list of clusters).

    Returns:
        dict: Micro and macro precision, recall, and F1 scores.
    """
    # Create sets of pairs for reference and predicted clusters
    def generate_pairs(clusters):
        pairs = set()
        for cluster in clusters:
            pairs.update({tuple(sorted((a, b))) for a in cluster for b in cluster if a != b})
        return pairs

    # Micro-level accumulators
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    # Macro-level accumulators
    macro_precision_sum = 0
    macro_recall_sum = 0
    macro_f1_sum = 0
    num_documents = len(reference_clusters_list)

    for ref_clusters, pred_clusters in zip(reference_clusters_list, predicted_clusters_list):
        precision, recall, f1 = calculate_blanc_score(ref_clusters, pred_clusters)

        # Update micro-level sums
        ref_pairs = generate_pairs(ref_clusters)
        pred_pairs = generate_pairs(pred_clusters)

        true_positives = len(ref_pairs & pred_pairs)
        false_positives = len(pred_pairs - ref_pairs)
        false_negatives = len(ref_pairs - pred_pairs)

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Update macro-level sums
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1

    # Calculate micro scores
    micro_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0.0
    micro_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Calculate macro scores
    macro_precision = macro_precision_sum / num_documents if num_documents > 0 else 0.0
    macro_recall = macro_recall_sum / num_documents if num_documents > 0 else 0.0
    macro_f1 = macro_f1_sum / num_documents if num_documents > 0 else 0.0

    return {
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        }
    }

def save_metrics_to_file(metrics, file_path):
    """
    Saves the given metrics in a specific format to a text file.

    Args:
        micro_f1 (float): Micro F1 score.
        macro_f1 (float): Macro F1 score.
        micro_precision (float): Micro Precision score.
        macro_precision (float): Macro Precision score.
        micro_recall (float): Micro Recall score.
        macro_recall (float): Macro Recall score.
        file_path (str): Path to the file where metrics will be saved.
    """

    with open(file_path, 'a') as file:
        file.write(str(metrics))

if __name__ == "__main__":
    reference_clusters = [[(39, 'located'), (2, 'Redactions'), (60, 'located'), (88, 'wish'), (9, 'denoted'), (20, 'made'), (95, 'development'), (73, 'referred'), (90, 'collaborate'), (93, 'discovery')]]
    predicted_clusters = [[(9, 'denoted'), (20, 'made'), (2, 'Redactions'), (73, 'referred'), (39, 'located'), (60, 'located')], [(93, 'discovery'), (95, 'development'), (88, 'wish'), (90, 'collaborate')]]

    print(calculate_muc_score(reference_clusters, predicted_clusters))
    print(calculate_b3_score(reference_clusters, predicted_clusters))
    print(calculate_ceaf_e_score(reference_clusters, predicted_clusters))
    print(calculate_blanc_score(reference_clusters, predicted_clusters))

    reference_clusters_list = [[[(39, 'located'), (2, 'Redactions'), (60, 'located'), (88, 'wish'), (9, 'denoted'), (20, 'made'), (95, 'development'), (73, 'referred'), (90, 'collaborate'), (93, 'discovery')]], [[(73, 'referred'), (2, 'Redactions'), (39, 'located'), (9, 'denoted'), (60, 'located')], [(88, 'wish'), (90, 'collaborate'), (93, 'discovery'), (20, 'made'), (95, 'development')]]]
    predicted_clusters_list = [[[(9, 'denoted'), (20, 'made'), (2, 'Redactions'), (73, 'referred'), (39, 'located'), (60, 'located')], [(93, 'discovery'), (95, 'development'), (88, 'wish'), (90, 'collaborate')]], [[(9, 'denoted'), (20, 'made'), (2, 'Redactions'), (73, 'referred'), (39, 'located'), (60, 'located')], [(93, 'discovery'), (95, 'development'), (88, 'wish'), (90, 'collaborate')]]]

    print(calculate_micro_macro_muc(reference_clusters_list, predicted_clusters_list))
    print(calculate_micro_macro_b3(reference_clusters_list, predicted_clusters_list))
    print(calculate_micro_macro_ceaf_e(reference_clusters_list, predicted_clusters_list))
    print(calculate_micro_macro_blanc(reference_clusters_list, predicted_clusters_list))