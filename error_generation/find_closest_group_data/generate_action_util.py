from error_generation.find_closest_group_data.token_level_closest_text import \
    calculate_distance_and_action_between_two_code


def generate_action_between_two_code(error_tokens, ac_tokens, max_distance=None):
    distance, action_list = calculate_distance_and_action_between_two_code(error_tokens, ac_tokens,
                                                                      max_distance=max_distance)
    if distance > max_distance:
        distance = -1
        action_list = None
    return distance, action_list


