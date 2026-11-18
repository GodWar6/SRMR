
SRMR_base_v1_prompt = \
    "The user has interacted in chronological order with items: {inters}. " \
    "Here is the candidate set for the next recommendation (candidate set size is [[candidate_set_size]]): {candidate_set}. " \
    "Please recommend from the candidate set. List the top 10 recommendations in numbered bullet points. Note that your reommendation should be RANKED. " \
    "Don't output any other content."

SRMR_base_prompt = \
    "The user has interacted in chronological order with items: {inters}. " \
    "Here is the candidate set for the next recommendation (candidate set size is [[candidate_set_size]]): {candidate_set}. " \
    "Please recommend from the candidate set. List the top 10 recommendations in numbered bullet points. " \
    "Don't output any other content."
'''
SRMR_wo_reflect_prompt = "[[example]]I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
    "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidates. You can not generate candidates that are not in the given candidate list."
'''
SRMR_wo_reflect_prompt = \
    "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records (pay special attention to the most recent purchases).\n" \
    "CRITICAL OUTPUT FORMAT:\n" \
    "You MUST output exactly 10 items. \n" \
    "You MUST format each line strictly as '1. Item Name'. \n" \
    "DO NOT output single or double quotes around the item names. \n" \
    "DO NOT output any explanations, reasoning, or extra words after the item name. \n" \
    "DO NOT output any other content."

SRMR_w_reflect = \
    "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "And after last previous recommendation (maybe a failed trial), you gives the following reflects:\n{reflect}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records and reflects. Please think step by step.\n" \
    "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidates. You can not generate candidates that are not in the given candidate list."

SRMR_w_reflect_v1 = \
    "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "And after last previous recommendation (maybe a failed trial), you give the following reflects:\n{reflect}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records and reflects. Please think step by step.\n" \
    "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidates. You can not generate candidates that are not in the given candidate list."

SRMR_w_reflect_v2 = \
    "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "And after last previous recommendation for me (maybe a failed trial), you receive the following reflects:\n{reflect}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records and reflects. Please think step by step.\n" \
    "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidates. You can not generate candidates that are not in the given candidate list."

SRMR_w_reflect_v3 = \
    "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "And after last previous recommendation, you gave the following reflections:\n{reflect}\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records (focusing on recent trends) and the reflections.\n" \
    "CRITICAL OUTPUT FORMAT:\n" \
    "You MUST output exactly 10 items. \n" \
    "You MUST format each line strictly as '1. Item Name'. \n" \
    "DO NOT output single or double quotes around the item names. \n" \
    "DO NOT output any explanations, reasoning, or extra words after the item name. \n" \
    "DO NOT output any other content."

SRMR_w_reflect_v4 = \
    "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "And after last previous recommendation (maybe a failed trial), you receive the following reflects:\n{reflect}\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records and reflects. Please think step by step.\n" \
    "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidates. You can not generate candidates that are not in the given candidate list."


rec_prompts = {
    "SRMR_wo_reflect": SRMR_wo_reflect_prompt,
    "SRMR_base": SRMR_base_prompt,
    "SRMR_base_v1": SRMR_base_v1_prompt,
    "SRMR_w_reflect_v0": SRMR_w_reflect,
    "SRMR_w_reflect_v1": SRMR_w_reflect_v1,
    "SRMR_w_reflect_v2": SRMR_w_reflect_v2,
    "SRMR_w_reflect_v3": SRMR_w_reflect_v3,
    "SRMR_w_reflect_v4": SRMR_w_reflect_v4,
}

