fs_prompt_end = 'Please provide your response based on Previous recommendations and Examples below:'

instruction_meta_example = "The user has interacted with items [[inters_1]] in chronological order. "\
    "Here is the candidate set for the next recommendation: [[candidate_set_1]]. "\
    "Please recommend from the candidate set. List the top 10 recommendations in numbered bullet points."

instruction_meta_example_v2 = "I've purchased the following products in the past in order:\n[[inters_1]]\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n[[candidate_set_1]]\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
    "Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidates. You can not generate candidates that are not in the given candidate list."


inters_1 = "['Yamaha NTX700 Acoustic Electric Classical Guitar.', 'Behringer UC200 Ultimate Stereo Chorus Instrument Effects Pedal.', 'Behringer Ultra-DI DI20 Professional Active 2-Channel DI-Box/Splitter.', \"D'Addario NYXL1046 Nickel Plated Electric Guitar Strings, Light.\", 'BEHRINGER REVERB MACHINE RV600.', 'BEHRINGER TU300.', 'Behringer VD400 Vintage Analog Delay Effects Pedal.', 'Electro-Harmonix 720 Stereo Looper Pedal.', 'BEHRINGER DIGITAL DELAY DD600.']"
instruction_example1 = f"Below is the instruction for the actor's previous recommendation. The user has interacted with items {inters_1} in chronological order. Please predict the next item that the user would interact with."
response__example1 = "DR Strings Zebra - Acoustic-Electric Round Core 11-50."
last_act_his_str_example1 = "['Behringer TM300 Ultimate Tube Amp Modeling Instrument Effects Pedal.', 'Behringer Compressor/Sustainer CS400 Ultimate Dynamics Effects Pedal.', 'JOYO JF-37 Bundle Acoustic Guitar Knob.', 'Behringer DC9 Dynamics Compressor Pedal.', 'BEHRINGER REVERB MACHINE RV600.', 'Behringer Noise Reducer NR300 Ulitmate Noise Reduction Instrument Effects Pedal.', 'Behringer TO800 Vintage Tube-Sound Overdrive Instrument Effects Pedal.', 'Behringer Ultra Metal UM300 Heavy Metal Distortion Instrument Effect Pedal.', 'Behringer Super Fuzz SF300 3-Mode Fuzz Distortion Instrument Effects Pedal.', 'Behringer EQ700 Ultimate 7-Band Graphic Equalizer.', 'Behringer Ultra Vibrato UV300 Classic Vibrato Instrument Effects Pedal.', 'DAddario Accessories Pro-Winder Guitar String Winder and Cutter  All-In-1 Restringing Tool  Includes Clippers, Bridge Pin Puller, Peg Winder  Designed to Fit Most Guitars.', 'Behringer Vintage Phaser VP1 Authentic Vintage-Style Phase Shifter Instrument Effects Pedal.', 'BEHRINGER TU300.', 'Behringer Hellbabe HB01 Ultimate Wah-Wah Pedal with Optical Control.', 'Behringer PH9 Classic 90 Phase Shifter.', 'BBE Sonic Stomp Sonic Maximizer Stomp Box Pedal for Guitars, Basses and Keyboards.', 'Donner Blues Drive Classical Electronic Vintage Overdrive Guitar Effect Pedal True Bypass Warm/Hot Modes.', 'Behringer UC200 Ultimate Stereo Chorus Instrument Effects Pedal.', 'Elixir Strings 80/20 Bronze Acoustic Guitar Strings w POLYWEB Coating, Light/Medium (.012-.056).']"
fs_example1 = "PREFERENCE: Based on the user's preference for guitar accessories and effects pedals, the identified category of music-related items can contribute to better reflection by focusing on similar products that match the user's interests in music equipment. The categories of items in the user's interaction history and previous recommendations mainly consist of guitar accessories and effects pedals, which are music-related items that suggest the user is interested in music equipment.\n" \
            "ANALYSIS: The previous predictions failed to meet the user's demands because they were too focused on Behringer brand products and did not consider a wider range of possible items the user may be interested in.\n" \
            "SUGGESTIONS: In the future, try to diversify recommendations by considering a broader range of brands and types of items that the user may like based on their historical interactions.\n"

inters_2 = "['Dunlop Acoustic Guitar Frets (37083022001).', 'ChromaCast CC-MINIGS Universal Folding Guitar Stand with Secure Lock.', 'GO-DPS ChromaCast Electric Guitar 6-Pocket Padded Gig Bag with Guitar Strap and Pick Sampler.', 'DAddario Accessories Pro-Winder Guitar String Winder and Cutter  All-In-1 Restringing Tool  Includes Clippers, Bridge Pin Puller, Peg Winder  Designed to Fit Most Guitars.', 'Snark SN2 Guitar Tuner.']"
instruction_example2 = f"Below is the instruction for the actor's previous recommendation. The user has interacted with items {inters_2} in chronological order. Please predict the next item that the user would interact with."
response__example2 = "Hosa GPR-101 PAK10 RCA to 1/4 TS Adaptors, Pack of 10."
last_act_his_str_example2 = "['Dunlop Series 5005 Pickholder, 1 Pack.', \"D'Addario EJ17-B25 Phosphor Bronze Acoustic Guitar Strings, Medium, 13-56, 25 Bulk Sets.\", \"D'Addario Assorted Pearl Celluloid Guitar Picks, 10 Pack, Medium.\", \"D'Addario EJ26 Phosphor Bronze Acoustic Guitar Strings, Custom Light, 11-52.\", 'Snark SN1 Guitar Tuner (Blue).', 'Fender Performance Series Microphone Cable for pro audio, and live sound.', 'Elixir Strings 80/20 Bronze Acoustic Guitar Strings w POLYWEB Coating, Extra Light (.010-.047).', 'Elixir Strings 80/20 Bronze Acoustic Guitar Strings w POLYWEB Coating, Light/Medium (.012-.056).', 'Fender 250L Nickel Plated Steel Electric Guitar Strings - Light.', 'Simtyso\u00ae Predmium Folk Acoustic Electric Guitar Banjo Trigger Capo Key Clamp.', 'DAddario Accessories Pro-Winder Guitar String Winder and Cutter  All-In-1 Restringing Tool  Includes Clippers, Bridge Pin Puller, Peg Winder  Designed to Fit Most Guitars.', 'Dunlop 5006SI Ergo Pickholder.', 'Ernie Ball Earthwood Light Phosphor Bronze Acoustic String Set, .011 - .052.', 'Dunlop Tortex Standard .50mm Red Guitar Pick - 12 Pack.', 'Hosa GTR-205 Straight to Straight Guitar Cable, 5 Feet.', 'On-Stage XCG4 Black Tripod Guitar Stand, Single.', 'Fender Clip-On Tuner FT-004 for Guitar, Ukulele, Bass, Violin, Mandolin, and Banjo.', 'Dunlop 471P3C Max Grip Jazz III Carbon Fiber Guitar Picks, 6-Pack.', \"Dunlop 44P.60 Nylon Standard, Light Gray, .60mm, 12/Player's Pack.\", 'Kyser Quick-Change Capo for 6-string acoustic guitars - Black.']"
fs_example2 = "PREFERENCE: The user may have a preference for audio equipment along with guitar gear, so exploring categories related to audio equipment in addition to guitar accessories could lead to better predictions.\n" \
            "ANALYSIS: The categories of items from previous user interactions mainly revolved around guitar accessories and gear, while the target item was an audio adaptor, showing a possible association with audio equipment. The previous recommendations failed because they were focused on guitar accessories and related items, while the user's next interaction was actually an RCA to 1/4 TS adaptor, which is more related to audio equipment than guitar gear.\n" \
            "SUGGESTIONS: Consider broadening the scope of recommendations to include a wider range of audio equipment and accessories, not just limited to guitar-related items.\n"

instruction_example3 = '[[previous_instruction 1]]'
response__example3 = '[[target_item 1]]' 
last_act_his_str_example3 = '[[last_act_his 1]]'
fs_example3 = '[[fs 1]]' 

instruction_example4 = '[[previous_instruction 2]]'
response__example4 = '[[target_item 2]]' 
last_act_his_str_example4 = '[[last_act_his 2]]'
fs_example4 = '[[fs 2]]' 


EP_fs_prompt = "You are an expert at providing constructive reflections to improve a sequential recommender system.\n\n" \
    "**Your Task**: Given the user's historical interactions and the actor's previous top-10 recommendations, analyze why the recommendations may have been suboptimal and suggest concrete improvements for the next recommendation round.\n\n" \
    "**Instructions (follow step by step)**:\n" \
    "1. **PREFERENCE**: In 1-2 sentences, summarize the user's likely preferences based on their interaction history (pay attention to the *most recent* items). Consider:\n" \
    "   - Brand/Manufacturer loyalty\n" \
    "   - Product categories and subcategories\n" \
    "   - Price range (if inferable)\n" \
    "   - Platform/game genre (if applicable)\n" \
    "   - Any obvious trends or shifts in interest.\n\n" \
    "2. **ANALYSIS**: In 1-2 sentences, diagnose the main issues with the actor's previous 10 recommendations. Focus on:\n" \
    "   - Whether the recommendations match the user's stated preferences.\n" \
    "   - Whether they are too narrow (e.g., all same brand) or too broad.\n" \
    "   - Whether they ignore recent trends or include outdated/irrelevant items.\n\n" \
    "3. **SUGGESTIONS**: In 1-2 sentences, give actionable advice for the actor's next recommendations. Be specific, e.g., \"prioritize items from brand X\", \"focus on action-adventure games\", \"include more accessories from the same product line\".\n\n" \
    "**Important Constraints**:\n" \
    "- DO NOT recommend specific item titles in your output; only provide high-level guidance.\n" \
    "- Be concise and directly relevant to the given user and their history.\n\n" \
    "Here are some examples (the actual items are placeholders):\n\n" \
    "## Previous instruction:\n" + instruction_example3 + "\n\n" \
    "## Standard answer (the correct next item):\n" + response__example3 + "\n\n" \
    "## Actor's recommendations (the 10 items they suggested):\n" + last_act_his_str_example3 + "\n\n" \
    "## Your response (example):\n" + fs_example3 + "\n\n" \
    "## Previous instruction:\n" + instruction_example4 + "\n\n" \
    "## Standard answer:\n" + response__example4 + "\n\n" \
    "## Actor's recommendations:\n" + last_act_his_str_example4 + "\n\n" \
    "## Your response:\n" + fs_example4 + "\n\n" \
    "Now, analyze the following case:\n\n" \
    "## Previous instruction:\n{instruction}\n\n" \
    "## Standard answer:\n{response}\n\n" \
    "## Actor's recommendations:\n{last_act_his_str}\n\n" \
    "## Your response:"


item_IP_example1 = "[[item_IP_list 1]]"
item_IP_example2 = "[[item_IP_list 2]]"

IP_fs_prompt = "You are an expert at providing reflections based on item attributes to improve a sequential recommender.\n\n" \
    "**Your Task**: Given the user's interaction history, the actor's previous recommendations, AND the attribute information (e.g., categories, brands, features) of those items, analyze why the recommendations may have missed the mark, and suggest attribute-focused improvements.\n\n" \
    "**Instructions** (follow step by step):\n" \
    "1. **PREFERENCE**: In 1-2 sentences, infer the user's preference patterns from the *attributes* of the items they've interacted with (especially the most recent ones). Look for:\n" \
    "   - Dominant categories (e.g., \"guitar pedals\", \"yarn\", \"action games\")\n" \
    "   - Common brands or materials\n" \
    "   - Feature clusters (e.g., \"vintage-style\", \"wireless\", \"portable\")\n\n" \
    "2. **ANALYSIS**: In 1-2 sentences, pinpoint what went wrong with the actor's recommendations in terms of attributes. For example:\n" \
    "   - Recommendations were too focused on one attribute while ignoring others.\n" \
    "   - They introduced attributes the user never showed interest in.\n" \
    "   - They failed to match the attribute profile of the user's recent interactions.\n\n" \
    "3. **SUGGESTIONS**: In 1-2 sentences, give concrete attribute-level guidance for the next recommendations (e.g., \"prioritize items with 'digital delay' feature\", \"include more 'acrylic' yarns\", \"focus on 'open-world' games\").\n\n" \
    "**Important**:\n" \
    "- Do not output specific item titles.\n" \
    "- Be attribute-driven and specific.\n\n" \
    "Here are some examples (attributes are placeholders):\n\n" \
    "## Previous instruction and items' attributes:\n" + item_IP_example1 + "\n\n" \
    "## Standard answer:\n" + response__example3 + "\n\n" \
    "## Actor's recommendations:\n" + last_act_his_str_example3 + "\n\n" \
    "## Your response:\n" + fs_example3 + "\n\n" \
    "## Previous instruction and items' attributes:\n" + item_IP_example2 + "\n\n" \
    "## Standard answer:\n" + response__example4 + "\n\n" \
    "## Actor's recommendations:\n" + last_act_his_str_example4 + "\n\n" \
    "## Your response:\n" + fs_example4 + "\n\n" \
    "Now, analyze the following case:\n\n" \
    "## Previous instruction and items' attributes:\n{instruction}\n\n" \
    "## Standard answer:\n{response}\n\n" \
    "## Actor's recommendations:\n{last_act_his_str}\n\n" \
    "## Your response:"

IP_fs_prompt_wo_example = "You are an expert at providing reflections based on item attributes to improve a sequential recommender.\n\n" \
    "**Task**: Analyze the user's interaction history, the actor's previous recommendations, and the attributes (titles, categories, brands) of those items. Provide a structured reflection.\n\n" \
    "**Steps**:\n" \
    "1. **PREFERENCE**: 1-2 sentences summarizing the user's likely preference from the attributes of their recent interactions.\n" \
    "2. **ANALYSIS**: 1-2 sentences identifying problems with the actor's recommendations in terms of attributes.\n" \
    "3. **SUGGESTIONS**: 1-2 sentences giving attribute-level guidance for future recommendations.\n\n" \
    "**Constraints**:\n" \
    "- Do not output item titles.\n" \
    "- Be specific and actionable.\n\n" \
    "Now analyze the following:\n\n" \
    "## Previous instruction and items' attributes (item titles, categories and brands):\n{instruction}\n\n" \
    "## Standard answer:\n{response}\n\n" \
    "## Actor's recommendations:\n{last_act_his_str}\n\n" \
    "## Your response:"

item_CF_scores_example1 = "[[item_CF_scores_list 1]]"
item_CF_scores_example2 = "[[item_CF_scores_list 2]]"

CF_fs_prompt = "You are an expert at providing reflections based on collaborative filtering scores to enhance a sequential recommender.\n\n" \
    "**Your Task**: Given the user's history, the actor's recommendations, AND the collaborative filtering (CF) scores (which indicate predicted relevance of each recommended item to the user), analyze why the actor may have chosen items with low CF scores, and suggest how to incorporate CF signals better.\n\n" \
    "**Instructions** (follow step by step):\n" \
    "1. **PREFERENCE**: In 1-2 sentences, summarize the user's likely preferences as revealed by the items with **high CF scores** in their history. (CF scores reflect similarity to other users with similar tastes.)\n\n" \
    "2. **ANALYSIS**: In 1-2 sentences, explain the discrepancy between the actor's recommendations and the CF scores. For instance:\n" \
    "   - The actor included items with low CF scores while ignoring high-scoring ones.\n" \
    "   - The actor over-relied on non-CF signals (e.g., text similarity) and neglected the collaborative signal.\n\n" \
    "3. **SUGGESTIONS**: In 1-2 sentences, advise how the actor should adjust future recommendations to better leverage CF scores (e.g., \"give more weight to items with CF scores > 0.8\", \"avoid items with CF scores below 0.5\").\n\n" \
    "**Important**:\n" \
    "- Do not output specific item titles.\n" \
    "- Focus on the scores and their implications.\n\n" \
    "Here are some examples (CF scores are placeholders):\n\n" \
    "## Previous instruction and items' collaborative filtering scores:\n" + item_CF_scores_example1 + "\n\n" \
    "## Standard answer:\n" + response__example3 + "\n\n" \
    "## Actor's recommendations:\n" + last_act_his_str_example3 + "\n\n" \
    "## Your response:\n" + fs_example3 + "\n\n" \
    "## Previous instruction and items' collaborative filtering scores:\n" + item_CF_scores_example2 + "\n\n" \
    "## Standard answer:\n" + response__example4 + "\n\n" \
    "## Actor's recommendations:\n" + last_act_his_str_example4 + "\n\n" \
    "## Your response:\n" + fs_example4 + "\n\n" \
    "Now, analyze the following case:\n\n" \
    "## Previous instruction and items' collaborative filtering scores:\n{instruction}\n\n" \
    "## Standard answer:\n{response}\n\n" \
    "## Actor's recommendations:\n{last_act_his_str}\n\n" \
    "## Your response:"

CF_fs_prompt_wo_example = "You are an expert at providing reflections based on collaborative filtering scores to enhance a sequential recommender.\n\n" \
    "**Task**: Given the user's history, the actor's recommendations, and the CF scores, provide a structured reflection.\n\n" \
    "**Steps**:\n" \
    "1. **PREFERENCE**: 1-2 sentences summarizing user preference from high-CF items in their history.\n" \
    "2. **ANALYSIS**: 1-2 sentences explaining the mismatch between actor's recommendations and CF scores.\n" \
    "3. **SUGGESTIONS**: 1-2 sentences advising how to better leverage CF scores.\n\n" \
    "**Constraints**:\n" \
    "- Do not output item titles.\n" \
    "- Focus on scores.\n\n" \
    "Now analyze the following:\n\n" \
    "## Previous instruction and items' collaborative filtering scores (item titles & CF scores):\n{instruction}\n\n" \
    "## Standard answer:\n{response}\n\n" \
    "## Actor's recommendations:\n{last_act_his_str}\n\n" \
    "## Your response:"


rec_prompt = \
    "The user has interacted in chronological order with items: {inters}. " \
    "Here is the candidate set for the next recommendation: {candidate_set}. " \
    "Please rank these candidates by the likelihood that the user will purchase next, based on the user's recent purchases (especially the last few items)." \
    "Output the top 10 recommendations as a numbered list, from most likely to least likely. " \
    "You must only choose from the given candidate set. Don't output any other content."

rec_w_reflect_prompt = \
    "The user has interacted in chronological order with items: {inters}. \n" \
    "Here is the candidate set for the next recommendation: {candidate_set}. \n" \
    "Hint: {reflect}.\n" \
    "Please rank these candidates by the likelihood that the user will purchase next, considering the hint and the user's recent purchases. " \
    "Output the top 10 recommendations as a numbered list, from most likely to least likely. " \
    "You must only choose from the given candidate set. Don't output any other content."

rec_prompt_v2 = "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records (pay special attention to the most recent purchases).\n" \
    "CRITICAL OUTPUT FORMAT:\n" \
    "You MUST output exactly 10 items. \n" \
    "You MUST format each line strictly as '1. Item Name'. \n" \
    "DO NOT output single or double quotes around the item names. \n" \
    "DO NOT output any explanations, reasoning, or extra words after the item name. \n" \
    "DO NOT output any other content."


rec_w_reflect_prompt_v2 = "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "And after the last previous recommendation, you gave the following reflections:\n{reflect}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records (focusing on recent trends) and the reflections.\n" \
    "CRITICAL OUTPUT FORMAT:\n" \
    "You MUST output exactly 10 items. \n" \
    "You MUST format each line strictly as '1. Item Name'. \n" \
    "DO NOT output single or double quotes around the item names. \n" \
    "DO NOT output any explanations, reasoning, or extra words after the item name. \n" \
    "DO NOT output any other content."

rec_w_reflect_prompt_v3 = "I've purchased the following products in the past in order:\n{inters}\n\n" \
    "And after the last previous recommendation, you gave the following reflections:\n{reflect}\n" \
    "Now there are [[candidate_set_size]] candidate products that I can consider to purchase next:\n{candidate_set}\n" \
    "Please rank these [[candidate_set_size]] products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records (focusing on recent trends) and the reflections.\n" \
    "CRITICAL OUTPUT FORMAT:\n" \
    "You MUST output exactly 10 items. \n" \
    "You MUST format each line strictly as '1. Item Name'. \n" \
    "DO NOT output single or double quotes around the item names. \n" \
    "DO NOT output any explanations, reasoning, or extra words after the item name. \n" \
    "DO NOT output any other content."
    
fs_prompts = {'EP': EP_fs_prompt, 'IP' : IP_fs_prompt, 'CF' : CF_fs_prompt}

rec_prompts = {1: rec_prompt, 2: rec_prompt_v2}
rec_w_reflect_prompts = {1: rec_w_reflect_prompt, 2: rec_w_reflect_prompt_v2, 3: rec_w_reflect_prompt_v3}