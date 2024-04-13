# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from pathlib import Path
import torch as t
from torch import Tensor
import numpy as np
import einops
from tqdm.notebook import tqdm
import plotly.express as px
from plotly_utils import imshow, line, scatter, bar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import webbrowser
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from IPython.display import display, HTML
from rich.table import Table, Column
from rich import print as rprint
import circuitsvis as cv
from pathlib import Path
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP
torch = t

t.set_grad_enabled(False)

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
device = t.device("mps") if torch.backends.mps.is_built() else "cpu"

MAIN = __name__ == "__main__"

# %%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
    

# %%
vocab = model.to_str_tokens(t.arange(model.cfg.d_vocab))
print(len(vocab))

# capital tokens should begin with a space and then a capital letter
capital_tokens = [token for token in vocab if len(token) > 1 and token[0] == ' ' and token[1].isupper()]
capital_indices = [model.tokenizer.encode(token)[0] for token in capital_tokens]
capital_indices = list(set(capital_indices))
capital_indices = torch.tensor(capital_indices).unsqueeze(0)
capital_indices = capital_indices.to(device)

lowercase_tokens = [token.lower() for token in capital_tokens]
lowercase_tokens = list(set(lowercase_tokens))
lowercase_indices = [model.tokenizer.encode(token)[0] for token in lowercase_tokens]
lowercase_indices = list(set(lowercase_indices))
lowercase_indices = torch.tensor(lowercase_indices).unsqueeze(0)
lowercase_indices = lowercase_indices.to(device)

print(f"Number of capital tokens: {len(capital_tokens)}")
print(f"Number of lowercase tokens: {len(lowercase_tokens)}")


# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter

# %%
# Collect word pairs
lowercase_words = [
    "cat", "dog", "hat", "mom", "dad", "sun", "car", "bus", "red", "big",
    "fox", "cow", "hen", "pig", "rat", "ant", "bee", "owl", "elk", "toy",
    "mug", "pen", "cup", "jam", "egg", "ice", "oil", "pie", "tea", "rug",
    "bat", "bed", "can", "fan", "ham", "jar", "kite", "lamp", "mask", "nest",
    "oven", "pump", "quilt", "rope", "sock", "tent", "vase", "wig", "yoyo", "zoo", 
    "he", "she", "it", "they", "we", "you", "me", "us", "him", "her",
    "his", "hers", "its", "theirs", "ours", "mine", "yours", "this", "that",
    "these", "those", "who", "whom", "whose", "which", "what", "where", "when",
    "why", "how", "if", "then", "else", "but", "and", "or", "not", "for", "to",
    "in", "on", "at", "by", "with", "about", "against", "between", "through", "into",
    "during", "before", "after", "above", "below", "from", "up", "down", "out", "off",
    "over", "under", "around", "throughout", "upon", "of", "as", "like", "about", "before",
    "after", "since", "while", "because", "though", "although", "even", "if", "unless", "until",
]

lowercase_words = list(set(lowercase_words))

caps_words = [word.capitalize() for word in lowercase_words]

lowercase_words = [' ' + word for word in lowercase_words]
caps_words = [' ' + word for word in caps_words]

new_lowercase_words = []
new_caps_words = []

for i in range(len(lowercase_words)):
    if model.to_tokens(lowercase_words[i], prepend_bos=False).shape[1] == 1 and model.to_tokens(caps_words[i], prepend_bos=False).shape[1] == 1:
        new_lowercase_words.append(lowercase_words[i])
        new_caps_words.append(caps_words[i])

lowercase_words = new_lowercase_words
caps_words = new_caps_words

lowercase_tokens = model.to_tokens(lowercase_words, prepend_bos=False)[:, 0]
capitalized_tokens = model.to_tokens(caps_words, prepend_bos=False)[:, 0]

lowercase_tokens = torch.Tensor(lowercase_tokens).to(device)
capitalized_tokens = torch.Tensor(capitalized_tokens).to(device)

lowercase_unembeddings = model.tokens_to_residual_directions(lowercase_tokens)
# normalize the unembeddings
lowercase_unembeddings = lowercase_unembeddings / torch.norm(lowercase_unembeddings, dim=-1, keepdim=True)

capitalized_unembeddings = model.tokens_to_residual_directions(capitalized_tokens)
# normalize the unembeddings
capitalized_unembeddings = capitalized_unembeddings / torch.norm(capitalized_unembeddings, dim=-1, keepdim=True)


all_unembeddings = torch.cat((lowercase_unembeddings, capitalized_unembeddings), dim=0)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# concatenate lowercase and capitalized unembeddings
all_unembeddings = torch.cat((lowercase_unembeddings, capitalized_unembeddings), dim=0)

# perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_unembeddings.cpu().numpy())

# split PCA results into lowercase and capitalized
lowercase_pca = pca_result[:len(lowercase_unembeddings)]
capitalized_pca = pca_result[len(lowercase_unembeddings):]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(lowercase_pca[:, 0], lowercase_pca[:, 1], color='blue', label='Lowercase', alpha=0.7)
plt.scatter(capitalized_pca[:, 0], capitalized_pca[:, 1], color='red', label='Capitalized', alpha=0.7)

# Add connecting lines between lowercase and capitalized word vector pairs
for i in range(len(lowercase_pca)):
    plt.plot([lowercase_pca[i, 0], capitalized_pca[i, 0]], [lowercase_pca[i, 1], capitalized_pca[i, 1]], 
             color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA2 Plot of Lowercase and Capitalized Word Vectors')
plt.legend()
plt.grid(True)
plt.show()

# %%

diff_vectors = capitalized_unembeddings - lowercase_unembeddings
diff_vectors = diff_vectors / torch.norm(diff_vectors, dim=-1, keepdim=True)
diff_vectors = diff_vectors.cpu()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

caps_and_lower_vectors = torch.cat([capitalized_unembeddings, lowercase_unembeddings], dim=0).cpu()

pca = PCA(n_components=2)
pca_results = pca.fit_transform(diff_vectors)

plt.figure(figsize=(8, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Difference Vectors')
plt.show()

avg_diff_vector = torch.mean(diff_vectors, dim=0)

# normalize diff vectors
diff_vectors = diff_vectors

# normalize avg diff vector
avg_diff_vector = avg_diff_vector

# calculate cosine similarity
cosine_similarities = torch.nn.functional.cosine_similarity(diff_vectors, avg_diff_vector.unsqueeze(0), dim=1)

print("Average Cosine Similarity:", cosine_similarities.mean().item())

avg_diff_vector = avg_diff_vector.to(device)

# %%

full_stop_prompts = ['The dog ran fast.', 'The cat walked slow.']
comma_prompts = ['The dog ran fast', 'The cat walked, slow.']

if MAIN:
    example_prompt = "The cat walked fast."
    example_answer = " The"
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# %%

prompt_format = [
        'A dog ran toward me.',
        'He slept well last night.',
        'Many people like Greek salad.',
        'The best color is purple.',
        'I sleep in a bed.',
        'The cat is so cute.',
        'Roger can play the piano.',
        'The sun is out to for me.',
        'Ten years is long.',
        'We should visit the beach.',
        'Screens emit blue light.',
        'This town is for old men.',
        'Stacy has a mom.',
        'Sports tend to be tedious.',
        'She might be a doctor.',
        'The dog ran fast.',
        'The cat walked slow.',
        'The news is always bad.',
        'Time to go to bed.',
        'Music is good for the soul.',
        'The sky is blue.',
        'The sun is shining.',
        'The moon is bright.',
]

prompts = []
for prompt in prompt_format:
    # check if prompt is 6 tokens long
    if len(model.to_tokens(prompt, prepend_bos=False)[0]) != 6:
        print("Error: Prompt '{}' is {} tokens long.".format(prompt, len(model.to_tokens(prompt, prepend_bos=False)[0])))
        continue
    prompts.append(prompt)

# %%
num_tokens_in_prompt = 7

tokens = model.to_tokens(prompts, prepend_bos=True)
tokens = tokens.to(device)
clean_tokens = tokens
clean_prompts = prompts

original_logits, cache = model.run_with_cache(tokens)

list_prompts = [prompts.split(' ') for prompts in prompts]


# create the three corrupted datasets
# period -> semicolon
corrupted_prompts_1 = []
for prompt in list_prompts:
    new_prompt = []
    for word in prompt[:-1]:
        new_prompt.append(word)
    new_prompt.append(prompt[-1].replace('.', ';'))
    corrupted_prompts_1.append(' '.join(new_prompt))    

corrupted_tokens_1 = model.to_tokens(corrupted_prompts_1, prepend_bos=True)[:, :num_tokens_in_prompt]
corrupted_tokens_1 = corrupted_tokens_1.to(device)

# lowercase first letter
corrupted_prompts_2 = []
for prompt in list_prompts:
    new_prompt = []
    new_prompt.append(prompt[0].lower())
    for word in prompt[1:]:
        new_prompt.append(word)
    corrupted_prompts_2.append(' '.join(new_prompt))

corrupted_tokens_2 = model.to_tokens(corrupted_prompts_2, prepend_bos=True)[:, :num_tokens_in_prompt]
corrupted_tokens_2 = corrupted_tokens_2.to(device)

# make first letter lowercase and period -> comma
corrupted_prompts_3 = []
for prompt in list_prompts:
    new_prompt = []
    new_prompt.append(prompt[0].lower())
    for word in prompt[1:]:
        new_prompt.append(word)
    new_prompt[-1] = new_prompt[-1].replace('.', ';')
    corrupted_prompts_3.append(' '.join(new_prompt))

corrupted_tokens_3 = model.to_tokens(corrupted_prompts_3, prepend_bos=True)[: , :num_tokens_in_prompt]
corrupted_tokens_3 = corrupted_tokens_3.to(device)

# %%
print(corrupted_prompts_1)
print(corrupted_prompts_2)
print(corrupted_prompts_3)
# %%

def decompose_residual_stream(model, tokens, avg_diff_vector):
    model.reset_hooks()
    _, cache = model.run_with_cache(tokens)
    
    residual_stack, labels = cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)
    residual_stack = residual_stack.to(device)
    batch_size = residual_stack.size(-2)
    print(batch_size)
    
    # project residual vectors onto the avg_diff_vector
    print(residual_stack.shape)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    projections =  einops.einsum(
        scaled_residual_stack, avg_diff_vector,
        "... batch d_model, d_model -> ..."
    ) / batch_size
    residual_stack.cpu()
    return projections, labels

layer_contributions, labels = decompose_residual_stream(model, clean_tokens, avg_diff_vector)

# %%

line(
    layer_contributions, 
    hovermode="x unified",
    title="Output Dot Product with Capitalization Direction",
    labels={"x": "Layer", "y": "Dot Product"},
    xaxis_tickvals=labels,
    width=800
)

# %%

def decompose_attention_heads(model, tokens, avg_diff_vector):
    model.reset_hooks()
    _, cache = model.run_with_cache(tokens)
    
    attn_head_outputs, labels = cache.stack_head_results(layer=-1, pos_slice=-1, return_labels=True)
    attn_head_outputs = attn_head_outputs.to(device)

    batch_size = attn_head_outputs.size(-3)
    
    # project attention head outputs onto the avg_diff_vector
    projections = torch.einsum(
        "heads batch d_model, d_model -> heads",
        attn_head_outputs, avg_diff_vector
    ) / batch_size
    
    return projections, labels

head_contributions, labels = decompose_attention_heads(model, corrupted_tokens_2, avg_diff_vector)
head_contributions = head_contributions.reshape(12, -1)

labels = np.array(labels).reshape(12, -1)

df = pd.DataFrame(head_contributions.cpu().numpy(), columns=labels[0])

plt.figure(figsize=(12, 10))
sns.heatmap(df, cmap='viridis', center=0, annot=True, fmt=".2f")
plt.xlabel('Attention Head')
plt.ylabel('Layer')
plt.title('Attention Head Contributions in the Capitalization Direction for Corrupted Dataset v3')
plt.tight_layout()
plt.show()

# %%

W_U = model.unembed.W_U

# Generate the most common 100 words in English

# Get the stopwords in English
stop_words = set(stopwords.words('english'))

# Count the frequency of each word in the English language
word_freq = Counter(stop_words)

# Get the most common 100 words
most_common_words = word_freq.most_common(100)

lowercase_words = [w.lower() for w, _ in most_common_words]

def capitalize_first_letter(word):
    return word[0].upper() + word[1:]

capital_words = [capitalize_first_letter(w) for w in lowercase_words]

word_pairs = [(' ' + w, ' ' + w_prime) for w, w_prime in zip(capital_words, lowercase_words)]

diff_vectors = []
capital_word_vectors = []
lowercase_word_vectors = []
for w, w_prime in word_pairs:
    w_index = model.to_tokens(w, prepend_bos=False)[0][0]
    w_prime_index = model.to_tokens(w_prime, prepend_bos=False)[0][0]
    diff = W_U.T[w_index] - W_U.T[w_prime_index]
    diff_vectors.append(diff)
    capital_word_vectors.append(W_U.T[w_index])
    lowercase_word_vectors.append(W_U.T[w_prime_index])
    
# Convert the vectors to PyTorch tensors
diff_vectors = torch.stack(diff_vectors)
capital_word_vectors = torch.stack(capital_word_vectors)
lowercase_word_vectors = torch.stack(lowercase_word_vectors)

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=2)
diff_vectors_pca = pca.fit_transform(utils.to_numpy(diff_vectors))
capital_word_vectors_pca = pca.transform(utils.to_numpy(capital_word_vectors))
lowercase_word_vectors_pca = pca.transform(utils.to_numpy(lowercase_word_vectors))

# Plot the differences
plt.figure(figsize=(8, 6))
plt.quiver(
    capital_word_vectors_pca[:, 0],
    capital_word_vectors_pca[:, 1],
    diff_vectors_pca[:, 0],
    diff_vectors_pca[:, 1],
    angles='xy',
    scale_units='xy',
    scale=1,
    color='blue',
)
plt.scatter(
    capital_word_vectors_pca[:, 0],
    capital_word_vectors_pca[:, 1],
    color='red',
    label='Capital Words',
)
plt.scatter(
    lowercase_word_vectors_pca[:, 0],
    lowercase_word_vectors_pca[:, 1],
    color='green',
    label='Lowercase Words',
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Differences between Capital and Lowercase Word Vectors')
plt.show()

# %%

def test_prompt_from_resid_post(resid_post, W_U, k=10, verbose=True):
    '''
    gets the top k logits from the residual post
    '''
    logits = resid_post @ W_U
    top_logits = torch.topk(logits, k)
    top_tokens = model.to_str_tokens(top_logits.indices)
    top_probs = torch.nn.functional.softmax(top_logits.values, dim=-1).tolist()[0]
    top_logits = top_logits.values.tolist()[0]

    if verbose:
        table = Table("Token", "Logit", "Probability")
        for token, logit, prob in zip(top_tokens, top_logits, top_probs):
            table.add_row(token, f"{logit:.4f}", f"{prob:.4f}")
        rprint(table)
    return top_tokens, top_logits, top_probs

# %%

caps_prompts = ['First I came. Second I saw.']
tokens = model.to_tokens(caps_prompts, prepend_bos=True)

caps_logits, caps_cache = model.run_with_cache(tokens)

# Get the final residual stream
resid_post = caps_cache["resid_post", -1]
resid_post = caps_cache.apply_ln_to_stack(resid_post, layer=-1, pos_slice=-1)
resid_post_final_token = resid_post[:, -1, :]

test_prompt_from_resid_post(resid_post_final_token - avg_diff_vector * 50, model.W_U)


# %%

alpha_lst = []
lower_prob_lst = []
upper_prob_lst = []
for alpha in np.arange(0, 50, 1):
    resid_post_final_token_new = resid_post_final_token - avg_diff_vector * alpha
    print(f"Alpha: {alpha}")
    tokens, logits, probs = test_prompt_from_resid_post(resid_post_final_token_new, model.W_U, k=10000)

    if " third" not in tokens:
        lower_prob = 0
    else:
        lower_idx = tokens.index(" third")
        lower_prob = probs[lower_idx]
    lower_prob_lst.append(lower_prob)

    if " Third" not in tokens:
        upper_prob = 0
    else:
        upper_idx = tokens.index(" Third")
        upper_prob = probs[upper_idx]
    upper_prob_lst.append(upper_prob)

    print(f"Alpha: {alpha}")
    print(f"Lower prob: {lower_prob}")
    print(f"Upper prob: {upper_prob}")
    print()

    alpha_lst.append(alpha)
    
plt.plot(alpha_lst, lower_prob_lst, label='" third"')
plt.plot(alpha_lst, upper_prob_lst, label='" Third"')
plt.xlabel("Alpha")
plt.ylabel("Probability")
plt.yscale('log')
plt.title('Probability of " third" and " Third" in the Top 100 Tokens')
plt.legend()
plt.show()

test_prompt_from_resid_post(resid_post_final_token, model.W_U)
test_prompt_from_resid_post(resid_post_final_token_new, model.W_U)

logits = resid_post_final_token @ model.W_U
logits_new = resid_post_final_token_new @ model.W_U

# %%
def logits_to_ave_prob_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens=None, # artifact from old function, to change
    capital_indices: List[int] = capital_indices,
    lowercase_indices: List[int] = lowercase_indices,
    per_prompt: bool = False,
) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
    '''
    Returns the probability of the model predicting a capitalized token.
    If per_prompt=True, return the array of probabilities rather than the average.
    '''
    # only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]  # [batch d_vocab]
    final_probs = torch.nn.functional.softmax(final_logits, dim=-1)

    # cumulative probability of capital tokens
    correct_probs = final_probs[:, capital_indices]
    correct_probs_sum = correct_probs.sum(dim=-1)

    # cumulative probability of lowercase tokens
    incorrect_probs = final_probs[:, lowercase_indices]
    incorrect_probs_sum = incorrect_probs.sum(dim=-1)

    # take prob difference
    answer_probs_diff = correct_probs_sum - incorrect_probs_sum

    return answer_probs_diff.mean() if not per_prompt else answer_probs_diff

# %%

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = None,
    per_prompt: bool = False
) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :] # [batch d_vocab]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens) # [batch 2]
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


# %%

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = t.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()


# %% 
from transformer_lens import patching

# %%

print(
    "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
    "Corrupted string v1 0:", model.to_string(corrupted_tokens_1[0]), "\n"
    "Corrupted string v2 0:", model.to_string(corrupted_tokens_2[0]), "\n"
    "Corrupted string v3 0:", model.to_string(corrupted_tokens_3[0])
)

clean_logits, clean_cache = model.run_with_cache(clean_tokens, return_type="logits")
corrupted_logits_1, corrupted_cache_1 = model.run_with_cache(corrupted_tokens_1, return_type="logits")
corrupted_logits_2, corrupted_cache_2 = model.run_with_cache(corrupted_tokens_2, return_type="logits")
corrupted_logits_3, corrupted_cache_3 = model.run_with_cache(corrupted_tokens_3, return_type="logits")


# %%

clean_prob_diff = logits_to_ave_prob_diff(clean_logits)
print(f"Clean probability diff: {clean_prob_diff:.4f}")

corrupted_prob_diff_1 = logits_to_ave_prob_diff(corrupted_logits_1)
print(f"Corrupted probability diff 1 {corrupted_prob_diff_1:.4f}")

corrupted_prob_diff_2 = logits_to_ave_prob_diff(corrupted_logits_2)
print(f"Corrupted probability 2: {corrupted_prob_diff_2:.4f}")

corrupted_prob_diff_3 = logits_to_ave_prob_diff(corrupted_logits_3)
print(f"Corrupted probability 3: {corrupted_prob_diff_3:.4f}")

# %%

def patch_residual_component(
    corrupted_residual_component: Float[Tensor, "batch pos d_model"],
    hook: HookPoint, 
    pos: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos d_model"]:
    '''
    Patches a given sequence position in the residual stream, using the value
    from the clean cache.
    '''
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component


def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    model.reset_hooks()
    seq_len = corrupted_tokens.size(1)
    results = t.zeros(model.cfg.n_layers, seq_len, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for position in range(seq_len):
            hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens, 
                fwd_hooks = [(utils.get_act_name("resid_pre", layer), hook_fn)], 
            )
            results[layer, position] = patching_metric(patched_logits)

    return results


# %%

act_patch_resid_pre_own_1 = get_act_patch_resid_pre(model, corrupted_tokens_1, clean_cache, logits_to_ave_prob_diff)
act_patch_resid_pre_own_2 = get_act_patch_resid_pre(model, corrupted_tokens_2, clean_cache, logits_to_ave_prob_diff)
act_patch_resid_pre_own_3 = get_act_patch_resid_pre(model, corrupted_tokens_3, clean_cache, logits_to_ave_prob_diff)

# %%

labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]
print(act_patch_resid_pre_own_1.shape)
print(labels)

# %%

imshow(
    act_patch_resid_pre_own_1, 
    x=labels, 
    title="Probability Difference From Patched Residual Stream (v1)", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600 # If you remove this argument, the plot will usually fill the available space
)

imshow(
    act_patch_resid_pre_own_2, 
    x=labels, 
    title="Probability Difference From Patched Residual Stream (v2)", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600
)

imshow(
    act_patch_resid_pre_own_3, 
    x=labels, 
    title="Probability Difference From Patched Residual Stream (v3)", 
    labels={"x":"Sequence Position", "y":"Layer"},
    width=600
)


# %%

act_patch_block_every_1 = patching.get_act_patch_block_every(model, corrupted_tokens_1, clean_cache, logits_to_ave_prob_diff)
act_patch_block_every_2 = patching.get_act_patch_block_every(model, corrupted_tokens_2, clean_cache, logits_to_ave_prob_diff)
act_patch_block_every_3 = patching.get_act_patch_block_every(model, corrupted_tokens_3, clean_cache, logits_to_ave_prob_diff)

# %%

imshow(
    act_patch_block_every_1,
    x=labels, 
    facet_col=0, # This argument tells plotly which dimension to split into separate plots
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"], # Subtitles of separate plots
    title="Probability Difference From Patched Attn Head Output (v1)", 
    labels={"x": "Sequence Position", "y": "Layer"},
    width=1000,
)

imshow(
    act_patch_block_every_2,
    x=labels, 
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Probability Difference From Patched Attn Head Output (v2)", 
    labels={"x": "Sequence Position", "y": "Layer"},
    width=1000
)

imshow(
    act_patch_block_every_3,
    x=labels, 
    facet_col=0,
    facet_labels=["Residual Stream", "Attn Output", "MLP Output"],
    title="Probability Difference From Patched Attn Head Output (v3)", 
    labels={"x": "Sequence Position", "y": "Layer"},
    width=1000
)


# %%

def get_act_patch_block_every(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch pos d_vocab"]], float]
) -> Float[Tensor, "layer pos"]:
    '''
    Returns an array of results of patching each position at each layer in the residual
    stream, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    model.reset_hooks()
    results = t.zeros(3, model.cfg.n_layers, tokens.size(1), device=device, dtype=t.float32)

    for component_idx, component in enumerate(["resid_pre", "attn_out", "mlp_out"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            for position in range(corrupted_tokens.shape[1]):
                hook_fn = partial(patch_residual_component, pos=position, clean_cache=clean_cache)
                patched_logits = model.run_with_hooks(
                    corrupted_tokens, 
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)], 
                )
                results[component_idx, layer, position] = patching_metric(patched_logits)

    return results

# %%

act_patch_attn_head_out_all_pos_1 = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    corrupted_tokens_1,
    clean_cache, 
    logits_to_ave_prob_diff
)

act_patch_attn_head_out_all_pos_2 = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    corrupted_tokens_2,
    clean_cache, 
    logits_to_ave_prob_diff
)

act_patch_attn_head_out_all_pos_3 = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    corrupted_tokens_3,
    clean_cache, 
    logits_to_ave_prob_diff
)

# %%

imshow(
    act_patch_attn_head_out_all_pos_1, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos) (v1)",
    width=600
)

imshow(
    act_patch_attn_head_out_all_pos_2, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos) (v2)",
    width=600
)

imshow(
    act_patch_attn_head_out_all_pos_3, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos) (v3)",
    width=600
)


# %%

def patch_head_vector(
    corrupted_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the clean cache.
    '''
    corrupted_head_vector[:, :, head_index] = clean_cache[hook.name][:, :, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_out_all_pos(
    model: HookedTransformer, 
    corrupted_tokens: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of results of patching at all positions for each head in each
    layer, using the value from the clean cache.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            hook_fn = partial(patch_head_vector, head_index=head, clean_cache=clean_cache)
            patched_logits = model.run_with_hooks(
                corrupted_tokens, 
                fwd_hooks = [(utils.get_act_name("z", layer), hook_fn)], 
                return_type="logits"
            )
            results[layer, head] = patching_metric(patched_logits)

    return results

# %%

print(model.to_string(corrupted_tokens_3[0]))
print(model.to_string(corrupted_tokens_2[0]))

act_patch_attn_head_out_all_pos_own_1 = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens_1, clean_cache, logits_to_ave_prob_diff)
act_patch_attn_head_out_all_pos_own_2 = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens_2, clean_cache, logits_to_ave_prob_diff)
act_patch_attn_head_out_all_pos_own_3 = get_act_patch_attn_head_out_all_pos(model, corrupted_tokens_3, clean_cache, logits_to_ave_prob_diff)


# %%

imshow(
    act_patch_attn_head_out_all_pos_own_1,
    title="Probability Difference From Patched Attn Head Output (v1)",
    labels={"x":"Head", "y":"Layer"},
    width=600
)

imshow(
    act_patch_attn_head_out_all_pos_own_2,
    title="Probability Difference From Patched Attn Head Output (v2)",
    labels={"x":"Head", "y":"Layer"},
    width=600
)

imshow(
    act_patch_attn_head_out_all_pos_own_3,
    title="Probability Difference From Patched Attn Head Output (v3)",
    labels={"x":"Head", "y":"Layer"},
    width=600
)

# %%
act_patch_attn_head_all_pos_every_1 = patching.get_act_patch_attn_head_all_pos_every(
    model, 
    corrupted_tokens_1,
    clean_cache, 
    logits_to_ave_prob_diff
)

act_patch_attn_head_all_pos_every_2 = patching.get_act_patch_attn_head_all_pos_every(
    model, 
    corrupted_tokens_2,
    clean_cache, 
    logits_to_ave_prob_diff
)

act_patch_attn_head_all_pos_every_3 = patching.get_act_patch_attn_head_all_pos_every(
    model, 
    corrupted_tokens_3,
    clean_cache, 
    logits_to_ave_prob_diff
)



# %%

imshow(
    act_patch_attn_head_all_pos_every_1, 
    facet_col=0, 
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos) (v1)",
    labels={"x": "Head", "y": "Layer"},
)

imshow(
    act_patch_attn_head_all_pos_every_2, 
    facet_col=0, 
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos) (v2)",
    labels={"x": "Head", "y": "Layer"},
)

imshow(
    act_patch_attn_head_all_pos_every_3, 
    facet_col=0, 
    facet_labels=["Output", "Query", "Key", "Value", "Pattern"],
    title="Activation Patching Per Head (All Pos) (v3)",
    labels={"x": "Head", "y": "Layer"},
)

# %%

def patch_attn_patterns(
    corrupted_head_vector: Float[Tensor, "batch head_index pos_q pos_k"],
    hook: HookPoint, 
    head_index: int, 
    clean_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the attn patterns of a given head at every sequence position, using 
    the value from the clean cache.
    '''
    corrupted_head_vector[:, head_index] = clean_cache[hook.name][:, head_index]
    return corrupted_head_vector


def get_act_patch_attn_head_all_pos_every(
    model: HookedTransformer,
    corrupted_tokens: Float[Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable
) -> Float[Tensor, "layer head"]:
    '''
    Returns an array of results of patching at all positions for each head in each
    layer (using the value from the clean cache) for output, queries, keys, values
    and attn pattern in turn.

    The results are calculated using the patching_metric function, which should be
    called on the model's logit output.
    '''
    results = t.zeros(5, model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)
    # Loop over each component in turn
    for component_idx, component in enumerate(["z", "q", "k", "v", "pattern"]):
        for layer in tqdm(range(model.cfg.n_layers)):
            for head in range(model.cfg.n_heads):
                # Get different hook function if we're doing attention probs
                hook_fn_general = patch_attn_patterns if component == "pattern" else patch_head_vector
                hook_fn = partial(hook_fn_general, head_index=head, clean_cache=clean_cache)
                # Get patched logits
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks = [(utils.get_act_name(component, layer), hook_fn)],
                    return_type="logits"
                )
                results[component_idx, layer, head] = patching_metric(patched_logits)

    return results

# %%

reference_text = corrupted_prompts_2[0]
# reference_text = clean_prompts[3]
import circuitsvis as cv
from IPython.display import display

_, cache = model.run_with_cache(corrupted_tokens_2)

print(cache)

html = cv.attention.attention_patterns(
    tokens=model.to_str_tokens(reference_text), 
    attention=cache["pattern", 8][0],
)
display(html)

# %%

def patch_or_freeze_head_vectors(
    orig_head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint, 
    new_cache: ActivationCache,
    orig_cache: ActivationCache,
    head_to_patch: Tuple[int, int], 
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    This helps implement step 2 of path patching. We freeze all head outputs (i.e. set them
    to their values in orig_cache), except for head_to_patch (if it's in this layer) which
    we patch with the value from new_cache.

    head_to_patch: tuple of (layer, head)
        we can use hook.layer() to check if the head to patch is in this layer
    '''
    # Setting using ..., otherwise changing orig_head_vector will edit cache value too
    orig_head_vector[...] = orig_cache[hook.name][...]
    if head_to_patch[0] == hook.layer():
        orig_head_vector[:, :, head_to_patch[1]] = new_cache[hook.name][:, :, head_to_patch[1]]
    return orig_head_vector


# new = corrupt
def get_path_patch_head_to_final_resid_post(
    model: HookedTransformer,
    patching_metric: Callable,
    new_toks = corrupted_tokens_3,
    orig_toks = clean_tokens,
    new_cache: Optional[ActivationCache] = corrupted_cache_3,
    orig_cache: Optional[ActivationCache] = clean_cache,
) -> Float[Tensor, "layer head"]:
    '''
    Performs path patching (see algorithm in appendix B of IOI paper), with:

        sender head = (each head, looped through, one at a time)
        receiver node = final value of residual stream

    Returns:
        tensor of metric values for every possible sender head
    '''
    model.reset_hooks()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=t.float32)

    resid_post_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    resid_post_name_filter = lambda name: name == resid_post_hook_name


    # ========== Step 1 ==========
    # Gather activations on x_orig and x_new

    # Note the use of names_filter for the run_with_cache function. Using it means we 
    # only cache the things we need (in this case, just attn head outputs).
    z_name_filter = lambda name: name.endswith("z")
    if new_cache is None:
        _, new_cache = model.run_with_cache(
            new_toks, 
            names_filter=z_name_filter, 
            return_type=None
        )
    if orig_cache is None:
        _, orig_cache = model.run_with_cache(
            orig_toks, 
            names_filter=z_name_filter, 
            return_type=None
        )


    # Looping over every possible sender head (the receiver is always the final resid_post)
    # Note use of itertools (gives us a smoother progress bar)
    for (sender_layer, sender_head) in tqdm(list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))):

        # ========== Step 2 ==========
        # Run on x_orig, with sender head patched from x_new, every other head frozen
        
        hook_fn = partial(
            patch_or_freeze_head_vectors,
            new_cache=new_cache, 
            orig_cache=orig_cache,
            head_to_patch=(sender_layer, sender_head),
        )
        model.add_hook(z_name_filter, hook_fn)
    
        _, patched_cache = model.run_with_cache(
            orig_toks, 
            names_filter=resid_post_name_filter, 
            return_type=None
        )
        # if (sender_layer, sender_head) == (9, 9):
        #     return patched_cache
        assert set(patched_cache.keys()) == {resid_post_hook_name}

        # ========== Step 3 ==========
        # Unembed the final residual stream value, to get our patched logits

        patched_logits = model.unembed(model.ln_final(patched_cache[resid_post_hook_name]))

        # Save the results
        results[sender_layer, sender_head] = patching_metric(patched_logits)

    return results

# %%

if MAIN:
    path_patch_head_to_final_resid_post = get_path_patch_head_to_final_resid_post(model, logits_to_ave_prob_diff)
    
    imshow(
        path_patch_head_to_final_resid_post,
        title="Direct effect on logit difference",
        labels={"x":"Head", "y":"Layer", "color": "Logit diff. variation"},
        coloraxis=dict(colorbar_ticksuffix = "%"),
        width=600,
    )

# %%

def patch_head_input(
    orig_activation: Float[Tensor, "batch pos head_idx d_head"],
    hook: HookPoint,
    patched_cache: ActivationCache,
    head_list: List[Tuple[int, int]],
) -> Float[Tensor, "batch pos head_idx d_head"]:
    '''
    Function which can patch any combination of heads in layers,
    according to the heads in head_list.
    '''
    heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
    orig_activation[:, :, heads_to_patch] = patched_cache[hook.name][:, :, heads_to_patch]
    return orig_activation


if MAIN:
    def get_path_patch_head_to_heads(
        receiver_heads: List[Tuple[int, int]],
        receiver_input: str,
        model: HookedTransformer,
        patching_metric: Callable,
        new_toks = corrupted_tokens_2,
        orig_toks = clean_tokens,
        new_cache: Optional[ActivationCache] = None,
        orig_cache: Optional[ActivationCache] = None,
    ) -> Float[Tensor, "layer head"]:
        '''
        Performs path patching (see algorithm in appendix B of IOI paper), with:

            sender head = (each head, looped through, one at a time)
            receiver node = input to a later head (or set of heads)

        The receiver node is specified by receiver_heads and receiver_input.
        Example (for S-inhibition path patching the queries):
            receiver_heads = [(8, 6), (8, 10), (7, 9), (7, 3)],
            receiver_input = "v"

        Returns:
            tensor of metric values for every possible sender head
        '''
        model.reset_hooks()

        assert receiver_input in ("k", "q", "v")
        receiver_layers = set(next(zip(*receiver_heads)))
        receiver_hook_names = [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
        receiver_hook_names_filter = lambda name: name in receiver_hook_names

        results = t.zeros(max(receiver_layers), model.cfg.n_heads, device=device, dtype=t.float32)
        
        # ========== Step 1 ==========
        # Gather activations on x_orig and x_new

        # Note the use of names_filter for the run_with_cache function. Using it means we 
        # only cache the things we need (in this case, just attn head outputs).
        z_name_filter = lambda name: name.endswith("z")
        if new_cache is None:
            _, new_cache = model.run_with_cache(
                new_toks, 
                names_filter=z_name_filter, 
                return_type=None
            )
        if orig_cache is None:
            _, orig_cache = model.run_with_cache(
                orig_toks, 
                names_filter=z_name_filter, 
                return_type=None
            )

        # Note, the sender layer will always be before the final receiver layer, otherwise there will
        # be no causal effect from sender -> receiver. So we only need to loop this far.
        for (sender_layer, sender_head) in tqdm(list(itertools.product(
            range(max(receiver_layers)),
            range(model.cfg.n_heads)
        ))):

            # ========== Step 2 ==========
            # Run on x_orig, with sender head patched from x_new, every other head frozen

            hook_fn = partial(
                patch_or_freeze_head_vectors,
                new_cache=new_cache, 
                orig_cache=orig_cache,
                head_to_patch=(sender_layer, sender_head),
            )
            model.add_hook(z_name_filter, hook_fn, level=1)
            
            _, patched_cache = model.run_with_cache(
                orig_toks, 
                names_filter=receiver_hook_names_filter,  
                return_type=None
            )
            # model.reset_hooks(including_permanent=True)
            assert set(patched_cache.keys()) == set(receiver_hook_names)

            # ========== Step 3 ==========
            # Run on x_orig, patching in the receiver node(s) from the previously cached value
            
            hook_fn = partial(
                patch_head_input, 
                patched_cache=patched_cache, 
                head_list=receiver_heads,
            )
            patched_logits = model.run_with_hooks(
                orig_toks,
                fwd_hooks = [(receiver_hook_names_filter, hook_fn)], 
                return_type="logits"
            )

            # Save the results
            results[sender_layer, sender_head] = patching_metric(patched_logits)

        return results

# %%


if MAIN:
    model.reset_hooks()
    
    s_inhibition_value_path_patching_results = get_path_patch_head_to_heads(
        receiver_heads=[(11, 0), (11, 8)],
        receiver_input = "k",
        model = model,
        patching_metric = logits_to_ave_prob_diff,
    )
    
    imshow(
        s_inhibition_value_path_patching_results,
        title="Path Patching: Paths to 11.0 and 11.8, v inputs", 
        labels={"x": "Head", "y": "Layer", "color": "Probability Difference"},
        width=600,
    )

