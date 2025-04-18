import copy
import logging
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from .utils import Beam, build_conv, generate_k_steps
from sal.utils.score import aggregate_scores

logger = logging.getLogger()

def _greedy_backtrack_search(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> list[Beam]:
    all_beams: list[Beam] = []
    finalized_beams: list[Beam] = []

    for i, prompt in enumerate(batch_of_prompts):
        root_beam = Beam(
            prompt=prompt,
            index=i,
            current_text="",
            next_texts=None,
            lookahead_texts=None,
            pruned=False,
            completed=False,
            stop_reasons=None,
            history=[],
            best_scores=[],
            all_scores=[],
            previous_text=None,
            completion_tokens=0,
            #step=0,
        )
        root_beam.step=0
        all_beams.append(root_beam)

    for step in tqdm(range(config.num_iterations), desc="Greedy Backtracking Search"):
        leaf_beams = [b for b in all_beams if not b.completed and not b.pruned]
        if config.max_backtrack_depth is not None:
            leaf_beams = [b for b in leaf_beams if step - b.step <= config.max_backtrack_depth]
        if not leaf_beams:
            break

        prompts = [b.prompt for b in leaf_beams]
        completions = [[b.current_text] for b in leaf_beams]
        scores = prm.score(prompts, completions)
        for b, s in zip(leaf_beams, scores):
            b.all_scores = s[0]

        best_beam = max(leaf_beams, key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy))

        conv = build_conv(best_beam.prompt, best_beam.current_text, config.system_prompt)
        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        templated_conv = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            continue_final_message=False,
            tokenize=False,
        )

        # simulate n outputs using multiple calls with n=1
        for _ in range(config.n):
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                stop=["\n\n"],
                include_stop_str_in_output=True,
                n=1,
            )
            gen_result = generate_k_steps([templated_conv], config.lookahead, llm, sampling_params, 1)[0]

            for text in gen_result.next_texts:
                new_beam = copy.deepcopy(best_beam)
                new_beam.current_text += text
                new_beam.history.append(text)
                new_beam.step = step + 1
                new_beam.completion_tokens += gen_result.completion_tokens
                new_beam.stop_reasons = gen_result.stop_reasons
                new_beam.completed = (text == "") or ("EOS" in new_beam.stop_reasons)
                all_beams.append(new_beam)
                if new_beam.completed:
                    finalized_beams.append(new_beam)

        if config.early_stop_when_x_finished is not None:
            if len(finalized_beams) >= config.early_stop_when_x_finished:
                break

    if config.sort_completed:
        finalized_beams = sorted(
            finalized_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        finalized_beams = finalized_beams[: config.n]

    if len(finalized_beams) != config.n:
        repeats = (config.n // len(finalized_beams)) + 1
        extended = [copy.deepcopy(b) for b in (finalized_beams * repeats)[: config.n]]
        finalized_beams = extended

    return finalized_beams

def greedy_backtrack_search(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _greedy_backtrack_search(problems, config, llm, prm)
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [aggregate_scores(b.all_scores, config.agg_strategy) for b in beams]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
