import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()
from sal.utils.score import aggregate_scores


def _dynamic_beam_search(prompt_batch, config: Config, llm: LLM, prm: PRM) -> list[Beam]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    initial_beams: list[Beam] = []
    for prompt in prompt_batch:
        for i in range(config.n):
            initial_beams.append(
                Beam(
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
                )
            )

    finalized_beams: list[Beam] = []

    for step in tqdm(range(config.num_iterations), desc="Dynamic Beam Search Steps"):
        if step == 0:
            active_beams = [b for b in initial_beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        if len(active_beams) == 0:
            break

        if step == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        conversations = [build_conv(b.prompt, b.current_text, config.system_prompt) for b in active_beams]

        continue_chat = step > 0
        add_prompt_prefix = step == 0

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        formatted_convs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=add_prompt_prefix,
            continue_final_message=continue_chat,
            tokenize=False,
        )

        lookahead = 0 if step == config.num_iterations - 1 else config.lookahead
        generation_outputs = generate_k_steps(formatted_convs, lookahead, llm, sampling_params, 1)

        prompts, completions = [], []
        for beam, output in zip(active_beams, generation_outputs, strict=True):
            beam.next_texts = output.next_texts
            beam.stop_reasons = output.stop_reasons
            beam.lookahead_texts = output.lookahead_texts
            beam.completion_tokens += output.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if beam.stop_reasons[0] in ("EOS", "length") or beam.next_texts[0] == "":
                beam.completed = True
                finalized_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        reward_scores = prm.score(prompts, completions)
        aggregated_scores = [[aggregate_scores(s, config.agg_strategy) for s in score] for score in reward_scores]

        for beam, score in zip(active_beams, reward_scores, strict=True):
            beam.all_scores = score[0]

        aggregated_scores = [aggregated_scores[i] for i, b in enumerate(active_beams) if not b.completed]
        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

        if config.filter_duplicates:
            unique_beams = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beams:
                    unique_beams[b.current_text] = i
            active_beams = [active_beams[i] for i in unique_beams.values()]
            aggregated_scores = [aggregated_scores[i] for i in unique_beams.values()]

        flat_scores = np.array(aggregated_scores).flatten()
        max_score = np.max(flat_scores)
        delta = getattr(config, "dynamic_beam_delta", 0.3)

        top_indices = [i for i, s in enumerate(flat_scores) if max_score - s <= delta]
        min_beams = getattr(config, "min_beams", 2)
        max_beams = getattr(config, "max_beams", config.n)

        if len(top_indices) < min_beams:
            top_indices = np.argsort(flat_scores)[-min_beams:]
        elif len(top_indices) > max_beams:
            top_indices = np.argsort(flat_scores)[-max_beams:]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

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
        logger.debug(
            f"Extending finalized_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_finalized_beams = [
            copy.deepcopy(b) for b in (finalized_beams * repeats)[: config.n]
        ]
        finalized_beams = extended_finalized_beams

    return finalized_beams


def run_dynamic_beam_search(example_batch, config: Config, llm: LLM, prm: PRM):
    problems = example_batch["problem"]
    beam_outputs = _dynamic_beam_search(problems, config, llm, prm)

    grouped_by_prompt = defaultdict(list)
    for beam in beam_outputs:
        grouped_by_prompt[beam.prompt].append(beam)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for problem in problems:
        beam_group = grouped_by_prompt[problem]
        completions = [b.current_text for b in beam_group]
        scores = [aggregate_scores(b.all_scores, config.agg_strategy) for b in beam_group]
        best_completion = completions[np.argmax(scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beam_group])
        results["pred"].append(best_completion)
        results["completion_tokens"].append([b.completion_tokens for b in beam_group])

    return results
