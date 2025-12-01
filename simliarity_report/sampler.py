"""Tinker sampler wrapper used for similarity report generation."""
from __future__ import annotations

from typing import List, Optional

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer


class TinkerSampler:
    """Wrapper around Tinker sampling client with renderer-aware prompts."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
    ) -> None:
        tokenizer = get_tokenizer(model_name)
        renderer_name = get_recommended_renderer_name(model_name)
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        service_client = tinker.ServiceClient()
        self.sampling_client = service_client.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )

    async def generate(self, messages: List[renderers.Message]) -> str:
        prompt = self.renderer.build_generation_prompt(messages)
        sampling_result = await self.sampling_client.sample_async(
            prompt=prompt,
            sampling_params=self.sampling_params,
            num_samples=1,
        )
        response_tokens = sampling_result.sequences[0].tokens
        raw_text = self.renderer.tokenizer.decode(response_tokens).strip()
        return raw_text


__all__ = ["TinkerSampler"]
