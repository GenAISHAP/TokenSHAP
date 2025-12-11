"""
AgentSHAP - Analyze tool importance in AI agent responses using Shapley values.

This module provides AgentSHAP, which analyzes which tools contribute most to
an agent's response quality by running the agent with different tool combinations
and measuring the impact on output similarity.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from .base import BaseSHAP, ModelBase, TextVectorizer
from .tools import Tool


class AgentSHAP(BaseSHAP):
    """
    Analyzes tool importance in AI agent responses using SHAP values.

    AgentSHAP measures how much each tool contributes to the quality of an agent's
    response by running the agent with different tool combinations and calculating
    Shapley values based on response similarity.

    Attributes:
        model: The language model to use (OpenAIModel, etc.)
        tools: List of Tool objects with bundled executors
        vectorizer: Text embedding model for similarity calculation
        max_iterations: Maximum iterations for agentic tool execution loop
        debug: Enable debug logging

    Example:
        from token_shap import AgentSHAP, OpenAIModel, create_function_tool

        model = OpenAIModel(model_name="gpt-4o", api_key="...")

        tools = [
            create_function_tool(
                name="get_weather",
                description="Get weather for a city",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                executor=lambda args: f"Weather in {args['city']}: 72F"
            )
        ]

        agent_shap = AgentSHAP(model=model, tools=tools)
        results_df, shapley_values = agent_shap.analyze(prompt="What's the weather in NYC?")
    """

    def __init__(self,
                 model: ModelBase,
                 tools: List[Tool],
                 vectorizer: Optional[TextVectorizer] = None,
                 max_iterations: int = 10,
                 debug: bool = False):
        """
        Initialize AgentSHAP.

        Args:
            model: ModelBase instance (OpenAIModel, OllamaModel, etc.)
            tools: List of Tool objects with bundled executors
            vectorizer: Text embedding model for similarity (uses default if None)
            max_iterations: Maximum iterations for agentic execution loop
            debug: Enable debug logging
        """
        super().__init__(model, vectorizer, debug)
        self.tools = tools
        self.max_iterations = max_iterations
        self._tool_map: Dict[str, Tool] = {t.name: t for t in tools}

        # Analysis state
        self._baseline_response: Optional[str] = None
        self._baseline_tool_usage: Dict[str, int] = {}
        self._prompt: Optional[str] = None
        self._tool_usage_counts: Dict[str, int] = {}
        self._tool_usage_per_combination: Dict[str, Dict[str, int]] = {}

    def _get_samples(self, content: Any) -> List[str]:
        """Get list of tool names for SHAP analysis."""
        return [tool.name for tool in self.tools]

    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for baseline generation with ALL tools."""
        return {"prompt": content, "tools": self.tools}

    def _prepare_combination_args(self, combination: List[str], original_content: Any) -> Dict:
        """Prepare arguments with only a SUBSET of tools."""
        filtered_tools = [t for t in self.tools if t.name in combination]
        return {"prompt": original_content, "tools": filtered_tools}

    def _get_combination_key(self, combination: List[str], indexes: Tuple[int, ...]) -> str:
        """Generate a unique key for this tool combination."""
        tools_str = "+".join(sorted(combination))
        return f"{tools_str}_idx{','.join(map(str, indexes))}"

    def _execute_with_tools(self,
                            prompt: str,
                            tools: List[Tool],
                            combination_key: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """Execute the model with the given tools."""
        if not tools:
            return self.model.generate(prompt=prompt), {}

        response, tool_usage = self.model.generate_with_tools(
            prompt=prompt,
            tools=[t.definition for t in tools],
            tool_executor=Tool.create_executor(tools),
            max_iterations=self.max_iterations
        )

        # Update global tracking
        for tool_name, count in tool_usage.items():
            self._tool_usage_counts[tool_name] = self._tool_usage_counts.get(tool_name, 0) + count

        self._debug_print(f"Tool usage for {combination_key or 'unknown'}: {tool_usage}")
        return response, tool_usage

    def _get_result_per_combination(self,
                                    content: Any,
                                    sampling_ratio: float,
                                    max_combinations: Optional[int] = None) -> Dict[str, Tuple[str, Tuple[int, ...], Dict[str, int]]]:
        """Get agent responses for different tool combinations."""
        samples = self._get_samples(content)
        n = len(samples)

        if n == 0:
            raise ValueError("No tools provided for analysis")

        # Essential combinations: each missing one tool
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)

        num_essential = len(essential_combinations)
        self._debug_print(f"Number of tools: {n}")
        self._debug_print(f"Number of essential combinations (each missing one tool): {num_essential}")

        # Calculate additional combinations
        theoretical_total = 2 ** n - 1
        theoretical_additional = theoretical_total - num_essential

        if max_combinations is not None:
            remaining_budget = max(0, max_combinations - num_essential)
        else:
            remaining_budget = theoretical_additional

        if sampling_ratio < 1.0:
            num_additional = min(int(theoretical_additional * sampling_ratio), remaining_budget)
        else:
            num_additional = remaining_budget

        self._debug_print(f"Additional combinations to sample: {num_additional}")

        # Generate additional random combinations
        additional_combinations = []
        if num_additional > 0:
            additional_combinations = self._generate_random_combinations(
                samples, num_additional, essential_combinations_set
            )

        all_combinations = essential_combinations + additional_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations)}")

        responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations, desc="Testing tool combinations")):
            self._debug_print(f"\nCombination {idx + 1}: {combination}")

            filtered_tools = [t for t in self.tools if t.name in combination]
            key = self._get_combination_key(combination, indexes)
            response, tool_usage = self._execute_with_tools(content, filtered_tools, key)

            self._tool_usage_per_combination[key] = tool_usage
            responses[key] = (response, indexes, tool_usage)

        return responses

    def _get_df_per_combination_with_usage(self,
                                           responses: Dict[str, Tuple[str, Tuple[int, ...], Dict[str, int]]],
                                           baseline_text: str) -> pd.DataFrame:
        """Create DataFrame with combination results including tool usage."""
        data = []
        for key, (response, indexes, tool_usage) in responses.items():
            row = {
                'combination_key': key,
                'tools_available': key.split('_idx')[0],
                'response': response,
                'indexes': indexes,
            }
            for tool_name, count in tool_usage.items():
                row[f'used_{tool_name}'] = count
            data.append(row)

        df = pd.DataFrame(data)

        # Calculate similarities
        all_texts = [baseline_text] + df["response"].tolist()
        vectors = self.vectorizer.vectorize(all_texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        similarities = self.vectorizer.calculate_similarity(base_vector, comparison_vectors)
        df["similarity"] = similarities

        return df

    def analyze(self,
                prompt: str,
                sampling_ratio: float = 0.5,
                max_combinations: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Analyze which tools contribute most to the agent's response.

        Args:
            prompt: The input prompt for the agent
            sampling_ratio: Fraction of non-essential combinations to sample (0-1)
            max_combinations: Maximum combinations to evaluate (None for all)

        Returns:
            Tuple of (results_df, shapley_values)
        """
        self._prompt = prompt
        self._tool_usage_counts = {}
        self._tool_usage_per_combination = {}

        if not self.tools:
            raise ValueError("No tools provided for analysis")

        print(f"\nAnalyzing {len(self.tools)} tools:")
        for tool in self.tools:
            print(f"  - {tool.name}")

        # Baseline with all tools
        print("\n1. Getting baseline response (all tools)...")
        self._baseline_response, self._baseline_tool_usage = self._execute_with_tools(
            prompt, self.tools, "baseline"
        )
        self._tool_usage_per_combination["baseline"] = self._baseline_tool_usage
        self._debug_print(f"Baseline response: {self._baseline_response[:200]}...")

        # Test combinations
        print("\n2. Testing tool combinations...")
        responses = self._get_result_per_combination(prompt, sampling_ratio, max_combinations)

        # Calculate similarities
        print("\n3. Calculating similarities...")
        self.results_df = self._get_df_per_combination_with_usage(responses, self._baseline_response)

        # Calculate Shapley values
        print("\n4. Computing Shapley values...")
        simple_responses = {k: (v[0], v[1]) for k, v in responses.items()}
        simple_df = self._get_df_per_combination(simple_responses, self._baseline_response)
        self.shapley_values = self._calculate_shapley_values(simple_df, prompt)

        # Clean up keys
        clean_shapley = {}
        for key, value in self.shapley_values.items():
            parts = key.rsplit('_', 1)
            tool_name = parts[0] if len(parts) > 1 else key
            clean_shapley[tool_name] = value
        self.shapley_values = clean_shapley

        print("\n Analysis complete!")
        return self.results_df, self.shapley_values

    def get_tool_usage_summary(self) -> pd.DataFrame:
        """Get summary of tool usage across all combinations."""
        if not self._tool_usage_per_combination:
            raise ValueError("No analysis results. Call analyze() first.")

        data = []
        for tool in self.tools:
            data.append({
                'tool_name': tool.name,
                'total_calls': self._tool_usage_counts.get(tool.name, 0),
                'baseline_calls': self._baseline_tool_usage.get(tool.name, 0),
                'shap_value': self.shapley_values.get(tool.name, 0) if self.shapley_values else 0,
            })

        return pd.DataFrame(data).sort_values('shap_value', ascending=False)

    def print_colored_tools(self) -> None:
        """
        Print tools with colors based on importance (like TokenSHAP).
        Red = high importance, Blue = low importance.
        """
        if self.shapley_values is None:
            raise ValueError("No analysis results. Call analyze() first.")

        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())

        def get_color(value):
            if max_value == min_value:
                norm_value = 0.5
            else:
                norm_value = (value - min_value) / (max_value - min_value)
            # Blue (low) to Red (high)
            if norm_value < 0.5:
                r = int(255 * (norm_value * 2))
                g = int(255 * (norm_value * 2))
                b = 255
            else:
                r = 255
                g = int(255 * (2 - norm_value * 2))
                b = int(255 * (2 - norm_value * 2))
            return r, g, b

        sorted_tools = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)
        for tool_name, value in sorted_tools:
            r, g, b = get_color(value)
            print(f"\033[38;2;{r};{g};{b}m{tool_name}\033[0m", end='  ')
        print()

    def highlight_tools_background(self) -> None:
        """
        Print tools with background colors based on importance.
        Yellow = high importance, White = low importance.
        """
        if self.shapley_values is None:
            raise ValueError("No analysis results. Call analyze() first.")

        min_value = min(self.shapley_values.values())
        max_value = max(self.shapley_values.values())

        sorted_tools = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)
        for tool_name, value in sorted_tools:
            if max_value == min_value:
                norm_value = 0.5
            else:
                norm_value = ((value - min_value) / (max_value - min_value)) ** 2
            r = 255
            g = 255
            b = int(255 - (norm_value * 255))
            print(f"\033[48;2;{r};{g};{b}m {tool_name} \033[0m", end=' ')
        print()

    def plot_colored_tools(self, figsize: Tuple[int, int] = (12, 3)) -> plt.Figure:
        """
        Plot tools with importance colors using matplotlib (like TokenSHAP).
        Uses coolwarm colormap: Blue = low, Red = high.
        """
        if self.shapley_values is None:
            raise ValueError("No analysis results. Call analyze() first.")

        sorted_tools = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        min_val = min(self.shapley_values.values())
        max_val = max(self.shapley_values.values())

        x_pos = 0.05
        for tool_name, value in sorted_tools:
            if max_val == min_val:
                norm_value = 0.5
            else:
                norm_value = (value - min_val) / (max_val - min_val)
            color = plt.cm.coolwarm(norm_value)

            ax.text(x_pos, 0.5, tool_name, color=color, fontsize=16, fontweight='bold',
                   ha='left', va='center', transform=ax.transAxes)
            x_pos += len(tool_name) * 0.025 + 0.05

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                                   norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.3, aspect=40)
        cbar.set_label('SHAP Value (Tool Importance)', fontsize=11)

        plt.tight_layout()
        return fig

    def plot_tool_importance(self,
                             figsize: Tuple[int, int] = (10, 6),
                             title: Optional[str] = None) -> plt.Figure:
        """
        Plot horizontal bar chart of tool importance with coolwarm colors.
        Red = high importance, Blue = low importance.
        """
        if self.shapley_values is None:
            raise ValueError("No analysis results. Call analyze() first.")

        sorted_tools = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)
        tools = [t[0] for t in sorted_tools]
        values = [t[1] for t in sorted_tools]

        min_val = min(values)
        max_val = max(values)

        fig, ax = plt.subplots(figsize=figsize)

        # Color bars using coolwarm colormap
        colors = []
        for v in values:
            if max_val == min_val:
                norm = 0.5
            else:
                norm = (v - min_val) / (max_val - min_val)
            colors.append(plt.cm.coolwarm(norm))

        bars = ax.barh(tools, values, color=colors, edgecolor='black', linewidth=0.5)

        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontsize=10)

        ax.set_xlabel('SHAP Value (Importance)', fontsize=12)
        ax.set_ylabel('Tool', fontsize=12)
        ax.set_title(title or 'Tool Importance Analysis (AgentSHAP)', fontsize=14)
        ax.invert_yaxis()

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                                   norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Importance', fontsize=10)

        plt.tight_layout()
        return fig

    def print_tool_ranking(self) -> None:
        """Print a formatted ranking of tool importance."""
        if self.shapley_values is None:
            raise ValueError("No analysis results. Call analyze() first.")

        sorted_tools = sorted(self.shapley_values.items(), key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 50)
        print("TOOL IMPORTANCE RANKING (AgentSHAP)")
        print("=" * 50)

        max_name_len = max(len(t[0]) for t in sorted_tools)
        max_value = max(t[1] for t in sorted_tools) if sorted_tools else 1

        for rank, (tool_name, value) in enumerate(sorted_tools, 1):
            bar_len = int((value / max_value) * 20) if max_value > 0 else 0
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"{rank}. {tool_name:<{max_name_len}} {bar} {value:.4f}")

        print("=" * 50)

    def get_detailed_results(self) -> Dict[str, Any]:
        """Get all detailed results for debugging and analysis."""
        return {
            'prompt': self._prompt,
            'baseline_response': self._baseline_response,
            'baseline_tool_usage': self._baseline_tool_usage,
            'shapley_values': self.shapley_values,
            'tool_usage_counts': self._tool_usage_counts,
            'tool_usage_per_combination': self._tool_usage_per_combination,
            'results_df': self.results_df,
            'tools': [{'name': t.name, 'description': t.description} for t in self.tools]
        }

    def compare_prompts(self,
                        prompts: List[str],
                        sampling_ratio: float = 0.5,
                        figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, List[Dict[str, float]]]:
        """
        Compare tool importance across different prompts.

        Shows how the same agent with the same tools assigns different importance
        to tools based on the query - demonstrating prompt-dependent behavior.

        Args:
            prompts: List of prompts to compare
            sampling_ratio: Fraction of combinations to sample
            figsize: Figure size (auto-calculated if None)

        Returns:
            Tuple of (matplotlib figure, list of shapley_values dicts)
        """
        n_prompts = len(prompts)
        all_shap_values = []
        all_responses = []

        # Run analysis for each prompt
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*70}")
            print(f"Prompt {i+1}/{n_prompts}: {prompt[:60]}...")
            print('='*70)

            _, shap_values = self.analyze(prompt, sampling_ratio=sampling_ratio)
            all_shap_values.append(shap_values.copy())
            all_responses.append(self._baseline_response)

        # Create horizontal comparison visualization (prompts as columns)
        tool_names = [t.name for t in self.tools]
        n_tools = len(tool_names)

        if figsize is None:
            figsize = (3.5 * n_prompts, max(3.5, n_tools * 0.7))

        fig, axes = plt.subplots(1, n_prompts, figsize=figsize)
        if n_prompts == 1:
            axes = [axes]

        # Get global min/max for consistent coloring
        all_values = [v for shap in all_shap_values for v in shap.values()]
        global_min = min(all_values)
        global_max = max(all_values)

        for idx, (ax, prompt, shap_values) in enumerate(zip(axes, prompts, all_shap_values)):
            # Sort by this prompt's SHAP values
            sorted_items = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
            tools_sorted = [t[0] for t in sorted_items]
            values_sorted = [t[1] for t in sorted_items]

            # Color using global normalization
            colors = []
            for v in values_sorted:
                if global_max == global_min:
                    norm = 0.5
                else:
                    norm = (v - global_min) / (global_max - global_min)
                colors.append(plt.cm.coolwarm(norm))

            bars = ax.barh(tools_sorted, values_sorted, color=colors, edgecolor='black', linewidth=0.5)

            # Add value labels
            for bar, value in zip(bars, values_sorted):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', va='center', fontsize=6)

            # Truncate prompt for title (shorter for horizontal layout)
            short_prompt = prompt[:35] + "..." if len(prompt) > 35 else prompt
            ax.set_title(f'"{short_prompt}"', fontsize=8, style='italic', wrap=True, pad=10)
            ax.invert_yaxis()
            ax.set_xlim(0, global_max * 1.3)
            ax.set_xlabel('SHAP Value', fontsize=7)
            ax.tick_params(axis='y', labelsize=7)

        # Add colorbar on the right side
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                                   norm=plt.Normalize(vmin=global_min, vmax=global_max))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('Tool Importance', fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        fig.suptitle('Agent Tools XAI', fontsize=14, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.82, right=0.90, wspace=0.4)

        return fig, all_shap_values
