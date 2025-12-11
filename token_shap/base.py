# base.py

from typing import Optional, List, Dict, Tuple, Union, Any, Callable, Set
import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import random
from itertools import combinations
import torch
from PIL import Image
import requests
import base64
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def encode_image_to_base64(image_path: str) -> str:
    """
    Converts an image file to a base64-encoded string.
    
    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def interact_with_ollama(
    prompt: Optional[str] = None, 
    messages: Optional[Any] = None,
    image_path: Optional[str] = None,
    model: str = 'openhermes2.5-mistral', 
    stream: bool = False, 
    output_handler: Callable[[str], None] = default_output_handler,
    api_url: Optional[str] = None
) -> str:
    """
    Sends a request to the Ollama API for chat or image generation tasks.
    
    Args:
    prompt (Optional[str]): Text prompt for generating content.
    messages (Optional[Any]): Structured chat messages for dialogues.
    image_path (Optional[str]): Path to an image for image-related operations.
    model (str): Model identifier for the API.
    stream (bool): If True, keeps the connection open for streaming responses.
    output_handler (Callable[[str], None]): Function to handle output messages.
    api_url (Optional[str]): API endpoint URL; if not provided, fetched from environment.

    Returns:
    str: Full response from the API.

    Raises:
    ValueError: If necessary parameters are not correctly provided or API URL is missing.
    """
    if not api_url:
        api_url = getenv('API_URL')
        if not api_url:
            raise ValueError('API_URL is not set. Provide it via the api_url variable or as an environment variable.')

    # Define endpoint based on whether it's a chat or a single generate request
    api_endpoint = f"{api_url}/api/{'chat' if messages else 'generate'}"
    data = {'model': model, 'stream': stream}

    # Populate the request data
    if messages:
        data['messages'] = messages
    elif prompt:
        data['prompt'] = prompt
        if image_path:
            data['images'] = [encode_image_to_base64(image_path)]
    else:
        raise ValueError("Either messages for chat or a prompt for generate must be provided.")

    # Send request
    response = requests.post(api_endpoint, json=data, stream=stream)
    if response.status_code != 200:
        output_handler(f"Failed to retrieve data: {response.status_code}")
        return ""

    # Handle streaming or non-streaming responses
    full_response = ""
    if stream:
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line.decode('utf-8'))
                response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
                full_response += response_part
                output_handler(response_part)
                if json_response.get('done', False):
                    break
    else:
        json_response = response.json()
        response_part = json_response.get('response', '') or json_response.get('message', {}).get('content', '')
        full_response += response_part
        output_handler(response_part)

    return full_response, response

def get_text_before_last_underscore(token):
    return token.rsplit('_', 1)[0]

class TextVectorizer:
    """Base class for text vectorization"""
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ModelBase(ABC):
    """Base class for all models (text and vision)"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.client = None
        
    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        """Generate response from model"""
        pass

    def generate_with_tools(self,
                           prompt: str,
                           tools: List[Dict],
                           tool_executor: Optional[Callable[[str, Dict], str]] = None,
                           max_iterations: int = 10) -> Tuple[str, Dict[str, int]]:
        """
        Generate response with tool calling support.

        Args:
            prompt: The user prompt
            tools: List of tool definitions (API format)
            tool_executor: Callable(tool_name, args) -> result string
            max_iterations: Max tool call iterations

        Returns:
            Tuple of (response_text, tool_usage_counts)

        Note: Subclasses should override this for proper tool support.
        Default implementation just calls generate() without tools.
        """
        # Default: no tool support, just generate
        return self.generate(prompt), {}

class HuggingFaceEmbeddings(TextVectorizer):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize HuggingFace sentence embeddings vectorizer - much simpler implementation
        
        Args:
            model_name: Name of the sentence-transformer model from HuggingFace
            device: Device to run model on ('cpu' or 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load model - SentenceTransformer handles all the complexity
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Please install with 'pip install sentence-transformers'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers - much simpler"""
        if not self.model:
            self._initialize_model()
            
        # SentenceTransformer handles batching, padding, etc. automatically
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Sentence-transformers models already return normalized vectors
        return np.dot(comparison_vectors, base_vector)

class OpenAIEmbeddings(TextVectorizer):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings vectorizer
        
        Args:
            api_key: OpenAI API key
            model: Embeddings model to use (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        if not self.client:
            self._initialize_client()
            
        # Process texts in batches to avoid rate limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                # Extract embeddings from response
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise Exception(f"Error getting embeddings from OpenAI: {str(e)}")
                
        return np.array(all_embeddings)
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Normalize vectors
        base_norm = np.linalg.norm(base_vector)
        comparison_norms = np.linalg.norm(comparison_vectors, axis=1)
        
        # Avoid division by zero
        if base_norm == 0 or np.any(comparison_norms == 0):
            return np.zeros(len(comparison_vectors))
            
        normalized_base = base_vector / base_norm
        normalized_comparisons = comparison_vectors / comparison_norms[:, np.newaxis]
        
        # Calculate cosine similarity
        similarities = np.dot(normalized_comparisons, normalized_base)
        return similarities

class TfidfTextVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = None
        
    def vectorize(self, texts: List[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer().fit(texts)
        return self.vectorizer.transform(texts).toarray()
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            base_vector.reshape(1, -1), comparison_vectors
        ).flatten()

import base64
from io import BytesIO
from typing import Optional, Union
from openai import OpenAI

class OpenAIModel(ModelBase):
    """Generic AI Model API wrapper supporting multiple providers"""

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        """
        :param model_name: Name of the model (e.g., "gpt-4-turbo", "gemini-2.0-flash", "claude-3").
        :param api_key: API key for the respective provider.
        :param base_url: Optional base URL (e.g., Gemini, Claude, OpenAI custom endpoint).
        """
        super().__init__(model_name, api_key=api_key)
        self.base_url = base_url
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the API client with the given base_url."""
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.base_url else OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with 'pip install openai'")

    def generate(self, prompt: str, image_path: Optional[Union[str, BytesIO]] = None) -> str:
        """Generates text based on a prompt with optional vision support."""
        if not self.client:
            self._initialize_client()

        try:
            messages = []
            if image_path:
                # Convert image to base64
                if isinstance(image_path, str):
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                elif isinstance(image_path, BytesIO):
                    base64_image = base64.b64encode(image_path.getvalue()).decode('utf-8')
                else:
                    raise ValueError("image_path must be either a file path or a BytesIO object")

                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                })
            else:
                messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5
            )

            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def generate_with_tools(self,
                           prompt: str,
                           tools: List[Dict],
                           tool_executor: Optional[Callable[[str, Dict], str]] = None,
                           max_iterations: int = 10) -> Tuple[str, Dict[str, int]]:
        """
        Generate response with tool calling support using OpenAI Chat Completions API.

        Args:
            prompt: The user prompt
            tools: List of tool definitions (OpenAI format)
            tool_executor: Callable(tool_name, args_dict) -> result string
            max_iterations: Max tool call iterations

        Returns:
            Tuple of (response_text, tool_usage_counts)
        """
        if not self.client:
            self._initialize_client()

        # Track tool usage
        tool_usage = {}

        # If no tools provided, just generate normally
        if not tools:
            return self.generate(prompt), tool_usage

        messages = [{"role": "user", "content": prompt}]

        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    temperature=0.5
                )

                message = response.choices[0].message

                # Check if model wants to call tools
                if not message.tool_calls:
                    # No more tool calls - return final response
                    return message.content or "", tool_usage

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    # Track usage
                    if func_name not in tool_usage:
                        tool_usage[func_name] = 0
                    tool_usage[func_name] += 1

                    # Execute the tool if executor provided
                    if tool_executor:
                        try:
                            result = tool_executor(func_name, func_args)
                        except Exception as e:
                            result = f"Error executing {func_name}: {str(e)}"
                    else:
                        result = f"Tool {func_name} called but no executor provided"

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

            except Exception as e:
                # On error, return what we have
                return f"Error in tool calling: {str(e)}", tool_usage

        # Max iterations reached - return last content
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"], tool_usage

        return "Max iterations reached without final response", tool_usage

class OllamaModel(ModelBase):
    """Ollama model implementation supporting both text and vision"""
    
    def __init__(self, model_name: str, api_url: str):
        super().__init__(model_name, api_url=api_url)
    
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        # Note: Implement the interact_with_ollama function from your original code
        # Add PLACEHOLDER comment to keep existing implementation
        text_response, _ = interact_with_ollama(
            model=self.model_name,
            prompt=prompt,
            image_path=image_path,
            api_url=self.api_url,
            output_handler=lambda o: o
        )
        return text_response

class LocalModel(ModelBase):
    """Local model implementation supporting both text and vision using HuggingFace models"""
    
    def __init__(self, 
                model_name: str,
                model_type: str = "text",  # "text" or "vision"
                max_new_tokens: int = 100,
                temperature: float = 0.5,
                device: str = "auto",
                torch_dtype: Optional[str] = "bfloat16",
                **model_kwargs):
        """
        Initialize local model
        
        Args:
            model_name: HuggingFace model name/path
            model_type: "text" or "vision"
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            device: Device to run model on ("auto", "cuda", "cpu")
            torch_dtype: Torch data type for model
            **model_kwargs: Additional kwargs for model initialization
        """
        super().__init__(model_name)
        self.device = device
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.torch_dtype = getattr(torch, torch_dtype) if torch_dtype else None
        self.model_kwargs = model_kwargs
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the appropriate model and tokenizer/processor"""
        try:
            if self.model_type == "text":
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device,
                    **self.model_kwargs
                )
            elif self.model_type == "vision":
                from transformers import AutoProcessor
                # Import the specific model class based on model_name
                # This is just an example - adjust based on your needs
                if "llama" in self.model_name.lower():
                    from transformers import LlamaForConditionalGeneration
                    model_class = LlamaForConditionalGeneration
                else:
                    from transformers import AutoModelForCausalLM
                    model_class = AutoModelForCausalLM
                    
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = model_class.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device,
                    **self.model_kwargs
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            raise ImportError(f"Error initializing model: {str(e)}")
            
    def _process_image(self, image_path: str) -> Image:
        """Load and process image file"""
        try:
            if image_path.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            return image
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        Generate text from prompt and optional image
        
        Args:
            prompt: Text prompt
            image_path: Optional path to image file
            
        Returns:
            Generated text response
        """
        try:
            if image_path and self.model_type != "vision":
                raise ValueError("Image input provided but model does not support vision")
                
            if image_path:
                # Vision model processing
                image = self._process_image(image_path)
                inputs = self.processor(
                    image, 
                    prompt,
                    return_tensors="pt"
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )
                return self.processor.decode(outputs[0])
            else:
                # Text-only processing
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                generated_tokens = outputs[0][input_length:]
                return self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()
                
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")

class BaseSHAP(ABC):
    """Base class for SHAP implementations"""
    
    def __init__(self, 
                 model: ModelBase,
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        self.model = model
        self.vectorizer = vectorizer
        self.debug = debug
        self.results_df = None
        self.shapley_values = None

    def _debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            print(message)

    def _calculate_baseline(self, content: Any, **kwargs) -> str:
        """Calculate baseline model response"""
        return self.model.generate(**self._prepare_generate_args(content, **kwargs))

    @abstractmethod
    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        pass

    def _generate_random_combinations(self, 
                                    samples: List[Any], 
                                    k: int, 
                                    exclude_combinations_set: Set[Tuple[int, ...]]) -> List[Tuple[List, Tuple[int, ...]]]:
        """
        Generate random combinations efficiently using binary representation
        """
        n = len(samples)
        sampled_combinations_set = set()
        max_attempts = k * 10  # Prevent infinite loops in case of duplicates
        attempts = 0

        while len(sampled_combinations_set) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)
            combination = [samples[i] for i in range(n) if bin_str[i] == '1']
            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_combinations_set:
                sampled_combinations_set.add((tuple(combination), indexes))

        if len(sampled_combinations_set) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations_set)} unique combinations out of requested {k}")
        return list(sampled_combinations_set)

    def _get_result_per_combination(self, 
                                content: Any, 
                                sampling_ratio: float,
                                max_combinations: Optional[int] = 1000) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        """
        Get model responses for combinations
        
        Args:
            content: Content to analyze
            sampling_ratio: Ratio of non-essential combinations to sample (0-1)
            max_combinations: Maximum number of combinations (must be >= n for n tokens)
        """
        samples = self._get_samples(content)
        n = len(samples)
        self._debug_print(f"Number of samples: {n}")
        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        # Always start with essential combinations (each missing one sample)
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)
        
        num_essential = len(essential_combinations)
        self._debug_print(f"Number of essential combinations: {num_essential}")
        if max_combinations is not None and max_combinations < num_essential:
            print(f"Warning: max_combinations ({max_combinations}) is less than the number of essential combinations "
                  f"({num_essential}). Will use all essential combinations despite the limit.")
            self._debug_print("No additional combinations will be added.")
            max_combinations = num_essential
        # Calculate how many additional combinations we can/should generate
        remaining_budget = float('inf')
        if max_combinations is not None:
            remaining_budget = max(0, max_combinations - num_essential)
            self._debug_print(f"Remaining combinations budget after essentials: {remaining_budget}")

        # If using sampling ratio, calculate possible additional combinations without generating them
        if sampling_ratio < 1.0:
            # Get theoretical number of total combinations
            theoretical_total = 2 ** n - 1
            theoretical_additional = theoretical_total - num_essential
            # Calculate desired number based on ratio
            desired_additional = int(theoretical_additional * sampling_ratio)
            # Take minimum of sampling ratio and max_combinations limits
            num_additional = min(desired_additional, remaining_budget)
        else:
            num_additional = remaining_budget

        num_additional = int(num_additional)  # Ensure integer
        self._debug_print(f"Number of additional combinations to sample: {num_additional}")

        # Generate additional random combinations if needed
        additional_combinations = []
        if num_additional > 0:
            additional_combinations = self._generate_random_combinations(
                samples, num_additional, essential_combinations_set
            )
            self._debug_print(f"Number of sampled combinations: {len(additional_combinations)}")
        else:
            self._debug_print("No additional combinations to sample.")

        # Process all combinations
        all_combinations = essential_combinations + additional_combinations
        self._debug_print(f"Total combinations to process: {len(all_combinations)}")

        responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations, desc="Processing combinations")):
            self._debug_print(f"\nProcessing combination {idx + 1}/{len(all_combinations)}:")
            self._debug_print(f"Combination: {combination}")
            self._debug_print(f"Indexes: {indexes}")

            args = self._prepare_combination_args(combination, content)
            response = self.model.generate(**args)
            self._debug_print(f"Received response for combination {idx + 1}")

            key = self._get_combination_key(combination, indexes)
            responses[key] = (response, indexes)

        return responses

    def _get_df_per_combination(self, responses: Dict[str, Tuple[str, Tuple[int, ...]]], baseline_text: str) -> pd.DataFrame:
        """Create DataFrame with combination results"""
        df = pd.DataFrame(
            [(key.split('_')[0], response[0], response[1])
             for key, response in responses.items()],
            columns=['Content', 'Response', 'Indexes']
        )

        all_texts = [baseline_text] + df["Response"].tolist()
        vectors = self.vectorizer.vectorize(all_texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        
        similarities = self.vectorizer.calculate_similarity(base_vector, comparison_vectors)
        df["Similarity"] = similarities

        return df

    def _calculate_shapley_values(self, df: pd.DataFrame, content: Any) -> Dict[str, float]:
        """Calculate Shapley values"""
        samples = self._get_samples(content)
        shapley_values = {}

        def normalize_shapley_values(values: Dict[str, float], power: float = 1) -> Dict[str, float]:
            min_value = min(values.values())
            shifted_values = {k: v - min_value for k, v in values.items()}
            powered_values = {k: v ** power for k, v in shifted_values.items()}
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1 / len(powered_values) for k in powered_values}
            return {k: v / total for k, v in powered_values.items()}

        for i, sample in enumerate(samples, start=1):
            with_sample = np.average(
                df[df["Indexes"].apply(lambda x: i in x)]["Similarity"].values
            )
            without_sample = np.average(
                df[df["Indexes"].apply(lambda x: i not in x)]["Similarity"].values
            )

            shapley_values[f"{sample}_{i}"] = with_sample - without_sample

        return normalize_shapley_values(shapley_values)

    @abstractmethod
    def _get_samples(self, content: Any) -> List[Any]:
        """Get samples from content for analysis"""
        pass

    @abstractmethod
    def _prepare_combination_args(self, combination: List[Any], original_content: Any) -> Dict:
        """Prepare model arguments for a combination"""
        pass

    @abstractmethod
    def _get_combination_key(self, combination: List[Any], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        pass

    def save_results(self, output_dir: str, metadata: Optional[Dict] = None) -> None:
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results_df is not None:
            self.results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
            
        if self.shapley_values is not None:
            with open(os.path.join(output_dir, "shapley_values.json"), 'w') as f:
                json.dump(self.shapley_values, f, indent=2)
                
        if metadata:
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)