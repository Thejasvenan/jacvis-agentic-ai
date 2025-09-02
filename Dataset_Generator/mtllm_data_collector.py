
import json
import os
import hashlib
from datetime import datetime
from typing import Callable, Any, Dict, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MTLLMDataPoint:
    
    function_name: str
    function_description: str
    system_prompt: str
    user_prompt: str
    input_parameters: dict
    
    response: str
    parsed_output: Any
    
    model_name: str
    timestamp: str
    call_site_id: str
    response_schema: dict
    
    def to_training_format(self) -> dict:
        return {
            "messages": [
                {
                    "role": "system", 
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": self.user_prompt
                },
                {
                    "role": "assistant", 
                    "content": self.response
                }
            ],
            "metadata": {
                "function_name": self.function_name,
                "call_site_id": self.call_site_id,
                "model_name": self.model_name,
                "timestamp": self.timestamp,
                "schema": self.response_schema
            }
        }


class MTLLMDataCollector:
    
    _instance = None
    
    def __new__(cls, output_dir: str = "mtllm_dataset"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, output_dir: str = "mtllm_dataset"):
        if self._initialized:
            return
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.collected_data: List[MTLLMDataPoint] = []
        
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "training_data").mkdir(exist_ok=True)
        (self.output_dir / "by_function").mkdir(exist_ok=True)
        
        self._initialized = True
        print(f"🔧 MTLLMDataCollector initialized with output_dir: {self.output_dir}")
    
    def capture_llm_call(self, function_name: str, system_prompt: str, user_prompt: str, 
                         response: str, model_name: str, input_params: Dict, 
                         response_schema: Dict = None) -> None:
        
        call_site_id = self._generate_call_site_id(function_name, input_params)
        
        data_point = MTLLMDataPoint(
            function_name=function_name,
            function_description=function_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_parameters=input_params,
            response=response,
            parsed_output=response,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            call_site_id=call_site_id,
            response_schema=response_schema or {}
        )
        
        self._save_data_point(data_point)
    
    def _generate_call_site_id(self, function_name: str, args: Dict) -> str:
        content = f"{function_name}_{str(sorted(args.keys()))}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _save_data_point(self, data_point: MTLLMDataPoint):
        
        timestamp_safe = data_point.timestamp.replace(":", "-").replace(".", "-")
        
        raw_file = self.output_dir / "raw_data" / f"{data_point.call_site_id}_{timestamp_safe}.json"
        with open(raw_file, 'w') as f:
            json.dump(asdict(data_point), f, indent=2, default=str)
        
        training_file = self.output_dir / "training_data" / f"{data_point.call_site_id}_{timestamp_safe}.json"
        with open(training_file, 'w') as f:
            json.dump(data_point.to_training_format(), f, indent=2)
        
        function_dir = self.output_dir / "by_function" / data_point.function_name
        function_dir.mkdir(exist_ok=True)
        
        function_file = function_dir / f"{timestamp_safe}.json"
        with open(function_file, 'w') as f:
            json.dump(data_point.to_training_format(), f, indent=2)
        
        print(f" Collected data for {data_point.function_name} -> {raw_file.name}")


data_collector = MTLLMDataCollector()