#!/usr/bin/env python3
"""
Hardware-Adaptive RAG System with Enhanced Chunking for Electrical Utility Documentation
- Recursive text splitting with intelligent overlap
- Safety-critical content detection and prioritization
- Evaluation pipeline for chunking quality assessment
- Hardware optimization preserved from original implementation
"""

import os
import json
import logging
import numpy as np
import psutil
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import re
import time
import hashlib
import pickle
from dataclasses import dataclass, asdict, field
from sentence_transformers import SentenceTransformer
import requests
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.venv/rag_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware detection and capabilities"""
    # CPU Info
    cpu_count: int
    cpu_freq_max: float
    cpu_brand: str

    # Memory Info
    total_ram_gb: float
    available_ram_gb: float

    # GPU Info
    has_gpu: bool
    gpu_name: str
    gpu_vram_gb: float
    gpu_compute_capability: Optional[str]

    # Storage Info
    storage_type: str  # SSD, HDD, or Unknown

    # Platform Info
    platform: str
    python_version: str

    # Performance tier (LOW, MEDIUM, HIGH, ULTRA)
    performance_tier: str


class HardwareDetector:
    """Detects hardware capabilities and suggests optimal parameters"""

    def __init__(self):
        self.hardware_info = self._detect_hardware()
        self.optimal_params = self._calculate_optimal_parameters()
        
        
        
    def _parse_cpu_frequency_from_brand(self, cpu_brand: str, cpu_count: int) -> float:
        """Simple frequency extraction from CPU brand string"""
        
        
        # Look for frequency patterns in the CPU name
        frequency_patterns = [
            r'@\s*(\d+\.?\d*)\s*GHz',     # "@ 2.60GHz" 
            r'(\d+\.?\d*)\s*GHz',         # "2.6GHz" anywhere in string
            r'(\d+\.?\d*)\s*G\s*Hz',      # "2.6 G Hz" with spaces
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, cpu_brand, re.IGNORECASE)
            if match:
                try:
                    freq_ghz = float(match.group(1))
                    freq_mhz = freq_ghz * 1000
                    logger.info(f"Extracted {freq_ghz}GHz from CPU name: '{cpu_brand}'")
                    return freq_mhz
                except ValueError:
                    continue
        
        # If no frequency found in name, fall back to original logic
        if any(model in cpu_brand.upper() for model in ['I9', 'RYZEN 9']):
            return 4500.0  # High-end CPUs
        elif any(model in cpu_brand.upper() for model in ['I7', 'RYZEN 7']):
            return 4000.0  # Upper mid-range CPUs  
        elif any(model in cpu_brand.upper() for model in ['I5', 'RYZEN 5']):
            return 3500.0  # Mid-range CPUs
        elif cpu_count >= 8:
            return 3000.0  # Multi-core, assume decent
        elif cpu_count >= 4:
            return 2500.0  # Quad-core
        else:
            return 2000.0  # Low-end

    def _detect_hardware(self) -> HardwareInfo:
        # CPU Detection
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        # Get CPU brand first
        try:
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(['wmic', 'cpu', 'get', 'name'],
                                        capture_output=True, text=True)
                cpu_brand = result.stdout.split('\n')[1].strip() if len(result.stdout.split('\n')) > 1 else "Unknown"
            else:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_brand = line.split(':')[1].strip()
                            break
                    else:
                        cpu_brand = "Unknown"
        except:
            cpu_brand = f"{platform.processor()} ({cpu_count} cores)"

        # FIXED: Smart frequency detection
        if cpu_freq and cpu_freq.max:
            cpu_freq_max = cpu_freq.max
        else:
            # Parse frequency from CPU brand string
            cpu_freq_max = self._parse_cpu_frequency_from_brand(cpu_brand, cpu_count)
            logger.info(f"Parsed CPU frequency from brand: {cpu_freq_max:.1f}GHz")

        # Memory Detection
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024 ** 3)
        available_ram_gb = memory.available / (1024 ** 3)

        # GPU Detection
        has_gpu, gpu_name, gpu_vram_gb, gpu_compute_capability = self._detect_gpu()

        # Storage Detection
        storage_type = self._detect_storage_type()

        # Performance Tier Calculation
        performance_tier = self._calculate_performance_tier(
            cpu_count, cpu_freq_max, total_ram_gb, has_gpu, gpu_vram_gb
        )

        hardware_info = HardwareInfo(
            cpu_count=cpu_count,
            cpu_freq_max=cpu_freq_max,
            cpu_brand=cpu_brand,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            gpu_compute_capability=gpu_compute_capability,
            storage_type=storage_type,
            platform=platform.system(),
            python_version=platform.python_version(),
            performance_tier=performance_tier
        )

        self._log_hardware_info(hardware_info)
        return hardware_info

    def _detect_gpu(self) -> Tuple[bool, str, float, Optional[str]]:
        """Detect GPU capabilities"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)

                # Get VRAM
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_vram_gb = gpu_props.total_memory / (1024 ** 3)

                # Get compute capability
                compute_capability = f"{gpu_props.major}.{gpu_props.minor}"

                # Test GPU accessibility
                try:
                    test_tensor = torch.randn(100, 100).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    return True, gpu_name, gpu_vram_gb, compute_capability
                except Exception as e:
                    logger.warning(f"GPU detected but not accessible: {e}")
                    return False, "GPU (Inaccessible)", 0.0, None
            else:
                return False, "No GPU", 0.0, None
        except ImportError:
            logger.info("PyTorch not available for GPU detection")
            return False, "PyTorch N/A", 0.0, None

    def _detect_storage_type(self) -> str:
        """Detect storage type (SSD vs HDD)"""
        try:
            if platform.system() == "Linux":
                # Check for SSD on Linux
                with open('/proc/mounts', 'r') as f:
                    for line in f:
                        if '/ ' in line:
                            device = line.split()[0]
                            if '/dev/nvme' in device or '/dev/sd' in device:
                                # Simple heuristic - NVMe is likely SSD
                                return "SSD" if 'nvme' in device else "Unknown"
                return "Unknown"
            elif platform.system() == "Windows":
                # Windows detection would require WMI queries
                return "Unknown"
            else:
                return "Unknown"
        except:
            return "Unknown"

    def _calculate_performance_tier(self, cpu_count: int, cpu_freq: float,
                                    ram_gb: float, has_gpu: bool, gpu_vram_gb: float) -> str:
        """Calculate performance tier based on hardware"""
        score = 0

        # CPU Score (0-40 points)
        if cpu_count >= 16:
            score += 20
        elif cpu_count >= 8:
            score += 15
        elif cpu_count >= 4:
            score += 10
        else:
            score += 5

        if cpu_freq >= 3.5:
            score += 20
        elif cpu_freq >= 2.5:
            score += 15
        elif cpu_freq >= 2.0:
            score += 10
        else:
            score += 5

        # RAM Score (0-30 points)
        if ram_gb >= 32:
            score += 30
        elif ram_gb >= 16:
            score += 20
        elif ram_gb >= 8:
            score += 10
        else:
            score += 5

        # GPU Score (0-30 points)
        if has_gpu:
            if gpu_vram_gb >= 8:
                score += 30
            elif gpu_vram_gb >= 6:
                score += 25
            elif gpu_vram_gb >= 4:
                score += 20
            else:
                score += 10

        # Determine tier
        if score >= 85:
            return "ULTRA"
        elif score >= 65:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        else:
            return "LOW"

    def _calculate_optimal_parameters(self) -> Dict[str, Any]:
        """Calculate optimal parameters based on hardware"""
        hw = self.hardware_info
        params = {}

        # Enhanced chunking parameters (based on specifications)
        params['chunk_size'] = 128  # Primary for safety procedures
        params['chunk_size_alt'] = 256  # For equipment specifications
        params['chunk_overlap'] = 16  # For 128-token chunks (12.5% overlap)
        params['chunk_overlap_alt'] = 25  # For 256-token chunks (10% overlap)
        params['similarity_top_k'] = 10  # Increased for Llama 3.2 3B context window
        params['similarity_threshold'] = 0.25  # Relaxed for better recall (will be post-filtered)

        # Embedding batch size based on GPU/CPU capability
        if hw.has_gpu:
            if hw.gpu_vram_gb >= 8:
                params['embedding_batch_size'] = 64
            elif hw.gpu_vram_gb >= 6:
                params['embedding_batch_size'] = 48
            elif hw.gpu_vram_gb >= 4:
                params['embedding_batch_size'] = 32
            else:
                params['embedding_batch_size'] = 16
        else:
            # CPU-only: smaller batches, depends on RAM
            if hw.total_ram_gb >= 16:
                params['embedding_batch_size'] = 16
            elif hw.total_ram_gb >= 8:
                params['embedding_batch_size'] = 8
            else:
                params['embedding_batch_size'] = 4

        # Parallel processing based on CPU cores
        if hw.cpu_count >= 8:
            params['max_workers'] = min(hw.cpu_count - 2, 8)  # Leave some cores free
            params['parallel_init'] = True
        elif hw.cpu_count >= 4:
            params['max_workers'] = hw.cpu_count - 1
            params['parallel_init'] = True
        else:
            params['max_workers'] = 1
            params['parallel_init'] = False

        # Cache settings based on available RAM
        if hw.available_ram_gb >= 8:
            params['max_cache_size_mb'] = 1000
            params['cache_ttl_hours'] = 48
        elif hw.available_ram_gb >= 4:
            params['max_cache_size_mb'] = 500
            params['cache_ttl_hours'] = 24
        else:
            params['max_cache_size_mb'] = 100
            params['cache_ttl_hours'] = 12

        # Model optimization flags
        params['warm_models_on_startup'] = hw.performance_tier in ['HIGH', 'ULTRA']
        params['optimize_data_loading'] = hw.storage_type == 'SSD' or hw.performance_tier in ['HIGH', 'ULTRA']
        params['use_model_cache'] = hw.available_ram_gb >= 4

        # Ollama parameters based on RAM
        if hw.total_ram_gb >= 32:
            params['ollama_keep_alive'] = "24h"
            params['max_context_length'] = 6000
        elif hw.total_ram_gb >= 16:
            params['ollama_keep_alive'] = "12h"
            params['max_context_length'] = 4000
        elif hw.total_ram_gb >= 8:
            params['ollama_keep_alive'] = "6h"
            params['max_context_length'] = 3000
        else:
            params['ollama_keep_alive'] = "1h"
            params['max_context_length'] = 2000

        return params

    def _log_hardware_info(self, hw: HardwareInfo):
        """Log detected hardware information"""
        logger.info("🖥️  Hardware Detection Results:")
        logger.info(f"   CPU: {hw.cpu_brand}")
        logger.info(f"   Cores: {hw.cpu_count} @ {hw.cpu_freq_max:.1f}GHz")
        logger.info(f"   RAM: {hw.total_ram_gb:.1f}GB total, {hw.available_ram_gb:.1f}GB available")
        logger.info(f"   GPU: {hw.gpu_name}")
        if hw.has_gpu:
            logger.info(f"   VRAM: {hw.gpu_vram_gb:.1f}GB")
            logger.info(f"   Compute: {hw.gpu_compute_capability}")
        logger.info(f"   Storage: {hw.storage_type}")
        logger.info(f"   Platform: {hw.platform} (Python {hw.python_version})")
        logger.info(f"   Performance Tier: {hw.performance_tier}")

    def get_recommended_params(self) -> Dict[str, Any]:
        """Get recommended parameters for RAG system"""
        return self.optimal_params.copy()

    def get_hardware_summary(self) -> str:
        """Get a human-readable hardware summary"""
        hw = self.hardware_info
        gpu_info = f"{hw.gpu_name} ({hw.gpu_vram_gb:.1f}GB)" if hw.has_gpu else "CPU Only"

        return f"""Hardware Summary:
• CPU: {hw.cpu_count} cores @ {hw.cpu_freq_max:.1f}GHz
• RAM: {hw.total_ram_gb:.1f}GB ({hw.available_ram_gb:.1f}GB available)
• GPU: {gpu_info}
• Performance Tier: {hw.performance_tier}
• Optimizations: {len(self.optimal_params)} parameters tuned"""


class SafetyKeywordDetector:
    """Detects and scores safety-critical content in electrical utility documents"""

    def __init__(self):
        self.safety_keywords = self._load_safety_keywords()
        self.equipment_keywords = self._load_equipment_keywords()
        self.voltage_patterns = self._compile_voltage_patterns()

    def _load_safety_keywords(self) -> Dict[str, List[str]]:
        """Load safety keywords organized by category"""
        return {
        'loto': [
            'lockout', 'tagout', 'loto', 'energy isolation', 'verification',
            'lock out', 'tag out', 'energy control', 'stored energy',
            'zero energy', 'isolation verification', 'energy source',
            'authorized employee', 'affected employee', 'qualified employee',
            'isolation device', 'lockout device', 'tagout device',
            'shutdown procedure', 'startup procedure', 'de-energize'
        ],
        
        'arc_flash': [
            'arc flash', 'arc fault', 'cal/cm²', 'cal/cm2', 'calories per square centimeter',
            'incident energy', 'flash protection', 'arc protection', 'flash hazard',
            'arc rated', 'arc flash boundary', 'flash suit', 'arc blast',
            'atpv', 'arc thermal performance value', 'hrc', 'hazard risk category',
            'flash protection boundary', 'limited approach', 'restricted approach'
        ],
        
        # HIGH PRIORITY SAFETY (Weight: 2.5)
        'electrical_safety': [
            'energized', 'de-energized', 'live', 'dead', 'hot work',
            'electrical hazard', 'shock hazard', 'electrical contact',
            'safe work practices', 'electrical safety', 'qualified person',
            'unqualified person', 'approach boundary', 'limited approach',
            'restricted approach', 'prohibited approach', 'minimum approach distance',
            'mad', 'voltage verification', 'voltage detector', 'lockout verification'
        ],
        
        'fall_protection': [
            'fall protection', 'fall arrest', 'harness', 'lanyard', 'anchor point',
            'fall distance', 'clearance', 'free fall', 'deceleration distance',
            'height', 'fall hazard', 'guardrail', 'safety line', 'total fall distance',
            'minimum clearance', 'bucket truck', 'lattice tower', 'aerial device',
            'pfas', 'personal fall arrest system', 'positioning system',
            'safety factor', 'anchor strength', 'fall clearance'
        ],
        
        'confined_space': [
            'confined space', 'permit required space', 'non-permit space',
            'atmospheric testing', 'ventilation', 'oxygen', 'toxic gas',
            'explosive atmosphere', 'engulfment', 'entrant', 'attendant',
            'entry permit', 'rescue', 'atmospheric monitoring', 'air quality',
            'lel', 'lower explosive limit', 'ppm', 'parts per million',
            'atmospheric hazard', 'space classification'
        ],
        
        # IMPORTANT SAFETY (Weight: 2.0)
        'ppe': [
            'personal protective equipment', 'ppe', 'safety equipment',
            'protective clothing', 'insulating gloves', 'safety glasses',
            'hard hat', 'safety helmet', 'face shield', 'hearing protection',
            'dielectric', 'insulated tools', 'rubber gloves', 'voltage rated',
            'flame resistant', 'fr clothing', 'arc rated clothing',
            'rubber insulating', 'electrical protective equipment'
        ],
        
        'grounding_bonding': [
            'grounding', 'bonding', 'ground', 'bond', 'equipotential',
            'ground fault', 'grounding electrode', 'bonding jumper',
            'equipment grounding', 'system grounding', 'safety ground',
            'temporary grounding', 'grounding cluster', 'bonding conductor',
            'ground verification', 'earth ground', 'ground resistance'
        ],
        
        'working_alone': [
            'working alone', 'solo work', 'lone worker', 'individual work',
            'unaccompanied work', 'single worker', 'isolated work',
            'check-in', 'monitoring', 'communication protocol',
            'emergency contact', 'buddy system', 'supervision',
            'man down', 'panic button', 'gps tracking', 'safety watch'
        ],
        
        'weather_safety': [
            'extreme weather', 'temperature', 'wind speed', 'lightning',
            'heat stress', 'cold stress', 'wind chill', 'hypothermia',
            'heat exhaustion', 'precipitation', 'visibility', 'storm',
            'work stoppage', 'weather monitoring', 'temperature threshold',
            'wind threshold', 'weather alert', 'chinook', 'whiteout'
        ],
        
        # MODERATE SAFETY (Weight: 1.5)
        'procedures': [
            'safety procedure', 'work procedure', 'operating procedure',
            'emergency procedure', 'shutdown procedure', 'startup procedure',
            'maintenance procedure', 'inspection procedure', 'testing procedure',
            'safety protocol', 'work instruction', 'safety requirement',
            'compliance', 'regulation', 'standard', 'guideline'
        ],
        
        'permits': [
            'permit', 'work permit', 'hot work permit', 'confined space permit',
            'electrical permit', 'eap', 'eip', 'lwp', 'tbp', 'gip',
            'equipment access permit', 'energy isolation permit', 'line work permit',
            'temporary bypass permit', 'generation interconnect permit',
            'permit system', 'authorization', 'clearance'
        ],
        
        'hazard_assessment': [
            'hazard assessment', 'risk assessment', 'job hazard analysis',
            'jha', 'safety assessment', 'hazard identification',
            'risk evaluation', 'hazard control', 'risk mitigation',
            'safety analysis', 'workplace hazard', 'occupational hazard'
        ],
        
        # EQUIPMENT & TESTING (Weight: 1.0)
        'electrical_equipment': [
            'switchgear', 'transformer', 'breaker', 'circuit breaker', 'disconnect',
            'switch', 'motor control center', 'mcc', 'panel', 'electrical panel',
            'distribution panel', 'load center', 'busway', 'bus duct',
            'cable', 'conductor', 'wire', 'conduit', 'junction box',
            'substation', 'transmission line', 'distribution line'
        ],
        
        'testing_procedures': [
            'insulation resistance', 'contact resistance', 'timing test',
            'polarization index', 'power factor', 'doble test',
            'dissolved gas analysis', 'dga', 'turns ratio', 'ttr',
            'dielectric test', 'hipot', 'megger', 'testing equipment',
            'calibration', 'test procedure', 'acceptance criteria'
        ],
        
        'emergency_response': [
            'emergency', 'rescue', 'evacuation', 'first aid',
            'emergency response', 'incident response', 'accident',
            'injury', 'emergency contact', 'emergency procedure',
            'emergency equipment', 'alarm', 'notification',
            'emergency shutdown', 'emergency isolation'
        ],
        
        # OPERATIONAL TERMS (Weight: 0.5)
        'operations': [
            'switching', 'operation', 'maintenance', 'inspection',
            'commissioning', 'testing', 'energizing', 'de-energizing',
            'isolation', 'restoration', 'outage', 'planned work',
            'emergency work', 'routine maintenance'
        ]
              
        }

    def _load_equipment_keywords(self) -> List[str]:
        """Load electrical equipment keywords"""
        return [
            'switchgear', 'transformer', 'breaker', 'disconnect', 'switch',
            'motor control center', 'mcc', 'panel', 'electrical panel',
            'distribution panel', 'load center', 'busway', 'bus duct',
            'cable', 'conductor', 'wire', 'conduit', 'junction box',
            'substation', 'transmission line', 'distribution line',
            'generator', 'motor', 'drive', 'vfd', 'variable frequency drive',
            'relay', 'protective relay', 'ct', 'current transformer',
            'pt', 'potential transformer', 'meter', 'monitoring equipment'
        ]

    def _compile_voltage_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for voltage detection"""
        patterns = [
            re.compile(r'\b\d+\.?\d*\s*k?v\b', re.IGNORECASE),  # 480V, 4.16kV, etc.
            re.compile(r'\b\d+\.?\d*\s*volts?\b', re.IGNORECASE),  # 120 volts
            re.compile(r'\b\d+\.?\d*\s*kilo\s*volts?\b', re.IGNORECASE),  # kilo volts
            re.compile(r'\bhigh\s*voltage\b', re.IGNORECASE),
            re.compile(r'\blow\s*voltage\b', re.IGNORECASE),
            re.compile(r'\bmedium\s*voltage\b', re.IGNORECASE),
        ]
        return patterns

    def analyze_chunk(self, text: str) -> Dict[str, Any]:
        """Analyze a text chunk for safety-critical content"""
        text_lower = text.lower()

        # Initialize scores
        category_scores = {}
        found_keywords = set()

        # Score by safety category
        total_keywords = 0
        for category, keywords in self.safety_keywords.items():
            category_count = 0
            for keyword in keywords:
                if keyword in text_lower:
                    category_count += 1
                    found_keywords.add(keyword)

            category_scores[category] = category_count
            total_keywords += category_count

        # Check for equipment mentions
        equipment_count = 0
        for equipment in self.equipment_keywords:
            if equipment in text_lower:
                equipment_count += 1
                found_keywords.add(equipment)

        # Check for voltage patterns
        voltage_count = 0
        for pattern in self.voltage_patterns:
            matches = pattern.findall(text)
            voltage_count += len(matches)
            for match in matches:
                found_keywords.add(match)

        # Calculate overall safety score
        safety_score = self._calculate_safety_score(
            category_scores, equipment_count, voltage_count, len(text)
        )

        # Determine criticality level
        criticality = self._determine_criticality(safety_score, category_scores)

        return {
            'safety_score': safety_score,
            'safety_criticality': criticality,
            'category_scores': category_scores,
            'equipment_count': equipment_count,
            'voltage_count': voltage_count,
            'keywords_found': list(found_keywords),
            'total_safety_keywords': total_keywords
        }

    def _calculate_safety_score(self, category_scores: Dict[str, int],
                                equipment_count: int, voltage_count: int,
                                text_length: int) -> float:
        """Calculate weighted safety score with comprehensive categories"""
        
        weights = {
            # Critical Safety Procedures
            'loto': 3.0,
            'arc_flash': 3.0,
            
            # High Priority Safety
            'electrical_safety': 2.5,
            'fall_protection': 2.5,
            'confined_space': 2.5,
            
            # Important Safety
            'ppe': 2.0,
            'grounding_bonding': 2.0,
            'working_alone': 2.0,
            'weather_safety': 2.0,
            
            # Moderate Safety
            'procedures': 1.5,
            'permits': 1.5,
            'hazard_assessment': 1.5,
            
            # Equipment & Testing
            'electrical_equipment': 1.0,
            'testing_procedures': 1.0,
            'emergency_response': 1.0,
            
            # Operational
            'operations': 0.5
        }
        
        weighted_score = 0
        for category, count in category_scores.items():
            weighted_score += count * weights.get(category, 1.0)
        
        # Add equipment and voltage mentions
        weighted_score += equipment_count * 0.5
        weighted_score += voltage_count * 1.0
        
        # Normalize by text length (keywords per 100 characters)
        if text_length > 0:
            normalized_score = (weighted_score / text_length) * 100
        else:
            normalized_score = 0
        
        return min(normalized_score, 10.0)  # Cap at 10.0



    def _determine_criticality(self, safety_score: float,
                               category_scores: Dict[str, int]) -> str:
        """Determine safety criticality level"""
        # High criticality: Strong LOTO or arc flash content
        if (category_scores.get('loto', 0) >= 2 or
                category_scores.get('arc_flash', 0) >= 2 or
                safety_score >= 3.0):
            return 'HIGH'

        # Medium criticality: Some safety content
        elif (sum(category_scores.values()) >= 2 or
              safety_score >= 1.0):
            return 'MEDIUM'

        # Low criticality: Minimal safety content
        else:
            return 'LOW'


class RecursiveTextSplitter:
    """Advanced text splitter that preserves sentence boundaries and maintains overlap"""

    def __init__(self, chunk_size: int = 128, chunk_overlap: int = 16,
                 min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')

    def split_text(self, text: str, source_file: str = "unknown") -> List['DocumentChunk']:
        """Split text into overlapping chunks while preserving sentence boundaries"""

        # First, split by paragraphs
        paragraphs = self.paragraph_breaks.split(text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Debug logging
        logger.debug(f"Splitting {source_file}: {len(text)} chars, {len(paragraphs)} paragraphs")

        # Handle single large paragraph by splitting on sentences
        if len(paragraphs) == 1 and self._count_tokens(paragraphs[0]) > self.chunk_size * 2:
            logger.debug(f"  Large single paragraph detected, splitting by sentences")
            return self._split_by_sentences(text, source_file)

        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_position = 0

        for i, paragraph in enumerate(paragraphs):
            # Check if adding this paragraph exceeds chunk size
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            potential_tokens = self._count_tokens(potential_chunk)
            current_tokens = self._count_tokens(current_chunk)

            logger.debug(
                f"  Paragraph {i + 1}: {len(paragraph)} chars, potential_tokens: {potential_tokens}, current_tokens: {current_tokens}")

            # Fixed logic: Create chunk if we exceed chunk size AND have meaningful content
            if potential_tokens > self.chunk_size and current_tokens >= self.min_chunk_size:
                # Create chunk from current content
                chunk = self._create_chunk(current_chunk, source_file, chunk_index, char_position)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                    logger.debug(
                        f"    Created chunk {chunk_index}: {len(current_chunk)} chars, {current_tokens} tokens")

                # Calculate overlap for next chunk
                overlap_text = self._get_overlap_text(current_chunk)
                char_position += len(current_chunk) - len(overlap_text)

                # Start new chunk with overlap + current paragraph
                if overlap_text:
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                current_chunk = potential_chunk

        # Handle final chunk - ALWAYS create if we have content above minimum
        if current_chunk.strip() and self._count_tokens(current_chunk) >= self.min_chunk_size:
            chunk = self._create_chunk(current_chunk, source_file, chunk_index, char_position)
            if chunk:
                chunks.append(chunk)
                logger.debug(
                    f"    Created final chunk {chunk_index + 1}: {len(current_chunk)} chars, {self._count_tokens(current_chunk)} tokens")

        logger.debug(f"Total chunks created for {source_file}: {len(chunks)}")
        return chunks

    def _split_by_sentences(self, text: str, source_file: str) -> List['DocumentChunk']:
        """Fallback method to split large single paragraphs by sentences"""

        # Split on sentence boundaries
        sentences = self.sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        logger.debug(f"  Splitting by sentences: {len(sentences)} sentences found")

        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_position = 0

        for i, sentence in enumerate(sentences):
            # Reconstruct sentence with proper ending
            if i < len(sentences) - 1:
                sentence = sentence + ". "  # Add period back

            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            potential_tokens = self._count_tokens(potential_chunk)
            current_tokens = self._count_tokens(current_chunk)

            if potential_tokens > self.chunk_size and current_tokens >= self.min_chunk_size:
                # Create chunk
                chunk = self._create_chunk(current_chunk, source_file, chunk_index, char_position)
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1
                    logger.debug(f"    Created sentence-based chunk {chunk_index}: {current_tokens} tokens")

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                char_position += len(current_chunk) - len(overlap_text)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                current_chunk = potential_chunk

        # Final chunk
        if current_chunk.strip() and self._count_tokens(current_chunk) >= self.min_chunk_size:
            chunk = self._create_chunk(current_chunk, source_file, chunk_index, char_position)
            if chunk:
                chunks.append(chunk)
                logger.debug(f"    Created final sentence-based chunk: {self._count_tokens(current_chunk)} tokens")

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return max(1, len(text.strip()) // 4)  # Ensure at least 1 token for non-empty text

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if not text:
            return ""

        target_overlap_chars = self.chunk_overlap * 4  # Convert tokens to chars

        if len(text) <= target_overlap_chars:
            return text

        # Try to break at sentence boundary within overlap region
        overlap_start = len(text) - target_overlap_chars
        text_section = text[overlap_start:]

        # Find the first sentence ending in the overlap section
        sentence_match = self.sentence_endings.search(text_section)
        if sentence_match:
            # Start overlap from after the sentence ending
            overlap_start += sentence_match.end()
            return text[overlap_start:]
        else:
            # No sentence boundary found, use word boundary
            words = text_section.split()
            if len(words) > 3:
                # Skip first few words to find a good breaking point
                return ' '.join(words[2:])
            else:
                return text_section

    def _create_chunk(self, content: str, source_file: str, chunk_index: int,
                      char_start: int) -> Optional['DocumentChunk']:
        """Create a DocumentChunk object"""
        content = content.strip()
        if len(content) < self.min_chunk_size:
            return None

        chunk_id = f"{source_file}_{chunk_index:03d}"

        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            source_file=source_file,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_start + len(content),
            embedding=None
        )


@dataclass
class DocumentChunk:
    """Enhanced data class for document chunks with safety metadata"""
    chunk_id: str
    content: str
    source_file: str
    chunk_index: int
    char_start: int
    char_end: int
    embedding: Optional[np.ndarray] = None

    # Enhanced safety metadata
    safety_score: float = 0.0
    safety_criticality: str = 'LOW'  # LOW, MEDIUM, HIGH
    category_scores: Dict[str, int] = field(default_factory=dict)
    equipment_count: int = 0
    voltage_count: int = 0
    keywords_found: List[str] = field(default_factory=list)

    enhanced_content: Optional[str] = None  # Content with file context for embedding
    file_metadata: Optional[Dict[str, str]] = None  # File information

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentChunk':
        """Create from dictionary"""
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


@dataclass
class RAGConfig:
    """Enhanced configuration class with chunking and safety parameters"""
    # Enhanced document processing parameters
    chunk_size: int = 128  # Primary for safety procedures
    chunk_size_alt: int = 256  # For equipment specifications
    chunk_overlap: int = 16  # For 128-token chunks (12.5% overlap)
    chunk_overlap_alt: int = 25  # For 256-token chunks (10% overlap)
    min_chunk_size: int = 50

    # Safety detection parameters
    enable_safety_detection: bool = True
    safety_boost_factor: float = 1.2  # Multiply similarity score for safety-critical chunks
    min_safety_score: float = 0.5  # Minimum score to be considered safety-relevant

    # Embedding (optimized model selection)
    embedding_model: str = 'BAAI/bge-small-en-v1.5'  # Finetuned Model of bge-small-en-v1.5 masonlf/bge-small-elec-finetune
    embedding_batch_size: int = 32  # Will be hardware-optimized
    lazy_model_loading: bool = False
    warm_models_on_startup: bool = True

    # Performance optimizations (hardware-adaptive)
    use_model_cache: bool = True
    parallel_init: bool = True
    optimize_data_loading: bool = True
    max_workers: int = 4  # Will be hardware-optimized

    # Enhanced retrieval parameters
    similarity_top_k: int = 10  # Increased for Llama 3.2 3B context window
    similarity_threshold: float = 0.25  # Pre-filter threshold (relaxed)
    final_top_k: int = 5  # Final number of chunks to return
    enable_safety_boosting: bool = True

    # LLM (keep model same, optimize parameters)
    ollama_model: str = 'Llama-3.2-3B-Finetune-q4k.gguf'  # Same model as requested
    ollama_host: str = 'http://localhost:11434'
    max_context_length: int = 4000  # Will be hardware-optimized
    temperature: float = 0.1
    warm_ollama_on_startup: bool = True
    ollama_keep_alive: str = "24h"  # Will be hardware-optimized

    # Caching (hardware-adaptive)
    enable_cache: bool = True
    cache_dir: str = '.venv/cache'
    cache_ttl_hours: int = 24  # Will be hardware-optimized
    max_cache_size_mb: int = 500  # Will be hardware-optimized
    semantic_cache_threshold: float = 0.85

    # Add these new parameters with defaults (add to existing RAGConfig)
    eval_test_size: int = 45  # Increased from 20
    final_top_k: int = 5  # New parameter
    enable_safety_boosting: bool = True  # New parameter  
    safety_boost_factor: float = 1.2  # New parameter
    min_safety_score: float = 0.5  # New parameter
    min_chunks_success_rate: float = 0.8  # New parameter
    min_safety_prioritization: float = 0.7  # New parameter

    def ensure_new_parameters(self):
        """Ensure all new parameters exist with default values (for runtime compatibility)"""
        defaults = {
            'min_relevance_at_k': 0.6,
            'min_coverage_at_k': 0.8,
            'min_keyword_coverage': 0.5,
            'max_response_time': 2.0,
            'chunk_size_alt': getattr(self, 'chunk_size_alt', 256),
            'chunk_overlap_alt': getattr(self, 'chunk_overlap_alt', 25),
            'final_top_k': getattr(self, 'final_top_k', 5),
            'enable_safety_boosting': getattr(self, 'enable_safety_boosting', True),
            'safety_boost_factor': getattr(self, 'safety_boost_factor', 1.2),
            'min_safety_score': getattr(self, 'min_safety_score', 0.5),
            
            # ADD THESE NEW FILE-AWARE PARAMETERS:
            'enable_file_context': getattr(self, 'enable_file_context', True),
            'file_context_strategy': getattr(self, 'file_context_strategy', 'prepend'),
            'max_file_context_length': getattr(self, 'max_file_context_length', 100),
            'include_file_topics': getattr(self, 'include_file_topics', True),
            'include_file_category': getattr(self, 'include_file_category', True),
        }

        for param, default_value in defaults.items():
            if not hasattr(self, param):
                setattr(self, param, default_value)
                logger.info(f"Set missing parameter '{param}' to default: {default_value}")

                
    # Evaluation parameters
    enable_evaluation: bool = True
    eval_test_size: int = 20  # Number of test queries for evaluation

    # Quality thresholds for pass/fail evaluation
    min_relevance_at_k: float = 0.6  # Minimum acceptable relevance@K
    min_coverage_at_k: float = 0.8  # Minimum acceptable coverage@K
    min_keyword_coverage: float = 0.5  # Minimum acceptable keyword coverage
    max_response_time: float = 2.0  # Maximum acceptable response time (seconds)
    
    enable_file_context: bool = True
    file_context_strategy: str = "prepend"  # "prepend", "append", "tags"
    max_file_context_length: int = 100  # Max characters for file context
    include_file_topics: bool = True
    include_file_category: bool = True
    
    # Directories
    docs_dir: str = '.venv/documents'
    data_dir: str = '.venv/data'

    # Hardware info (added)
    hardware_optimized: bool = False
    hardware_tier: str = "UNKNOWN"

    def apply_hardware_optimizations(self, hardware_detector: HardwareDetector):
        """Apply hardware-specific optimizations"""
        params = hardware_detector.get_recommended_params()

        # Apply each parameter
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Mark as hardware optimized
        self.hardware_optimized = True
        self.hardware_tier = hardware_detector.hardware_info.performance_tier

        logger.info(f"✅ Applied {len(params)} hardware optimizations for {self.hardware_tier} tier")

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'RAGConfig':
        """Load configuration from JSON file with backward compatibility"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Handle backward compatibility for parameter name changes
        compatibility_mapping = {
            'top_k_chunks': 'similarity_top_k',  # Old -> New parameter name
            'target_precision': 'min_relevance_at_k',  # Map old evaluation params
            'target_recall': 'min_coverage_at_k',
        }

        # Convert old parameter names to new ones
        for old_param, new_param in compatibility_mapping.items():
            if old_param in data and old_param != new_param:
                if new_param not in data:  # Only map if new param doesn't already exist
                    data[new_param] = data[old_param]
                del data[old_param]

        # Add missing new parameters with defaults
        default_values = {
            # New evaluation parameters
            'min_relevance_at_k': 0.6,
            'min_coverage_at_k': 0.8,
            'min_keyword_coverage': 0.5,
            'max_response_time': 2.0,

            # Other potentially missing parameters
            'chunk_size_alt': 256,
            'chunk_overlap_alt': 25,
            'final_top_k': 5,
            'enable_safety_boosting': True,
            'safety_boost_factor': 1.2,
            'min_safety_score': 0.5,
        }

        # Add any missing parameters
        for param, default_value in default_values.items():
            if param not in data:
                data[param] = default_value
                logger.info(f"Added missing config parameter '{param}' with default value: {default_value}")

        # Remove any unknown parameters that might cause issues
        known_params = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_params}

        # Log if we filtered out any parameters
        filtered_out = set(data.keys()) - set(filtered_data.keys())
        if filtered_out:
            logger.info(f"Filtered out unknown config parameters: {filtered_out}")

        return cls(**filtered_data)

    def ensure_new_parameters(self):
        """Ensure all new parameters exist with default values (for runtime compatibility)"""
        defaults = {
            'min_relevance_at_k': 0.6,
            'min_coverage_at_k': 0.8,
            'min_keyword_coverage': 0.5,
            'max_response_time': 2.0,
            'chunk_size_alt': getattr(self, 'chunk_size_alt', 256),
            'chunk_overlap_alt': getattr(self, 'chunk_overlap_alt', 25),
            'final_top_k': getattr(self, 'final_top_k', 5),
            'enable_safety_boosting': getattr(self, 'enable_safety_boosting', True),
            'safety_boost_factor': getattr(self, 'safety_boost_factor', 1.2),
            'min_safety_score': getattr(self, 'min_safety_score', 0.5),
        }

        for param, default_value in defaults.items():
            if not hasattr(self, param):
                setattr(self, param, default_value)
                logger.info(f"Set missing parameter '{param}' to default: {default_value}")


class FileAwareChunker:
    """Extracts file context and enhances chunks with file information"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
    def extract_file_metadata(self, filename: str) -> Dict[str, str]:
        """Extract meaningful metadata from filename"""
        
        # Remove file extension
        base_name = filename.replace('.txt', '').replace('.md', '')
        
        # File metadata extraction
        file_metadata = {
            'filename': filename,
            'title': self._clean_filename_to_title(base_name),
            'category': self._detect_file_category(base_name),
            'topics': self._extract_topics_from_filename(base_name),
            'document_type': self._detect_document_type(base_name)
        }
        
        return file_metadata
    
    def _clean_filename_to_title(self, filename: str) -> str:
        """Convert filename to readable title"""
        # Replace common separators with spaces
        title = filename.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Remove version numbers, dates, etc.
        title = re.sub(r'\b\d{4}\b', '', title)  # Remove years
        title = re.sub(r'\bv\d+(\.\d+)?\b', '', title, re.IGNORECASE)  # Remove version numbers
        
        # Clean up spacing and capitalize
        title = re.sub(r'\s+', ' ', title).strip().title()
        
        return title
    
    def _detect_file_category(self, filename: str) -> str:
        """Detect file category from filename"""
        filename_lower = filename.lower()
        
        categories = {
            'safety': ['safety', 'safe', 'hazard', 'risk', 'protection', 'ppe', 'emergency'],
            'procedures': ['procedure', 'protocol', 'process', 'instruction', 'guide', 'manual'],
            'standards': ['standard', 'spec', 'specification', 'requirement', 'code', 'regulation'],
            'testing': ['test', 'testing', 'inspection', 'maintenance', 'commissioning'],
            'training': ['training', 'education', 'course', 'certification', 'competency'],
            'loto': ['loto', 'lockout', 'tagout', 'isolation', 'energy control'],
            'arc_flash': ['arc', 'flash', 'incident energy', 'cal/cm'],
            'working_alone': ['alone', 'solo', 'individual', 'lone worker'],
            'confined_space': ['confined', 'space', 'entry', 'permit space'],
            'fall_protection': ['fall', 'height', 'protection', 'harness', 'anchor'],
            'electrical': ['electrical', 'electric', 'power', 'voltage', 'current', 'energy']
        }
        
        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_topics_from_filename(self, filename: str) -> str:
        """Extract topic keywords from filename as comma-separated string"""
        filename_lower = filename.lower()
        
        topics = []
        topic_keywords = {
            'transformer': ['transformer', 'xfmr'],
            'switchgear': ['switchgear', 'switch gear', 'switch'],
            'transmission': ['transmission', 'trans', 'line'],
            'distribution': ['distribution', 'dist'],
            'substation': ['substation', 'sub station'],
            'voltage': ['voltage', 'kv', 'volt'],
            'grounding': ['ground', 'grounding', 'earth'],
            'protection': ['protection', 'protective', 'relay'],
            'maintenance': ['maintenance', 'maint', 'repair'],
            'safety': ['safety', 'safe'],
            'arc_flash': ['arc', 'flash'],
            'loto': ['loto', 'lockout', 'tagout'],
            'ppe': ['ppe', 'personal protective'],
            'working_alone': ['working alone', 'alone', 'solo'],
            'confined_space': ['confined space', 'confined'],
            'fall_protection': ['fall protection', 'fall', 'height']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                topics.append(topic.replace('_', ' '))
        
        return ', '.join(topics[:3])  # Limit to top 3 topics
    
    def _detect_document_type(self, filename: str) -> str:
        """Detect document type from filename"""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['procedure', 'protocol', 'instruction']):
            return 'procedure'
        elif any(word in filename_lower for word in ['standard', 'specification', 'requirement']):
            return 'standard'
        elif any(word in filename_lower for word in ['manual', 'guide', 'handbook']):
            return 'manual'
        elif any(word in filename_lower for word in ['policy', 'regulation', 'rule']):
            return 'policy'
        else:
            return 'document'
    
    def create_file_aware_content(self, chunk_content: str, file_metadata: Dict[str, str]) -> str:
        """Create enhanced content that includes file context"""
        
        if not self.config.enable_file_context:
            return chunk_content
            
        # Build file context parts
        file_context_parts = []
        
        # Add document title
        if file_metadata['title']:
            file_context_parts.append(f"Document: {file_metadata['title']}")
        
        # Add category if meaningful
        if file_metadata['category'] != 'general':
            file_context_parts.append(f"Category: {file_metadata['category']}")
        
        # Add key topics
        if file_metadata['topics']:
            file_context_parts.append(f"Topics: {file_metadata['topics']}")
        
        # Add document type if meaningful
        if file_metadata['document_type'] != 'document':
            file_context_parts.append(f"Type: {file_metadata['document_type']}")
        
        # Combine file context
        if file_context_parts:
            file_context = ' | '.join(file_context_parts)
            
            # Limit context length
            if len(file_context) > self.config.max_file_context_length:
                file_context = file_context[:self.config.max_file_context_length] + "..."
            
            # Apply strategy
            if self.config.file_context_strategy == "prepend":
                enhanced_content = f"[{file_context}]\n\n{chunk_content}"
            elif self.config.file_context_strategy == "append":
                enhanced_content = f"{chunk_content}\n\n[{file_context}]"
            else:  # tags
                enhanced_content = f"#{file_metadata['category']} {chunk_content}"
        else:
            enhanced_content = chunk_content
        
        return enhanced_content


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0

    def is_expired(self, ttl_hours: int) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > (ttl_hours * 3600)

    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


class AdaptiveRAGCache:
    """Cache system that adapts to hardware capabilities"""

    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        self.config = config
        self.hardware_info = hardware_detector.hardware_info
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory caches
        self._query_cache: Dict[str, CacheEntry] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'semantic_hits': 0,
            'embeddings_cached': 0,
            'total_queries': 0,
            'memory_optimized': True
        }

        # Hardware-adaptive cache loading
        self._cache_loaded = False
        if config.parallel_init and self.hardware_info.cpu_count > 2:
            self._load_thread = threading.Thread(target=self._load_persistent_cache_async)
            self._load_thread.start()
        else:
            self._load_persistent_cache()
            self._cache_loaded = True

    def _load_persistent_cache_async(self):
        """Load cache asynchronously based on hardware capability"""
        try:
            self._load_persistent_cache()
            self._cache_loaded = True
            logger.debug("Cache loaded asynchronously")
        except Exception as e:
            logger.warning(f"Async cache loading failed: {e}")
            self._cache_loaded = True

    def _wait_for_cache(self):
        """Wait for cache loading to complete"""
        if hasattr(self, '_load_thread'):
            self._load_thread.join()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for better cache hits"""
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        # Expand common acronyms
        normalized = normalized.replace('loto', 'lockout tagout')
        normalized = normalized.replace('ppe', 'personal protective equipment')
        return normalized

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query"""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _load_persistent_cache(self):
        """Load cache from disk with memory optimization"""
        cache_file = self.cache_dir / "query_cache.pkl"
        if cache_file.exists():
            try:
                # Check file size before loading on low-memory systems
                file_size_mb = cache_file.stat().st_size / (1024 * 1024)

                if (self.hardware_info.available_ram_gb < 4 and file_size_mb > 100):
                    logger.warning(f"Skipping large cache file ({file_size_mb:.1f}MB) on low-memory system")
                    return

                with open(cache_file, 'rb') as f:
                    self._query_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._query_cache)} cached queries")
                self._cleanup_expired()
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self._query_cache = {}

    def _cleanup_expired(self):
        """Remove expired cache entries with memory consideration"""
        expired_keys = [
            key for key, entry in self._query_cache.items()
            if entry.is_expired(self.config.cache_ttl_hours)
        ]

        # On low memory systems, be more aggressive with cleanup
        if self.hardware_info.available_ram_gb < 4:
            # Also remove least recently used entries if cache is large
            if len(self._query_cache) > 100:
                sorted_entries = sorted(
                    self._query_cache.items(),
                    key=lambda x: x[1].last_accessed
                )
                # Remove oldest 25% of entries
                remove_count = len(sorted_entries) // 4
                for key, _ in sorted_entries[:remove_count]:
                    expired_keys.append(key)

        for key in expired_keys:
            del self._query_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} cache entries")

    def get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result with hardware-adaptive loading"""
        if not self.config.enable_cache:
            return None

        self._wait_for_cache()
        query_hash = self._get_query_hash(query)

        if query_hash in self._query_cache:
            entry = self._query_cache[query_hash]
            if not entry.is_expired(self.config.cache_ttl_hours):
                entry.touch()
                self.stats['hits'] += 1
                logger.info("⚡ Cache hit!")
                return entry.value
            else:
                del self._query_cache[query_hash]

        return None

    def cache_result(self, query: str, result: Dict[str, Any], query_embedding: Optional[np.ndarray] = None):
        """Cache query result with memory management"""
        if not self.config.enable_cache:
            return

        self._wait_for_cache()

        # Check memory usage before caching on low-memory systems
        if self.hardware_info.available_ram_gb < 4:
            current_cache_size = len(self._query_cache)
            if current_cache_size > 200:  # Limit cache size on low memory
                # Remove oldest entry
                oldest_key = min(self._query_cache.keys(),
                                 key=lambda k: self._query_cache[k].last_accessed)
                del self._query_cache[oldest_key]

        query_hash = self._get_query_hash(query)

        try:
            size_bytes = len(pickle.dumps(result))
        except:
            size_bytes = len(str(result).encode())

        entry = CacheEntry(
            key=query_hash,
            value=result,
            timestamp=time.time(),
            size_bytes=size_bytes
        )
        entry.touch()

        self._query_cache[query_hash] = entry

        if query_embedding is not None and self.hardware_info.available_ram_gb >= 2:
            self._embedding_cache[query_hash] = query_embedding
            self.stats['embeddings_cached'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with hardware info"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self.stats,
            'hit_rate_percent': hit_rate,
            'cached_queries': len(self._query_cache),
            'cached_embeddings': len(self._embedding_cache),
            'hardware_tier': self.hardware_info.performance_tier,
            'available_ram_gb': self.hardware_info.available_ram_gb
        }

    def clear_cache(self):
        """Clear all caches"""
        self._query_cache.clear()
        self._embedding_cache.clear()


class HardwareAdaptiveEmbeddingGenerator:
    """Embedding generator that adapts to hardware capabilities"""

    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        self.config = config
        self.hardware_info = hardware_detector.hardware_info
        self.model = None
        self._model_loading = False
        self._model_lock = threading.Lock()
        self.device = self._get_optimal_device()
        self.is_warmed = False

        # Always preload model for fast TTFT
        self._load_model()
        if config.warm_models_on_startup:
            self._warm_model()

    def _get_optimal_device(self):
        """Determine optimal device based on hardware"""
        if self.hardware_info.has_gpu and self.hardware_info.gpu_vram_gb >= 2:
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda:0'
                    logger.info(f"🚀 Using GPU: {self.hardware_info.gpu_name} ({self.hardware_info.gpu_vram_gb:.1f}GB)")

                    # Test GPU accessibility
                    test_tensor = torch.randn(10, 10).to(device)
                    del test_tensor
                    torch.cuda.empty_cache()
                    return device
                else:
                    logger.info("GPU detected but CUDA not available, using CPU")
                    return 'cpu'
            except Exception as e:
                logger.warning(f"GPU test failed: {e}, using CPU")
                return 'cpu'
        else:
            logger.info("Using CPU for embeddings")
            return 'cpu'

    def _load_model(self):
        """Load model with hardware-optimized settings"""
        with self._model_lock:
            if self.model is not None:
                return

            if self._model_loading:
                while self._model_loading:
                    time.sleep(0.1)
                return

            self._model_loading = True

            try:
                logger.info(f"📦 Loading {self.config.embedding_model} on {self.device}...")
                start_time = time.time()

                # Hardware-adaptive cache settings
                cache_folder = None
                if self.config.use_model_cache and self.hardware_info.available_ram_gb >= 4:
                    cache_folder = Path.home() / ".cache" / "sentence_transformers"

                self.model = SentenceTransformer(
                    self.config.embedding_model,
                    cache_folder=cache_folder,
                    device=self.device
                )

                actual_device = str(self.model.device) if hasattr(self.model, 'device') else "unknown"
                load_time = time.time() - start_time

                # Show performance info
                vram_info = ""
                if self.device.startswith('cuda'):
                    try:
                        import torch
                        vram_used = torch.cuda.memory_allocated() / 1024 ** 2
                        vram_info = f" (VRAM: {vram_used:.1f}MB)"
                    except:
                        pass

                logger.info(f"✅ Model loaded in {load_time:.2f}s on {actual_device}{vram_info}")

            except Exception as e:
                logger.error(f"Failed to load model on {self.device}: {e}")

                # Fallback to CPU
                if self.device.startswith('cuda'):
                    logger.info("🔄 Trying CPU fallback...")
                    try:
                        self.device = 'cpu'
                        self.model = SentenceTransformer(
                            self.config.embedding_model,
                            device='cpu'
                        )
                        logger.info("✅ CPU fallback successful")
                    except Exception as e2:
                        logger.error(f"CPU fallback failed: {e2}")
                        raise
                else:
                    raise
            finally:
                self._model_loading = False

    def _warm_model(self):
        """Warm model with hardware-appropriate workload"""
        if self.model is None or self.is_warmed:
            return

        logger.info("🔥 Warming embedding model...")
        start_time = time.time()

        # Hardware-adaptive warm-up
        if self.hardware_info.performance_tier in ['HIGH', 'ULTRA']:
            warm_queries = [
                "lockout tagout procedure safety",
                "arc flash protection equipment requirements",
                "electrical safety grounding procedures",
                "personal protective equipment standards",
                "electrical switching operations protocol"
            ]
        else:
            # Lighter warm-up for lower-tier hardware
            warm_queries = [
                "electrical safety",
                "lockout tagout",
                "arc flash protection"
            ]

        try:
            for query in warm_queries:
                _ = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

            if self.device.startswith('cuda'):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass

            warm_time = time.time() - start_time
            self.is_warmed = True
            logger.info(f"🔥 Model warmed in {warm_time:.2f}s")

        except Exception as e:
            logger.warning(f"Model warming failed: {e}")

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Generate embeddings using enhanced content if available"""
        if not chunks:
            return np.array([])

        logger.info(f"🧮 Generating file-aware embeddings for {len(chunks)} chunks...")
        
        # Use enhanced content for embedding if available
        texts = []
        file_aware_count = 0
        
        for chunk in chunks:
            if hasattr(chunk, 'enhanced_content') and chunk.enhanced_content:
                texts.append(chunk.enhanced_content)
                file_aware_count += 1
            else:
                texts.append(chunk.content)
        
        if file_aware_count > 0:
            logger.info(f"📁 Using file-aware content for {file_aware_count}/{len(chunks)} chunks")

        try:
            start_time = time.time()

            # Use hardware-optimized batch size
            batch_size = self.config.embedding_batch_size

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(chunks) > 20,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            embed_time = time.time() - start_time
            throughput = len(chunks) / embed_time

            logger.info(f"✅ File-aware embeddings generated in {embed_time:.2f}s ({throughput:.1f} chunks/s)")
            return embeddings

        except Exception as e:
            logger.error(f"File-aware embedding generation failed: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        try:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise


class HardwareAdaptiveVectorStore:
    """Vector store with hardware-optimized operations and safety boosting"""

    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        self.config = config
        self.hardware_info = hardware_detector.hardware_info
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []
        self.is_loaded = False

    def load(self) -> bool:
        """Load vector store with hardware-optimized approach"""
        data_path = Path(self.config.data_dir)
        required_files = ['embeddings.npy', 'chunks.json', 'metadata.json']
        missing_files = [f for f in required_files if not (data_path / f).exists()]

        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
            return False

        try:
            start_time = time.time()

            # Hardware-adaptive loading strategy
            if (self.config.optimize_data_loading and
                    self.hardware_info.cpu_count > 2 and
                    self.hardware_info.available_ram_gb >= 4):

                # Parallel loading for capable hardware
                with ThreadPoolExecutor(max_workers=2) as executor:
                    embeddings_future = executor.submit(np.load, data_path / "embeddings.npy")
                    chunks_future = executor.submit(self._load_chunks, data_path / "chunks.json")

                    self.embeddings = embeddings_future.result()
                    self.chunks = chunks_future.result()
            else:
                # Sequential loading for limited hardware
                self.embeddings = np.load(data_path / "embeddings.npy")
                self.chunks = self._load_chunks(data_path / "chunks.json")

            self.is_loaded = True
            load_time = time.time() - start_time

            # Memory usage info
            embeddings_size_mb = self.embeddings.nbytes / (1024 * 1024)
            logger.info(f"✅ Vector store loaded in {load_time:.2f}s")
            logger.info(f"   📊 {len(self.chunks)} chunks, {embeddings_size_mb:.1f}MB embeddings")

            # Log safety statistics
            self._log_safety_statistics()

            return True

        except Exception as e:
            logger.error(f"Vector store loading failed: {e}")
            return False

    def _load_chunks(self, chunks_file: Path) -> List[DocumentChunk]:
        """Load chunks with memory optimization"""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            return [DocumentChunk.from_dict(chunk) for chunk in chunks_data]
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            return []

    def _log_safety_statistics(self):
        """Log safety statistics for loaded chunks"""
        if not self.chunks:
            return

        safety_stats = {
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0
        }

        total_safety_score = 0
        for chunk in self.chunks:
            criticality = getattr(chunk, 'safety_criticality', 'LOW')
            safety_stats[criticality] = safety_stats.get(criticality, 0) + 1
            total_safety_score += getattr(chunk, 'safety_score', 0)

        avg_safety_score = total_safety_score / len(self.chunks)

        logger.info(
            f"📊 Safety Analysis: HIGH={safety_stats['HIGH']}, MEDIUM={safety_stats['MEDIUM']}, LOW={safety_stats['LOW']}")
        logger.info(f"📊 Average Safety Score: {avg_safety_score:.2f}")

    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> List[Tuple[DocumentChunk, float]]:
        """Search with hardware-optimized similarity computation and safety boosting"""
        if not self.is_loaded:
            raise ValueError("Vector store not loaded")

        if top_k is None:
            top_k = self.config.similarity_top_k

        # Hardware-adaptive similarity computation
        if self.hardware_info.has_gpu and len(self.embeddings) > 1000:
            # Use GPU for large datasets if available
            try:
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                query_tensor = torch.from_numpy(query_embedding).to(device)
                embeddings_tensor = torch.from_numpy(self.embeddings).to(device)

                similarities = torch.mm(query_tensor.unsqueeze(0), embeddings_tensor.T).squeeze()
                similarities = similarities.cpu().numpy()

            except Exception as e:
                logger.warning(f"GPU similarity computation failed: {e}, using CPU")
                # Fallback to CPU
                similarities = np.dot(query_embedding, self.embeddings.T)
        else:
            # CPU computation
            similarities = np.dot(query_embedding, self.embeddings.T)

        # Apply safety boosting if enabled
        if self.config.enable_safety_boosting:
            similarities = self._apply_safety_boosting(similarities)

        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by threshold and return results
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= self.config.similarity_threshold:
                results.append((self.chunks[idx], float(score)))

        # Take final top_k after all filtering
        results = results[:self.config.final_top_k]

        return results

    def _apply_safety_boosting(self, similarities: np.ndarray) -> np.ndarray:
        """Apply safety boosting to similarity scores"""
        boosted_similarities = similarities.copy()

        for i, chunk in enumerate(self.chunks):
            criticality = getattr(chunk, 'safety_criticality', 'LOW')

            if criticality == 'HIGH':
                boosted_similarities[i] *= self.config.safety_boost_factor
            elif criticality == 'MEDIUM':
                boosted_similarities[i] *= (1 + (self.config.safety_boost_factor - 1) * 0.5)
            # LOW criticality gets no boost

        return boosted_similarities

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add chunks with memory monitoring"""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count mismatch")

        # ✅ ASSIGN INDIVIDUAL EMBEDDINGS TO CHUNKS
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        self.chunks = chunks
        self.embeddings = embeddings
        self.is_loaded = True
        
        # Log memory usage and safety statistics
        embeddings_size_mb = embeddings.nbytes / (1024 * 1024)
        logger.info(f"📊 Added {len(chunks)} chunks ({embeddings_size_mb:.1f}MB embeddings)")
        self._log_safety_statistics()

    def save(self):
        """Save with hardware considerations"""
        if not self.is_loaded:
            logger.warning("No data to save")
            return

        data_path = Path(self.config.data_dir)
        data_path.mkdir(exist_ok=True)

        start_time = time.time()

        # Save embeddings
        np.save(data_path / "embeddings.npy", self.embeddings)

        # Save chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(data_path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # Save metadata with hardware and safety info
        safety_stats = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        total_safety_score = 0
        for chunk in self.chunks:
            criticality = getattr(chunk, 'safety_criticality', 'LOW')
            safety_stats[criticality] = safety_stats.get(criticality, 0) + 1
            total_safety_score += getattr(chunk, 'safety_score', 0)

        metadata = {
            'num_chunks': len(self.chunks),
            'embedding_dim': self.embeddings.shape[1],
            'model_name': self.config.embedding_model,
            'source_files': list(set(chunk.source_file for chunk in self.chunks)),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware_tier': self.hardware_info.performance_tier,
            'optimized_for': f"{self.hardware_info.cpu_count}CPU_{self.hardware_info.gpu_name}",
            'safety_statistics': safety_stats,
            'average_safety_score': total_safety_score / len(self.chunks) if self.chunks else 0,
            'chunking_strategy': 'recursive_with_overlap',
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap
        }

        with open(data_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        save_time = time.time() - start_time
        logger.info(f"💾 Vector store saved in {save_time:.2f}s")


class DocumentProcessor:
    """Enhanced document processor with recursive chunking and safety detection"""

    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        self.config = config
        self.hardware_info = hardware_detector.hardware_info
        self.text_splitter = RecursiveTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size
        )
        self.safety_detector = SafetyKeywordDetector() if config.enable_safety_detection else None

    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents with hardware-adaptive processing"""
        docs_path = Path(self.config.docs_dir)

        if not docs_path.exists():
            docs_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Please add .txt files to: {docs_path.absolute()}")
            return []

        documents = []
        supported_extensions = ['.txt', '.md']

        # First, collect all files to process
        all_files = []
        for ext in supported_extensions:
            for file_path in docs_path.glob(f"*{ext}"):
                all_files.append((file_path, ext))

        if not all_files:
            logger.warning(f"No .txt or .md files found in {docs_path.absolute()}")
            return []

        logger.info(f"Found {len(all_files)} document files to process")

        # Hardware-adaptive document loading
        max_workers = min(self.config.max_workers, self.hardware_info.cpu_count)

        if max_workers > 1 and self.hardware_info.available_ram_gb >= 4 and len(all_files) > 2:
            # Parallel loading for capable hardware with multiple files
            logger.debug(f"Using parallel loading with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for file_path, ext in all_files:
                    futures.append(executor.submit(self._load_single_document, file_path, ext))

                for future in as_completed(futures):
                    try:
                        doc = future.result()
                        if doc:
                            documents.append(doc)
                    except Exception as e:
                        logger.error(f"Error loading document: {e}")
        else:
            # Sequential loading (safer fallback)
            logger.debug("Using sequential loading")
            for file_path, ext in all_files:
                try:
                    doc = self._load_single_document(file_path, ext)
                    if doc:
                        documents.append(doc)
                    else:
                        logger.debug(f"Skipped {file_path.name} (too small or empty)")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"📚 Loaded {len(documents)} documents successfully")

        if len(documents) == 0 and len(all_files) > 0:
            logger.warning(
                f"Found {len(all_files)} files but loaded 0 documents - check file contents and min_chunk_size ({self.config.min_chunk_size})")

        return documents

    def _load_single_document(self, file_path: Path, ext: str) -> Optional[Dict[str, Any]]:
        """Load a single document with detailed error reporting"""
        try:
            logger.debug(f"Loading document: {file_path}")

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            content_length = len(content.strip())
            logger.debug(f"Document {file_path.name}: {content_length} characters")

            if content_length < self.config.min_chunk_size:
                logger.debug(f"Skipping {file_path.name}: too small ({content_length} < {self.config.min_chunk_size})")
                return None

            doc = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'content': content,
                'size': len(content),
                'extension': ext
            }

            logger.debug(f"Successfully loaded: {file_path.name} ({len(content)} chars)")
            return doc

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """Clean text with performance optimization"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Enhanced chunk document with file-aware content"""
        content = self.clean_text(document['content'])
        filename = document['filename']

        if len(content) < self.config.min_chunk_size:
            return []

        # Create file-aware chunker if enabled
        if self.config.enable_file_context:
            if not hasattr(self, 'file_aware_chunker'):
                self.file_aware_chunker = FileAwareChunker(self.config)
            
            # Extract file metadata
            file_metadata = self.file_aware_chunker.extract_file_metadata(filename)
        else:
            file_metadata = {'filename': filename, 'title': filename, 'category': 'general', 'topics': '', 'document_type': 'document'}

        # Use recursive text splitter
        chunks = self.text_splitter.split_text(content, filename)

        # Enhance chunks with file context and safety detection
        enhanced_chunks = []
        for chunk in chunks:
            # Create file-aware content for embedding if enabled
            if self.config.enable_file_context:
                chunk.enhanced_content = self.file_aware_chunker.create_file_aware_content(
                    chunk.content, file_metadata
                )
            else:
                chunk.enhanced_content = chunk.content
            
            # Store file metadata
            chunk.file_metadata = file_metadata

            # Apply safety detection if enabled
            if self.safety_detector:
                safety_analysis = self.safety_detector.analyze_chunk(chunk.content)

                # Update chunk with safety metadata
                chunk.safety_score = safety_analysis['safety_score']
                chunk.safety_criticality = safety_analysis['safety_criticality']
                chunk.category_scores = safety_analysis['category_scores']
                chunk.equipment_count = safety_analysis['equipment_count']
                chunk.voltage_count = safety_analysis['voltage_count']
                chunk.keywords_found = safety_analysis['keywords_found']

            enhanced_chunks.append(chunk)

        # Log enhanced statistics for this document
        if enhanced_chunks:
            safety_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for chunk in enhanced_chunks:
                safety_counts[chunk.safety_criticality] += 1

            logger.debug(f"Enhanced processing for {filename}:")
            logger.debug(f"  File metadata: {file_metadata}")
            logger.debug(f"  Safety analysis: {safety_counts}")

        return enhanced_chunks

    def process_all_documents(self) -> List[DocumentChunk]:
        """Process all documents with hardware optimization and enhanced chunking"""
        documents = self.load_documents()
        if not documents:
            return []

        logger.info("🔄 Processing documents with recursive chunking and safety detection...")

        all_chunks = []

        # Hardware-adaptive processing
        if (self.config.max_workers > 1 and
                self.hardware_info.cpu_count > 2 and
                self.hardware_info.available_ram_gb >= 4):

            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self.chunk_document, doc) for doc in documents]

                for future in as_completed(futures):
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Document processing error: {e}")
        else:
            # Sequential processing
            for doc in documents:
                try:
                    chunks = self.chunk_document(doc)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {doc['filename']}: {e}")

        # Log comprehensive statistics
        if all_chunks:
            safety_stats = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            total_safety_score = 0

            for chunk in all_chunks:
                safety_stats[chunk.safety_criticality] += 1
                total_safety_score += chunk.safety_score

            avg_safety_score = total_safety_score / len(all_chunks)

            logger.info(f"📝 Created {len(all_chunks)} chunks with enhanced processing")
            logger.info(
                f"🔒 Safety Distribution: HIGH={safety_stats['HIGH']}, MEDIUM={safety_stats['MEDIUM']}, LOW={safety_stats['LOW']}")
            logger.info(f"📊 Average Safety Score: {avg_safety_score:.2f}")

        return all_chunks


class OllamaClient:
    """Ollama client with hardware-adaptive settings"""

    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        self.config = config
        self.hardware_info = hardware_detector.hardware_info
        self.base_url = config.ollama_host
        self.session = requests.Session()
        self.is_warmed = False

        # Hardware-adaptive initialization
        if config.parallel_init and self.hardware_info.cpu_count > 2:
            threading.Thread(target=self._check_connection, daemon=True).start()
            if config.warm_ollama_on_startup:
                threading.Thread(target=self._warm_model, daemon=True).start()
        else:
            self._check_connection()
            if config.warm_ollama_on_startup:
                self._warm_model()

    def _check_connection(self):
        """Check Ollama connection with hardware considerations"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]

                if self.config.ollama_model not in model_names:
                    logger.warning(f"Model {self.config.ollama_model} not found. Available: {model_names}")
                else:
                    logger.info(f"🦙 Ollama connected: {self.config.ollama_model}")

                    # Set hardware-appropriate keep-alive
                    try:
                        keep_alive_payload = {
                            "model": self.config.ollama_model,
                            "keep_alive": self.config.ollama_keep_alive
                        }
                        self.session.post(f"{self.base_url}/api/generate",
                                          json=keep_alive_payload, timeout=10)
                        logger.info(f"🦙 Keep-alive: {self.config.ollama_keep_alive}")
                    except Exception as e:
                        logger.warning(f"Keep-alive setup failed: {e}")
            else:
                logger.warning("Ollama server not responding")
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama: {e}")

    def _warm_model(self):
        """Warm Ollama with hardware-appropriate workload"""
        if self.is_warmed:
            return

        logger.info("🔥 Warming Ollama model...")
        start_time = time.time()

        try:
            # Hardware-adaptive warm-up
            if self.hardware_info.total_ram_gb >= 16:
                warm_prompt = "Explain electrical safety procedures in detail."
                max_tokens = 100
            elif self.hardware_info.total_ram_gb >= 8:
                warm_prompt = "What is electrical safety? Provide key points."
                max_tokens = 50
            else:
                warm_prompt = "Define electrical safety."
                max_tokens = 25

            payload = {
                "model": self.config.ollama_model,
                "prompt": warm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens
                },
                "keep_alive": self.config.ollama_keep_alive
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                warm_time = time.time() - start_time
                self.is_warmed = True
                logger.info(f"🔥 Ollama warmed in {warm_time:.2f}s")
            else:
                logger.warning(f"Ollama warming failed: {response.status_code}")

        except Exception as e:
            logger.warning(f"Ollama warming failed: {e}")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response with hardware-adaptive parameters"""
        try:
            # Adjust max_tokens based on available RAM
            if self.hardware_info.total_ram_gb < 8:
                max_tokens = min(max_tokens, 300)  # Limit for low-RAM systems

            payload = {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": max_tokens
                },
                "keep_alive": self.config.ollama_keep_alive
            }

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return "Error: Could not generate response"

        except Exception as e:
            return f"Error: {e}"


class RAGEvaluator:
    """Improved evaluation pipeline with transparent, meaningful metrics for RAG systems"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.test_queries = self._create_test_dataset()

    def _create_test_dataset(self) -> List[Dict[str, Any]]:
        """Create comprehensive test dataset for electrical utility safety evaluation"""
        return [
            # === LOCKOUT/TAGOUT PROCEDURES (High Priority) ===
            {
                "query": "What is lockout tagout procedure?",
                "expected_keywords": ["lockout", "tagout", "loto", "energy", "isolation", "verification", "authorized employee"],
                "safety_critical": True,
                "category": "loto",
                "min_relevant_chunks": 2
            },
            {
                "query": "What are the steps for energy isolation in LOTO?",
                "expected_keywords": ["energy isolation", "shutdown", "isolation", "disconnect", "verify", "stored energy"],
                "safety_critical": True,
                "category": "loto",
                "min_relevant_chunks": 2
            },
            {
                "query": "Who can perform lockout tagout procedures?",
                "expected_keywords": ["authorized employee", "qualified", "training", "competency", "responsible"],
                "safety_critical": True,
                "category": "loto",
                "min_relevant_chunks": 1
            },
            {
                "query": "How do you verify equipment is de-energized?",
                "expected_keywords": ["verification", "test", "voltage detector", "de-energized", "isolation"],
                "safety_critical": True,
                "category": "loto",
                "min_relevant_chunks": 2
            },

            # === ARC FLASH PROTECTION ===
            {
                "query": "How do I perform arc flash protection?",
                "expected_keywords": ["arc flash", "ppe", "protection", "boundary", "cal/cm", "incident energy"],
                "safety_critical": True,
                "category": "arc_flash",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is arc flash boundary?",
                "expected_keywords": ["arc flash boundary", "boundary", "distance", "cal/cm", "incident energy", "1.2"],
                "safety_critical": True,
                "category": "arc_flash",
                "min_relevant_chunks": 1
            },
            {
                "query": "What are HRC categories for electrical work?",
                "expected_keywords": ["hrc", "hazard risk category", "incident energy", "cal/cm", "ppe", "protection"],
                "safety_critical": True,
                "category": "arc_flash",
                "min_relevant_chunks": 2
            },

            # === ELECTRICAL SAFETY ===
            {
                "query": "What PPE is required for electrical work?",
                "expected_keywords": ["ppe", "gloves", "protection", "safety", "insulated", "dielectric", "flame resistant"],
                "safety_critical": True,
                "category": "ppe",
                "min_relevant_chunks": 2
            },
            {
                "query": "What are minimum approach distances for electrical work?",
                "expected_keywords": ["minimum approach distance", "mad", "voltage", "boundary", "approach", "qualified"],
                "safety_critical": True,
                "category": "electrical_safety",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is the difference between qualified and unqualified persons?",
                "expected_keywords": ["qualified person", "unqualified", "training", "electrical", "boundary", "authorized"],
                "safety_critical": True,
                "category": "electrical_safety",
                "min_relevant_chunks": 1
            },

            # === GROUNDING AND BONDING ===
            {
                "query": "Explain electrical grounding procedures",
                "expected_keywords": ["ground", "grounding", "earth", "electrical", "bonding", "connection", "safety"],
                "safety_critical": True,
                "category": "grounding",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is the difference between grounding and bonding?",
                "expected_keywords": ["grounding", "bonding", "equipotential", "connection", "conductor", "earth"],
                "safety_critical": True,
                "category": "grounding",
                "min_relevant_chunks": 1
            },
            {
                "query": "How do you create an equipotential zone?",
                "expected_keywords": ["equipotential", "bonding", "zone", "resistance", "ohms", "connection"],
                "safety_critical": True,
                "category": "grounding",
                "min_relevant_chunks": 1
            },

            # === CONFINED SPACE SAFETY ===
            {
                "query": "What are confined space entry requirements?",
                "expected_keywords": ["confined space", "permit", "atmospheric testing", "ventilation", "attendant", "entry"],
                "safety_critical": True,
                "category": "confined_space",
                "min_relevant_chunks": 2
            },
            {
                "query": "What atmospheric tests are required for confined spaces?",
                "expected_keywords": ["atmospheric testing", "oxygen", "lel", "toxic", "ppm", "gas", "meter"],
                "safety_critical": True,
                "category": "confined_space",
                "min_relevant_chunks": 2
            },
            {
                "query": "What are the oxygen level requirements for confined space entry?",
                "expected_keywords": ["oxygen", "19.5", "23.5", "percent", "atmospheric", "testing"],
                "safety_critical": True,
                "category": "confined_space",
                "min_relevant_chunks": 1
            },

            # === FALL PROTECTION ===
            {
                "query": "What fall protection is required for utility work?",
                "expected_keywords": ["fall protection", "harness", "lanyard", "anchor", "height", "pfas"],
                "safety_critical": True,
                "category": "fall_protection",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is the minimum clearance distance for fall protection?",
                "expected_keywords": ["clearance", "distance", "fall", "lanyard", "deceleration", "height"],
                "safety_critical": True,
                "category": "fall_protection",
                "min_relevant_chunks": 1
            },
            {
                "query": "What are anchor point requirements for fall protection?",
                "expected_keywords": ["anchor point", "strength", "5000", "lb", "capacity", "certified"],
                "safety_critical": True,
                "category": "fall_protection",
                "min_relevant_chunks": 1
            },

            # === TRANSFORMER TESTING ===
            {
                "query": "What are the main transformer tests performed?",
                "expected_keywords": ["transformer", "insulation resistance", "turns ratio", "power factor", "dissolved gas"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is polarization index testing?",
                "expected_keywords": ["polarization index", "pi", "insulation", "resistance", "10 minute", "1 minute"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 1
            },
            {
                "query": "What are acceptable insulation resistance values for transformers?",
                "expected_keywords": ["insulation resistance", "transformer", "100", "megohm", "acceptance", "criteria"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 1
            },

            # === CIRCUIT BREAKER TESTING ===
            {
                "query": "What tests are performed on circuit breakers?",
                "expected_keywords": ["circuit breaker", "contact resistance", "timing", "insulation", "sf6", "mechanical"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is contact resistance testing for circuit breakers?",
                "expected_keywords": ["contact resistance", "micro-ohmmeter", "circuit breaker", "erosion", "connections"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 1
            },
            {
                "query": "What are SF6 gas analysis requirements?",
                "expected_keywords": ["sf6", "gas", "analysis", "purity", "moisture", "decomposition", "pressure"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 1
            },

            # === WEATHER SAFETY ===
            {
                "query": "What are extreme weather work restrictions?",
                "expected_keywords": ["extreme weather", "temperature", "wind", "work stoppage", "threshold", "safety"],
                "safety_critical": True,
                "category": "weather",
                "min_relevant_chunks": 2
            },
            {
                "query": "At what wind speed must work stop?",
                "expected_keywords": ["wind speed", "60", "km/h", "stop work", "weather", "threshold"],
                "safety_critical": True,
                "category": "weather",
                "min_relevant_chunks": 1
            },
            {
                "query": "What are cold weather working procedures?",
                "expected_keywords": ["cold weather", "temperature", "hypothermia", "exposure", "clothing", "wind chill"],
                "safety_critical": True,
                "category": "weather",
                "min_relevant_chunks": 2
            },

            # === WORKING ALONE SAFETY ===
            {
                "query": "What are working alone safety requirements?",
                "expected_keywords": ["working alone", "solo work", "communication", "check-in", "monitoring", "gps"],
                "safety_critical": True,
                "category": "working_alone",
                "min_relevant_chunks": 2
            },
            {
                "query": "How often must workers check in when working alone?",
                "expected_keywords": ["check-in", "30 minutes", "60 minutes", "communication", "interval"],
                "safety_critical": True,
                "category": "working_alone",
                "min_relevant_chunks": 1
            },

            # === PERMIT SYSTEMS ===
            {
                "query": "What is an Energy Isolation Permit?",
                "expected_keywords": ["energy isolation permit", "eip", "loto", "transmission", "verification"],
                "safety_critical": True,
                "category": "permits",
                "min_relevant_chunks": 1
            },
            {
                "query": "What permits are required for electrical work?",
                "expected_keywords": ["permit", "eap", "eip", "lwp", "electrical", "work permit"],
                "safety_critical": True,
                "category": "permits",
                "min_relevant_chunks": 2
            },
            {
                "query": "How long are work permits valid?",
                "expected_keywords": ["permit", "validity", "12 hours", "duration", "non-renewable"],
                "safety_critical": False,
                "category": "permits",
                "min_relevant_chunks": 1
            },

            # === FLAME RESISTANT CLOTHING ===
            {
                "query": "What is flame resistant clothing required for electrical work?",
                "expected_keywords": ["flame resistant", "fr", "clothing", "arc rated", "atpv", "hrc"],
                "safety_critical": True,
                "category": "ppe",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is ATPV rating for FR clothing?",
                "expected_keywords": ["atpv", "arc thermal performance value", "cal/cm", "rating", "protection"],
                "safety_critical": True,
                "category": "ppe",
                "min_relevant_chunks": 1
            },

            # === HAZARD ASSESSMENT ===
            {
                "query": "What is included in a daily hazard assessment?",
                "expected_keywords": ["hazard assessment", "daily", "environmental", "risk", "evaluation", "controls"],
                "safety_critical": True,
                "category": "assessment",
                "min_relevant_chunks": 2
            },
            {
                "query": "What is the risk scoring matrix formula?",
                "expected_keywords": ["risk", "probability", "severity", "P x S = R", "matrix"],
                "safety_critical": False,
                "category": "assessment",
                "min_relevant_chunks": 1
            },

            # === EMERGENCY PROCEDURES ===
            {
                "query": "What should you do if someone contacts an energized conductor?",
                "expected_keywords": ["electrical contact", "emergency", "de-energize", "rescue", "cpr", "first aid"],
                "safety_critical": True,
                "category": "emergency",
                "min_relevant_chunks": 1
            },
            {
                "query": "What is the emergency response for arc flash events?",
                "expected_keywords": ["arc flash", "emergency", "burns", "cool", "water", "medical", "airway"],
                "safety_critical": True,
                "category": "emergency",
                "min_relevant_chunks": 1
            },

            # === TECHNICAL SPECIFICATIONS ===
            {
                "query": "What are typical voltage levels in electrical utilities?",
                "expected_keywords": ["voltage", "kv", "transmission", "distribution", "480v", "13.8kv"],
                "safety_critical": False,
                "category": "technical",
                "min_relevant_chunks": 1
            },
            {
                "query": "What is dissolved gas analysis used for?",
                "expected_keywords": ["dissolved gas analysis", "dga", "transformer", "oil", "fault", "arcing"],
                "safety_critical": False,
                "category": "testing",
                "min_relevant_chunks": 1
            },

            # === REGULATORY COMPLIANCE ===
            {
                "query": "What OSHA standards apply to electrical utility work?",
                "expected_keywords": ["osha", "1910", "electrical", "safety", "standards", "compliance"],
                "safety_critical": True,
                "category": "compliance",
                "min_relevant_chunks": 1
            },
            {
                "query": "What are NFPA 70E requirements?",
                "expected_keywords": ["nfpa", "70e", "electrical", "safety", "arc flash", "ppe"],
                "safety_critical": True,
                "category": "compliance",
                "min_relevant_chunks": 1
            },

            # === EQUIPMENT SPECIFIC ===
            {
                "query": "What safety precautions are needed for switchgear work?",
                "expected_keywords": ["switchgear", "safety", "arc flash", "ppe", "voltage", "isolation"],
                "safety_critical": True,
                "category": "equipment",
                "min_relevant_chunks": 1
            },
            {
                "query": "How do you safely work on transmission lines?",
                "expected_keywords": ["transmission", "line", "safety", "voltage", "grounding", "isolation"],
                "safety_critical": True,
                "category": "equipment",
                "min_relevant_chunks": 2
            },

            # === TRAINING AND COMPETENCY ===
            {
                "query": "What training is required for electrical workers?",
                "expected_keywords": ["training", "electrical", "qualified", "competency", "certification", "refresher"],
                "safety_critical": True,
                "category": "training",
                "min_relevant_chunks": 2
            },
            {
                "query": "How often must safety training be renewed?",
                "expected_keywords": ["training", "renewal", "annual", "refresher", "certification", "competency"],
                "safety_critical": False,
                "category": "training",
                "min_relevant_chunks": 1
            }
        ]

    def evaluate_retrieval_quality(self, rag_system) -> Dict[str, float]:
        """Evaluate retrieval quality with transparent, meaningful metrics"""
        if not rag_system.vector_store.is_loaded:
            logger.warning("Vector store not loaded - cannot evaluate")
            return {}

        logger.info("🧪 Evaluating retrieval quality with improved metrics...")

        # Initialize metric accumulators
        metrics = {
            'relevance_at_k': [],  # % of top-K chunks that are relevant
            'coverage_at_k': [],  # % of queries where at least 1 relevant chunk found
            'keyword_coverage': [],  # % of expected keywords found in top-K
            'safety_prioritization': [],  # Safety chunks ranked higher when relevant
            'response_times': [],  # Query response times
            'chunks_retrieved': [],  # Number of chunks actually retrieved
        }

        detailed_results = []  # For debugging and analysis

        for test_case in self.test_queries[:self.config.eval_test_size]:
            try:
                start_time = time.time()

                # Generate query embedding and retrieve chunks
                query_embedding = rag_system.embedder.embed_query(test_case["query"])
                retrieved_chunks = rag_system.vector_store.search(
                    query_embedding,
                    top_k=self.config.similarity_top_k
                )

                response_time = time.time() - start_time
                metrics['response_times'].append(response_time)
                metrics['chunks_retrieved'].append(len(retrieved_chunks))

                # Calculate metrics for this query
                query_metrics = self._evaluate_single_query(retrieved_chunks, test_case)

                # Collect metrics
                metrics['relevance_at_k'].append(query_metrics['relevance_at_k'])
                metrics['coverage_at_k'].append(query_metrics['coverage_at_k'])
                metrics['keyword_coverage'].append(query_metrics['keyword_coverage'])
                metrics['safety_prioritization'].append(query_metrics['safety_prioritization'])

                # Store detailed results for analysis
                detailed_results.append({
                    'query': test_case['query'],
                    'category': test_case['category'],
                    'safety_critical': test_case['safety_critical'],
                    'metrics': query_metrics,
                    'chunks_found': len(retrieved_chunks),
                    'response_time': response_time
                })

                logger.debug(f"Query: {test_case['query'][:50]}... | "
                             f"Rel@K: {query_metrics['relevance_at_k']:.2f} | "
                             f"Cov@K: {query_metrics['coverage_at_k']:.2f} | "
                             f"Keywords: {query_metrics['keyword_coverage']:.2f}")

            except Exception as e:
                logger.warning(f"Evaluation failed for query '{test_case['query']}': {e}")

        # Calculate final metrics
        results = self._calculate_final_metrics(metrics, detailed_results)
        self._log_evaluation_results(results, detailed_results)

        return results

    def _evaluate_single_query(self, retrieved_chunks: List[Tuple[DocumentChunk, float]],
                               test_case: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single query with transparent metrics"""

        if not retrieved_chunks:
            return {
                'relevance_at_k': 0.0,
                'coverage_at_k': 0.0,
                'keyword_coverage': 0.0,
                'safety_prioritization': 0.0
            }

        expected_keywords = test_case["expected_keywords"]

        # 1. Relevance@K: What percentage of retrieved chunks are relevant?
        relevant_chunks = 0
        safety_relevant_chunks = []
        all_found_keywords = set()

        for chunk, score in retrieved_chunks:
            content_lower = chunk.content.lower()
            chunk_keywords = [kw for kw in expected_keywords if kw.lower() in content_lower]

            if chunk_keywords:  # Chunk contains at least one expected keyword
                relevant_chunks += 1
                all_found_keywords.update(chunk_keywords)

                # Track safety-critical chunks for prioritization metric
                if hasattr(chunk, 'safety_criticality'):
                    safety_relevant_chunks.append((chunk, score, chunk.safety_criticality))

        relevance_at_k = relevant_chunks / len(retrieved_chunks)

        # 2. Coverage@K: Did we find at least one relevant chunk?
        coverage_at_k = 1.0 if relevant_chunks > 0 else 0.0

        # 3. Keyword Coverage: What percentage of expected keywords were found?
        keyword_coverage = len(all_found_keywords) / len(expected_keywords)

        # 4. Safety Prioritization: Are safety-critical chunks ranked higher?
        safety_prioritization = self._calculate_safety_prioritization(
            safety_relevant_chunks, test_case['safety_critical']
        )

        return {
            'relevance_at_k': relevance_at_k,
            'coverage_at_k': coverage_at_k,
            'keyword_coverage': keyword_coverage,
            'safety_prioritization': safety_prioritization
        }

    def _calculate_safety_prioritization(self, safety_relevant_chunks: List[Tuple],
                                         query_is_safety_critical: bool) -> float:
        """Calculate how well safety-critical content is prioritized"""

        if not query_is_safety_critical or not safety_relevant_chunks:
            return 1.0  # Not applicable or no relevant chunks found

        # Check if high-criticality chunks appear before low-criticality ones
        criticality_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

        # Get criticality scores for relevant chunks in order of retrieval
        chunk_criticalities = []
        for chunk, score, criticality in safety_relevant_chunks:
            chunk_criticalities.append(criticality_scores.get(criticality, 1))

        if len(chunk_criticalities) <= 1:
            return 1.0  # Only one chunk, prioritization not applicable

        # Calculate how often higher criticality comes before lower criticality
        correct_pairs = 0
        total_pairs = 0

        for i in range(len(chunk_criticalities)):
            for j in range(i + 1, len(chunk_criticalities)):
                total_pairs += 1
                if chunk_criticalities[i] >= chunk_criticalities[j]:
                    correct_pairs += 1

        return correct_pairs / total_pairs if total_pairs > 0 else 1.0

    def _calculate_final_metrics(self, metrics: Dict[str, List[float]],
                                 detailed_results: List[Dict]) -> Dict[str, Any]:
        """Calculate final aggregated metrics with confidence intervals"""

        results = {}

        # Basic aggregated metrics
        for metric_name, values in metrics.items():
            if values:  # Only calculate if we have data
                results[f'{metric_name}_mean'] = sum(values) / len(values)
                results[f'{metric_name}_std'] = (
                                                        sum((x - results[f'{metric_name}_mean']) ** 2 for x in
                                                            values) / len(values)
                                                ) ** 0.5 if len(values) > 1 else 0.0
                results[f'{metric_name}_min'] = min(values)
                results[f'{metric_name}_max'] = max(values)
            else:
                results[f'{metric_name}_mean'] = 0.0
                results[f'{metric_name}_std'] = 0.0
                results[f'{metric_name}_min'] = 0.0
                results[f'{metric_name}_max'] = 0.0

        # Category-specific analysis
        category_metrics = {}
        categories = set(result['category'] for result in detailed_results)

        for category in categories:
            category_results = [r for r in detailed_results if r['category'] == category]
            if category_results:
                category_metrics[category] = {
                    'relevance_at_k': sum(r['metrics']['relevance_at_k'] for r in category_results) / len(
                        category_results),
                    'coverage_at_k': sum(r['metrics']['coverage_at_k'] for r in category_results) / len(
                        category_results),
                    'keyword_coverage': sum(r['metrics']['keyword_coverage'] for r in category_results) / len(
                        category_results),
                    'query_count': len(category_results)
                }

        results['category_breakdown'] = category_metrics

        # Overall quality indicators
        results['queries_evaluated'] = len(detailed_results)
        results['avg_chunks_per_query'] = results['chunks_retrieved_mean']
        results['total_evaluation_time'] = sum(metrics['response_times'])

        # Pass/fail indicators based on configuration thresholds
        results['quality_checks'] = {
            'good_relevance': results['relevance_at_k_mean'] >= self.config.min_relevance_at_k,
            'good_coverage': results['coverage_at_k_mean'] >= self.config.min_coverage_at_k,
            'good_keywords': results['keyword_coverage_mean'] >= self.config.min_keyword_coverage,
            'fast_response': results['response_times_mean'] <= self.config.max_response_time
        }

        return results

    def _log_evaluation_results(self, results: Dict[str, Any], detailed_results: List[Dict]):
        """Log evaluation results with clear explanations"""

        logger.info("📊 RAG System Evaluation Results:")
        logger.info("=" * 50)

        # Main metrics with explanations
        logger.info("🎯 Retrieval Quality Metrics:")
        logger.info(f"   • Relevance@K: {results['relevance_at_k_mean']:.3f} ± {results['relevance_at_k_std']:.3f}")
        logger.info(f"     (% of retrieved chunks that contain expected keywords)")

        logger.info(f"   • Coverage@K: {results['coverage_at_k_mean']:.3f} ± {results['coverage_at_k_std']:.3f}")
        logger.info(f"     (% of queries that found at least 1 relevant chunk)")

        logger.info(
            f"   • Keyword Coverage: {results['keyword_coverage_mean']:.3f} ± {results['keyword_coverage_std']:.3f}")
        logger.info(f"     (% of expected keywords found in top-K results)")

        logger.info(
            f"   • Safety Prioritization: {results['safety_prioritization_mean']:.3f} ± {results['safety_prioritization_std']:.3f}")
        logger.info(f"     (How well safety-critical content is ranked higher)")

        # Performance metrics
        logger.info("⚡ Performance Metrics:")
        logger.info(
            f"   • Avg Response Time: {results['response_times_mean']:.3f}s ± {results['response_times_std']:.3f}s")
        logger.info(f"   • Avg Chunks Retrieved: {results['chunks_retrieved_mean']:.1f}")
        logger.info(f"   • Total Queries Evaluated: {results['queries_evaluated']}")

        # Quality checks
        logger.info("✅ Quality Checks:")
        checks = results['quality_checks']
        for check_name, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"   • {check_name.replace('_', ' ').title()}: {status}")

        # Category breakdown
        if 'category_breakdown' in results and results['category_breakdown']:
            logger.info("📂 Performance by Category:")
            for category, metrics in results['category_breakdown'].items():
                logger.info(f"   • {category.upper()}:")
                logger.info(f"     - Relevance@K: {metrics['relevance_at_k']:.3f}")
                logger.info(f"     - Coverage@K: {metrics['coverage_at_k']:.3f}")
                logger.info(f"     - Keyword Coverage: {metrics['keyword_coverage']:.3f}")
                logger.info(f"     - Queries: {metrics['query_count']}")

        # Flag potential issues based on configuration thresholds
        if results['relevance_at_k_mean'] < self.config.min_relevance_at_k:
            logger.warning(
                f"⚠️  Low relevance@K ({results['relevance_at_k_mean']:.3f} < {self.config.min_relevance_at_k}) suggests chunking or embedding issues")

        if results['coverage_at_k_mean'] < self.config.min_coverage_at_k:
            logger.warning(
                f"⚠️  Low coverage@K ({results['coverage_at_k_mean']:.3f} < {self.config.min_coverage_at_k}) suggests missing content or poor retrieval")

        if results['keyword_coverage_mean'] < self.config.min_keyword_coverage:
            logger.warning(
                f"⚠️  Low keyword coverage ({results['keyword_coverage_mean']:.3f} < {self.config.min_keyword_coverage}) suggests semantic matching issues")

        if results['response_times_mean'] > self.config.max_response_time:
            logger.warning(
                f"⚠️  Slow response times ({results['response_times_mean']:.3f}s > {self.config.max_response_time}s) may impact user experience")

        if results['safety_prioritization_mean'] < 0.7:
            logger.warning("⚠️  Poor safety prioritization - check safety boosting settings")


class HardwareAdaptiveRAGSystem:
    """Main RAG system with comprehensive hardware adaptation and enhanced chunking"""

    def __init__(self, config: RAGConfig, hardware_detector: HardwareDetector):
        config.ensure_new_parameters()
        self.config = config
        self.hardware_detector = hardware_detector
        self.hardware_info = hardware_detector.hardware_info

        start_time = time.time()
        logger.info(f"🚀 Initializing Enhanced RAG System ({self.hardware_info.performance_tier} tier)")

        # Initialize components with hardware awareness
        if (config.parallel_init and
                self.hardware_info.cpu_count > 2 and
                self.hardware_info.available_ram_gb >= 4):

            # Parallel initialization for capable hardware
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'processor': executor.submit(DocumentProcessor, config, hardware_detector),
                    'embedder': executor.submit(HardwareAdaptiveEmbeddingGenerator, config, hardware_detector),
                    'vector_store': executor.submit(HardwareAdaptiveVectorStore, config, hardware_detector),
                    'llm_client': executor.submit(OllamaClient, config, hardware_detector),
                }

                self.processor = futures['processor'].result()
                self.embedder = futures['embedder'].result()
                self.vector_store = futures['vector_store'].result()
                self.llm_client = futures['llm_client'].result()
        else:
            # Sequential initialization for limited hardware
            self.processor = DocumentProcessor(config, hardware_detector)
            self.embedder = HardwareAdaptiveEmbeddingGenerator(config, hardware_detector)
            self.vector_store = HardwareAdaptiveVectorStore(config, hardware_detector)
            self.llm_client = OllamaClient(config, hardware_detector)

        self.cache = AdaptiveRAGCache(config, hardware_detector) if config.enable_cache else None
        self.evaluator = RAGEvaluator(config) if config.enable_evaluation else None

        init_time = time.time() - start_time
        logger.info(f"✅ Enhanced RAG System initialized in {init_time:.2f}s")

    def build_index(self, force_rebuild: bool = False) -> bool:
        """Build index with enhanced chunking and safety detection"""
        if force_rebuild and self.cache:
            self.cache.clear_cache()

        if not force_rebuild and self.vector_store.load():
            return True

        logger.info("🔨 Building new index with enhanced chunking...")

        chunks = self.processor.process_all_documents()
        if not chunks:
            logger.error("No chunks created")
            return False

        embeddings = self.embedder.generate_embeddings(chunks)
        if embeddings.size == 0:
            logger.error("No embeddings generated")
            return False

        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save()

        logger.info("✅ Enhanced index built successfully")
        return True

    def _build_prompt(self, query: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Build prompt with safety-aware context prioritization"""
        context_texts = []
        total_length = 0
        max_context = self.config.max_context_length

        # Sort chunks by safety criticality and similarity score
        sorted_chunks = sorted(context_chunks, key=lambda x: (
            {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(getattr(x[0], 'safety_criticality', 'LOW'), 1),
            x[1]  # similarity score
        ), reverse=True)

        for chunk, score in sorted_chunks:
            safety_info = f"[Safety: {getattr(chunk, 'safety_criticality', 'LOW')}]" if self.config.enable_safety_detection else ""
            chunk_text = f"[Source: {chunk.source_file}] {safety_info}\n{chunk.content}"

            if total_length + len(chunk_text) < max_context:
                context_texts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                break

        context = "\n\n".join(context_texts)

        prompt = f"""You are an expert assistant for electrical utility procedures. Answer the question based on the documents provided in the context.
        
        INSTRUCTIONS:
        Use ONLY information from the context - never add external knowledge
        For safety procedures, include all steps and warnings - never abbreviate critical information  
        If the context lacks relevant information, clearly state "The provided documents do not contain information about [topic]"
        Keep responses concise but complete for safety-critical topics
        When possible, reference which source document contains the information

        
        Context:
        {context}

        Question: {query}

        Answer:"""

        return prompt
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query with enhanced retrieval and safety prioritization"""
        if not self.vector_store.is_loaded:
            return {
                'error': 'Vector store not loaded. Please build index first.',
                'answer': '',
                'sources': [],
                'query': question
            }

        start_time = time.time()

        if self.cache:
            self.cache.stats['total_queries'] += 1

        try:
            # Check cache
            if self.cache:
                cached_result = self.cache.get_cached_result(question)
                if cached_result:
                    cached_result['cached'] = True
                    cached_result['total_time'] = time.time() - start_time
                    cached_result['hardware_tier'] = self.hardware_info.performance_tier
                    return cached_result

            if self.cache:
                self.cache.stats['misses'] += 1

            # Generate embedding
            query_embedding = self.embedder.embed_query(question)

            # Retrieve chunks with safety boosting
            retrieved_chunks = self.vector_store.search(query_embedding)

            if not retrieved_chunks:
                return {
                    'answer': 'No relevant information found in the knowledge base.',
                    'sources': [],
                    'query': question,
                    'retrieval_time': time.time() - start_time,
                    'num_chunks_retrieved': 0,
                    'hardware_tier': self.hardware_info.performance_tier,
                    'cached': False
                }

            # Generate response
            prompt = self._build_prompt(question, retrieved_chunks)
            response = self.llm_client.generate(prompt)

            # Prepare sources with safety information
            sources = []
            for chunk, score in retrieved_chunks:
                source_info = {
                    'source_file': chunk.source_file,
                    'chunk_id': chunk.chunk_id,
                    'similarity_score': score,
                    'content_preview': chunk.content
                }

                # Add safety information if available
                if self.config.enable_safety_detection:
                    source_info.update({
                        'safety_criticality': getattr(chunk, 'safety_criticality', 'LOW'),
                        'safety_score': getattr(chunk, 'safety_score', 0.0),
                        'safety_keywords': getattr(chunk, 'keywords_found', [])[:5]  # Top 5 keywords
                    })

                sources.append(source_info)

            total_time = time.time() - start_time

            result = {
                'answer': response,
                'sources': sources,
                'query': question,
                'retrieval_time': total_time,
                'num_chunks_retrieved': len(retrieved_chunks),
                'prompt_length': len(prompt),
                'hardware_tier': self.hardware_info.performance_tier,
                'cached': False,
                'safety_boosted': self.config.enable_safety_boosting
            }

            # Cache result
            if self.cache:
                self.cache.cache_result(question, result.copy(), query_embedding)

            return result

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'error': str(e),
                'answer': '',
                'sources': [],
                'query': question,
                'hardware_tier': self.hardware_info.performance_tier
            }

    def evaluate_system(self) -> Dict[str, Any]:
        """Evaluate system performance with chunking quality metrics"""
        if not self.evaluator:
            return {'evaluation_disabled': True}

        return self.evaluator.evaluate_retrieval_quality(self)

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get comprehensive hardware information"""
        return {
            'summary': self.hardware_detector.get_hardware_summary(),
            'details': asdict(self.hardware_info),
            'optimizations': self.hardware_detector.get_recommended_params(),
            'config_applied': self.config.hardware_optimized
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_stats()
        return {'caching_disabled': True}

    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety analysis statistics"""
        if not self.vector_store.is_loaded:
            return {'index_not_loaded': True}

        safety_stats = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        total_safety_score = 0
        safety_keywords_count = {}

        for chunk in self.vector_store.chunks:
            criticality = getattr(chunk, 'safety_criticality', 'LOW')
            safety_stats[criticality] += 1
            total_safety_score += getattr(chunk, 'safety_score', 0)

            # Count safety keywords
            keywords = getattr(chunk, 'keywords_found', [])
            for keyword in keywords:
                safety_keywords_count[keyword] = safety_keywords_count.get(keyword, 0) + 1

        # Top 10 most common safety keywords
        top_keywords = sorted(safety_keywords_count.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total_chunks': len(self.vector_store.chunks),
            'safety_distribution': safety_stats,
            'average_safety_score': total_safety_score / len(
                self.vector_store.chunks) if self.vector_store.chunks else 0,
            'top_safety_keywords': top_keywords,
            'safety_detection_enabled': self.config.enable_safety_detection
        }

    def clear_cache(self):
        """Clear all caches"""
        if self.cache:
            self.cache.clear_cache()


def main():
    """Enhanced main function with comprehensive features"""
    overall_start = time.time()

    print("🚀 Enhanced Hardware-Adaptive RAG System for Electrical Utility Documentation")
    print("   ✨ Recursive Chunking • 🔒 Safety Detection • 📊 Quality Evaluation")
    print("=" * 80)

    # Hardware Detection
    print("\n🔍 Detecting Hardware Capabilities...")
    hardware_detector = HardwareDetector()

    print(f"\n{hardware_detector.get_hardware_summary()}")

    # Configuration with Hardware Optimization
    config_file = "rag_config.json"
    if Path(config_file).exists():
        config = RAGConfig.load(config_file)
        print(f"\n📝 Loaded configuration from {config_file}")
    else:
        config = RAGConfig()
        print(f"\n📝 Created default configuration")

    # Apply hardware optimizations
    print(f"\n⚡ Applying hardware optimizations for {hardware_detector.hardware_info.performance_tier} tier...")
    config.apply_hardware_optimizations(hardware_detector)
    config.save(config_file)

    print(f"   📊 Enhanced Chunking Parameters:")
    print(f"     • Chunk Size: {config.chunk_size} tokens (primary)")
    print(f"     • Chunk Overlap: {config.chunk_overlap} tokens")
    print(f"     • Safety Detection: {config.enable_safety_detection}")
    print(f"     • Safety Boosting: {config.enable_safety_boosting}")
    print(f"     • Top-K Retrieval: {config.similarity_top_k}")

    # Initialize RAG system
    print(f"\n🚀 Initializing Enhanced RAG System...")
    init_start = time.time()
    rag = HardwareAdaptiveRAGSystem(config, hardware_detector)
    init_time = time.time() - init_start

    # Build index
    print(f"\n📚 Building/Loading Document Index with Enhanced Chunking...")

    # Quick diagnostic check
    docs_path = Path(config.docs_dir)
    if docs_path.exists():
        txt_files = list(docs_path.glob("*.txt"))
        md_files = list(docs_path.glob("*.md"))
        print(f"   📁 Found in {docs_path}: {len(txt_files)} .txt files, {len(md_files)} .md files")

        if len(txt_files) + len(md_files) == 0:
            print(f"   ⚠️  No documents found! Please add .txt or .md files to: {docs_path.absolute()}")
        else:
            for file_path in (txt_files + md_files)[:3]:  # Show first 3 files
                try:
                    size = file_path.stat().st_size
                    print(f"   📄 {file_path.name}: {size} bytes")
                except:
                    print(f"   📄 {file_path.name}: (size unknown)")
    else:
        print(f"   ⚠️  Documents directory doesn't exist: {docs_path.absolute()}")

    index_start = time.time()
    if not rag.build_index():
        print("❌ Failed to build index.")
        return
    index_time = time.time() - index_start

    total_startup_time = time.time() - overall_start

    print(f"\n✅ Enhanced RAG System Ready!")
    print(f"   📁 Documents: {config.docs_dir}/")
    print(f"   🤖 Model: {config.ollama_model}")
    print(f"   📄 Chunks: {len(rag.vector_store.chunks)}")
    print(f"   🧠 Cache: {'Enabled' if config.enable_cache else 'Disabled'}")
    print(f"   🔧 Hardware Tier: {config.hardware_tier}")
    print(f"   🔒 Safety Detection: {'Enabled' if config.enable_safety_detection else 'Disabled'}")

    # Show safety statistics
    safety_stats = rag.get_safety_statistics()
    if 'safety_distribution' in safety_stats:
        dist = safety_stats['safety_distribution']
        print(f"   📊 Safety Distribution: HIGH={dist['HIGH']}, MEDIUM={dist['MEDIUM']}, LOW={dist['LOW']}")

    print(f"\n⏱️  Performance:")
    print(f"   • Initialization: {init_time:.2f}s")
    print(f"   • Index loading: {index_time:.2f}s")
    print(f"   • Total startup: {total_startup_time:.2f}s")

    # Test first query
    print(f"\n🏁 Testing first query performance...")
    first_query_start = time.time()
    test_result = rag.query("What is lockout tagout procedure?")
    first_query_time = time.time() - first_query_start

    if 'error' not in test_result:
        print(f"   ⚡ First query TTFT: {first_query_time:.2f}s")
        print(f"   🚀 Models warmed and ready!")

        # Show safety boosting if enabled
        if test_result.get('safety_boosted'):
            print(f"   🔒 Safety boosting applied to results")
    else:
        print(f"   ⚠️  First query failed: {test_result.get('error', 'Unknown error')}")

    # Run evaluation if enabled
    if config.enable_evaluation:
        print(f"\n🧪 Running comprehensive system evaluation...")
        print(f"   📊 Dataset: {len(rag.evaluator.test_queries)} total queries")
        print(f"   🎯 Evaluating: {config.eval_test_size} queries")
        print(f"   📈 Categories: LOTO, Arc Flash, PPE, Grounding, Confined Space, Fall Protection, etc.")
        
        eval_start = time.time()
        eval_results = rag.evaluate_system()
        eval_time = time.time() - eval_start
        
        if 'evaluation_disabled' not in eval_results:
            grade = eval_results.get('grade_letter', 'N/A')
            overall_score = eval_results.get('overall_grade', 0) * 100
            
            print(f"   🎓 Overall Grade: {grade} ({overall_score:.1f}%)")
            print(f"   📊 Relevance@K: {eval_results.get('relevance_at_k_mean', 0):.3f}")
            print(f"   📊 Coverage@K: {eval_results.get('coverage_at_k_mean', 0):.3f}")
            print(f"   📊 Safety Prioritization: {eval_results.get('safety_prioritization_mean', 0):.3f}")
            print(f"   ⏱️ Evaluation Time: {eval_time:.1f}s")
            
            # Show quality check summary
            checks = eval_results.get('quality_checks', {})
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            print(f"   ✅ Quality Checks: {passed_checks}/{total_checks} passed")
            
            # Flag major issues
            if eval_results.get('relevance_at_k_mean', 0) < 0.5:
                print(f"   ⚠️ WARNING: Low relevance scores detected")
            if eval_results.get('coverage_at_k_mean', 0) < 0.7:
                print(f"   ⚠️ WARNING: Poor query coverage detected")
                
        else:
            print("   ❌ Evaluation disabled in configuration")

    # Interactive Mode
    print(f"\n💬 Interactive Mode - Enhanced RAG System")
    print(
        "Commands: 'quit', 'stats', 'safety', 'eval', 'hardware'/'gpu', 'clear-cache', 'benchmark'/'speed-test', or ask a question")
    print("-" * 80)

    while True:
        try:
            query = input(f"\n🔍 Query [{rag.hardware_info.performance_tier}]: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if query.lower() in ['hardware', 'gpu']:
                hw_info = rag.get_hardware_info()
                print(f"\n🔧 Hardware Information:")
                print(hw_info['summary'])
                continue

            if query.lower() == 'safety':
                safety_stats = rag.get_safety_statistics()
                if 'index_not_loaded' in safety_stats:
                    print("⚠️ Index not loaded")
                else:
                    print(f"\n🔒 Safety Analysis Statistics:")
                    dist = safety_stats['safety_distribution']
                    print(f"   Total Chunks: {safety_stats['total_chunks']}")
                    print(f"   Safety Distribution: HIGH={dist['HIGH']}, MEDIUM={dist['MEDIUM']}, LOW={dist['LOW']}")
                    print(f"   Average Safety Score: {safety_stats['average_safety_score']:.2f}")
                    print(
                        f"   Safety Detection: {'Enabled' if safety_stats['safety_detection_enabled'] else 'Disabled'}")

                    if safety_stats['top_safety_keywords']:
                        print(f"   Top Safety Keywords:")
                        for keyword, count in safety_stats['top_safety_keywords'][:5]:
                            print(f"     • {keyword}: {count}")
                continue

        

            if query.lower() == 'eval-summary':
                if config.enable_evaluation:
                    eval_results = rag.evaluate_system()
                    grade = eval_results.get('grade_letter', 'N/A')
                    overall_score = eval_results.get('overall_grade', 0) * 100
                    
                    print(f"\n🎓 Quick Evaluation Summary:")
                    print(f"   Overall Grade: {grade} ({overall_score:.1f}%)")
                    print(f"   Queries Tested: {eval_results.get('queries_evaluated', 0)}")
                    print(f"   Avg Relevance: {eval_results.get('relevance_at_k_mean', 0):.3f}")
                    print(f"   Avg Coverage: {eval_results.get('coverage_at_k_mean', 0):.3f}")
                    
                    # Show failing quality checks
                    failed_checks = [name for name, passed in eval_results.get('quality_checks', {}).items() if not passed]
                    if failed_checks:
                        print(f"   ❌ Failed Checks: {', '.join(failed_checks)}")
                    else:
                        print(f"   ✅ All quality checks passed!")
                else:
                    print("❌ Evaluation not enabled")
                continue


            
            if query.lower() in ['benchmark', 'speed-test']:
                print("🏁 Running hardware-optimized benchmark...")
                speeds = []
                test_queries = [
                    "What is electrical safety?",
                    "How do lockout tagout procedures work?",
                    "What PPE is required for arc flash protection?",
                    "Explain electrical grounding procedures",
                    "What are switching operation safety requirements?"
                ]

                for i, test_query in enumerate(test_queries):
                    start = time.time()
                    result = rag.query(test_query)
                    query_time = time.time() - start
                    speeds.append(query_time)
                    cache_status = "cached" if result.get('cached') else "fresh"
                    tier = result.get('hardware_tier', 'UNKNOWN')
                    safety_boost = "boosted" if result.get('safety_boosted') else "standard"
                    print(f"   Query {i + 1}: {query_time:.2f}s ({cache_status}, {tier}, {safety_boost})")

                avg_speed = sum(speeds) / len(speeds)
                print(f"🏁 Average: {avg_speed:.2f}s on {rag.hardware_info.performance_tier} hardware")
                continue

            if query.lower() == 'stats':
                stats = rag.get_cache_stats()
                print("\n📊 System Statistics:")
                if 'caching_disabled' in stats:
                    print("   Caching is disabled")
                else:
                    print(f"   Total queries: {stats['total_queries']}")
                    print(f"   Cache hits: {stats['hits']}")
                    print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
                    print(f"   Hardware tier: {stats.get('hardware_tier', 'UNKNOWN')}")
                    print(f"   Available RAM: {stats.get('available_ram_gb', 0):.1f}GB")

                # Add safety statistics
                safety_stats = rag.get_safety_statistics()
                if 'index_not_loaded' not in safety_stats:
                    dist = safety_stats['safety_distribution']
                    print(f"   Safety chunks: HIGH={dist['HIGH']}, MEDIUM={dist['MEDIUM']}, LOW={dist['LOW']}")
                continue

            if query.lower() == 'clear-cache':
                rag.clear_cache()
                print("✅ Cache cleared")
                continue

            if not query:
                continue

            print("🤔 Processing...")
            result = rag.query(query)

            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue

            # Display result with enhanced information
            cache_info = " ⚡ (cached)" if result.get('cached', False) else ""
            safety_info = " 🔒 (safety boosted)" if result.get('safety_boosted', False) else ""
            tier_info = f" [{result.get('hardware_tier', 'UNKNOWN')}]"

            print(f"\n📝 Answer{cache_info}{safety_info}{tier_info}:")
            print(result['answer'])

            print(f"\n📚 Sources ({result['num_chunks_retrieved']} chunks):")
            for i, source in enumerate(result['sources'], 1):
                safety_display = ""
                if 'safety_criticality' in source:
                    criticality = source['safety_criticality']
                    safety_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(criticality, "⚪")
                    safety_display = f" {safety_icon}{criticality}"

                print(f"   {i}. {source['source_file']} (similarity: {source['similarity_score']:.3f}){safety_display}")

                # Show safety keywords if available
                if source.get('safety_keywords'):
                    keywords = ', '.join(source['safety_keywords'][:3])
                    print(f"      Keywords: {keywords}")

            time_key = 'total_time' if result.get('cached') else 'retrieval_time'
            print(f"\n⏱️  Response time: {result.get(time_key, 0):.2f}s")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n👋 Goodbye! Enhanced RAG System with {rag.hardware_info.performance_tier} tier hardware")

    # Final statistics
    if config.enable_cache:
        final_stats = rag.get_cache_stats()
        print(
            f"📊 Final session stats: {final_stats.get('total_queries', 0)} queries, {final_stats.get('hit_rate_percent', 0):.1f}% cache hit rate")


if __name__ == "__main__":
    main()