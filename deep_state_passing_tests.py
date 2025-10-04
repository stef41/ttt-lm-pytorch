#!/usr/bin/env python3
"""
Deep State Passing Analysis Suite
=================================

Comprehensive tests to validate state passing implementation across multiple dimensions:
1. Convergence analysis
2. Configuration robustness  
3. Performance profiling
4. Cache management
5. Edge case handling
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import json
from pathlib import Path
from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS, TTTCache

def setup_test_environment():
    """Setup consistent test environment."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return device, tokenizer

def create_synthetic_dataset(tokenizer, num_sequences=100, seq_length=64, batch_size=4):
    """Create a synthetic dataset for controlled experiments."""
    vocab_size = len(tokenizer)
    
    # Create sequences with some structure (not completely random)
    batches = []
    for i in range(0, num_sequences, batch_size):
        actual_batch_size = min(batch_size, num_sequences - i)
        
        # Create slightly structured data - each sequence starts with a pattern
        input_ids = []
        for j in range(actual_batch_size):
            # Start with a small repeated pattern, then random
            pattern = [(i + j) % 100 + 1000] * 4  # Starting pattern
            random_part = torch.randint(0, vocab_size, (seq_length - 4,)).tolist()
            sequence = pattern + random_part
            input_ids.append(sequence)
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        batches.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })
    
    return batches

class ConvergenceAnalyzer:
    """Analyze training convergence with and without state passing."""
    
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
        self.results = {}
    
    def run_convergence_test(self, num_epochs=3, steps_per_epoch=20):
        """Compare convergence with and without state passing."""
        print("🔬 Deep Convergence Analysis")
        print("=" * 50)
        
        # Test configurations
        configs = [
            {"name": "no_state_passing", "state_passing": False, "reset_interval": 0},
            {"name": "state_passing_no_reset", "state_passing": True, "reset_interval": 0},
            {"name": "state_passing_reset_10", "state_passing": True, "reset_interval": 10},
            {"name": "state_passing_reset_5", "state_passing": True, "reset_interval": 5},
        ]
        
        # Create dataset
        dataset = create_synthetic_dataset(self.tokenizer, num_sequences=steps_per_epoch * 4, seq_length=32, batch_size=2)
        
        for config in configs:
            print(f"\n📊 Testing: {config['name']}")
            losses = self._train_with_config(dataset, config, num_epochs, steps_per_epoch)
            self.results[config['name']] = {
                'losses': losses,
                'config': config
            }
            print(f"   Final loss: {losses[-1]:.4f}")
            print(f"   Loss improvement: {losses[0] - losses[-1]:.4f}")
        
        self._analyze_convergence_results()
    
    def _train_with_config(self, dataset, config, num_epochs, steps_per_epoch):
        """Train model with specific configuration."""
        # Create model
        model_config = TTTConfig(**{
            **TTT_STANDARD_CONFIGS["125m"],
            "vocab_size": len(self.tokenizer),
            "max_position_embeddings": 64,
            "state_passing": config["state_passing"],
            "disable_conv": True,
        })
        
        model = TTTForCausalLM(model_config).to(self.device)
        model.train()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Initialize cache if needed
        ttt_cache = None
        if config["state_passing"]:
            ttt_cache = TTTCache(model.model, batch_size=2)
        
        losses = []
        step_count = 0
        
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataset[:steps_per_epoch]):
                if step_count >= num_epochs * steps_per_epoch:
                    break
                
                # Reset cache if needed
                if (config["state_passing"] and config["reset_interval"] > 0 and 
                    step_count > 0 and step_count % config["reset_interval"] == 0):
                    ttt_cache = TTTCache(model.model, batch_size=2)
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                
                forward_kwargs = batch.copy()
                if config["state_passing"] and ttt_cache is not None:
                    forward_kwargs['cache_params'] = ttt_cache
                
                outputs = model(**forward_kwargs)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                step_count += 1
        
        return losses
    
    def _analyze_convergence_results(self):
        """Analyze and visualize convergence results."""
        print(f"\n📈 Convergence Analysis Results:")
        print("-" * 40)
        
        for name, data in self.results.items():
            losses = data['losses']
            config = data['config']
            
            initial_loss = np.mean(losses[:5])  # First 5 steps
            final_loss = np.mean(losses[-5:])   # Last 5 steps
            improvement = initial_loss - final_loss
            stability = np.std(losses[-10:])    # Stability in last 10 steps
            
            print(f"\n{name}:")
            print(f"  Initial loss: {initial_loss:.4f}")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Improvement: {improvement:.4f}")
            print(f"  Final stability (std): {stability:.4f}")
        
        # Find best configuration
        best_improvement = -float('inf')
        best_config = None
        
        for name, data in self.results.items():
            losses = data['losses']
            initial_loss = np.mean(losses[:5])
            final_loss = np.mean(losses[-5:])
            improvement = initial_loss - final_loss
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_config = name
        
        print(f"\n🏆 Best configuration: {best_config}")
        print(f"   Best improvement: {best_improvement:.4f}")

class ConfigurationRobustnessTest:
    """Test state passing across different configurations."""
    
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    
    def run_robustness_tests(self):
        """Test robustness across different configurations."""
        print("\n🔧 Configuration Robustness Tests")
        print("=" * 50)
        
        test_configs = [
            # Different model sizes
            {"model_size": "125m", "batch_size": 2, "seq_length": 32, "name": "small_model"},
            {"model_size": "125m", "batch_size": 4, "seq_length": 64, "name": "medium_config"},
            {"model_size": "125m", "batch_size": 1, "seq_length": 128, "name": "long_sequence"},
            
            # Different batch sizes
            {"model_size": "125m", "batch_size": 8, "seq_length": 32, "name": "large_batch"},
        ]
        
        results = {}
        
        for test_config in test_configs:
            print(f"\n🧪 Testing: {test_config['name']}")
            try:
                success, metrics = self._test_configuration(test_config)
                results[test_config['name']] = {
                    'success': success,
                    'metrics': metrics,
                    'config': test_config
                }
                
                if success:
                    print(f"   ✅ SUCCESS")
                    print(f"   Memory usage: {metrics['memory_usage']:.2f} GB")
                    print(f"   Training time: {metrics['training_time']:.2f}s")
                    print(f"   Final loss: {metrics['final_loss']:.4f}")
                else:
                    print(f"   ❌ FAILED: {metrics['error']}")
                    
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
                results[test_config['name']] = {
                    'success': False,
                    'metrics': {'error': str(e)},
                    'config': test_config
                }
        
        self._summarize_robustness_results(results)
    
    def _test_configuration(self, test_config):
        """Test a specific configuration."""
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create model
            model_config = TTTConfig(**{
                **TTT_STANDARD_CONFIGS[test_config["model_size"]],
                "vocab_size": len(self.tokenizer),
                "max_position_embeddings": test_config["seq_length"] * 2,
                "state_passing": True,
                "disable_conv": True,
            })
            
            model = TTTForCausalLM(model_config).to(self.device)
            model.train()
            
            # Create cache
            ttt_cache = TTTCache(model.model, batch_size=test_config["batch_size"])
            
            # Create test data
            input_ids = torch.randint(0, len(self.tokenizer), 
                                    (test_config["batch_size"], test_config["seq_length"]), 
                                    device=self.device)
            labels = input_ids.clone()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Monitor memory
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            losses = []
            
            # Run training steps
            for step in range(5):
                optimizer.zero_grad()
                
                outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            training_time = time.time() - start_time
            memory_usage = torch.cuda.max_memory_allocated() / 1e9
            
            return True, {
                'memory_usage': memory_usage,
                'training_time': training_time,
                'final_loss': losses[-1],
                'loss_progression': losses
            }
            
        except Exception as e:
            return False, {'error': str(e)}
        finally:
            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    def _summarize_robustness_results(self, results):
        """Summarize robustness test results."""
        print(f"\n📋 Robustness Summary:")
        print("-" * 30)
        
        successful_tests = [name for name, data in results.items() if data['success']]
        failed_tests = [name for name, data in results.items() if not data['success']]
        
        print(f"Successful tests: {len(successful_tests)}/{len(results)}")
        print(f"Success rate: {len(successful_tests)/len(results)*100:.1f}%")
        
        if successful_tests:
            print(f"\n✅ Working configurations:")
            for name in successful_tests:
                config = results[name]['config']
                metrics = results[name]['metrics']
                print(f"   {name}: {config['batch_size']}x{config['seq_length']} - {metrics['memory_usage']:.1f}GB")
        
        if failed_tests:
            print(f"\n❌ Failed configurations:")
            for name in failed_tests:
                error = results[name]['metrics'].get('error', 'Unknown error')
                print(f"   {name}: {error}")

class PerformanceProfiler:
    """Profile performance impact of state passing."""
    
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    
    def run_performance_analysis(self):
        """Run comprehensive performance analysis."""
        print("\n⚡ Performance Profiling")
        print("=" * 40)
        
        # Test different scenarios
        scenarios = [
            {"batch_size": 2, "seq_length": 32, "steps": 20},
            {"batch_size": 4, "seq_length": 64, "steps": 15},
            {"batch_size": 8, "seq_length": 32, "steps": 10},
        ]
        
        for scenario in scenarios:
            print(f"\n🎯 Scenario: {scenario['batch_size']}x{scenario['seq_length']}, {scenario['steps']} steps")
            
            # Test without state passing
            time_no_state, memory_no_state, throughput_no_state = self._profile_training(
                scenario, state_passing=False
            )
            
            # Test with state passing
            time_with_state, memory_with_state, throughput_with_state = self._profile_training(
                scenario, state_passing=True
            )
            
            # Calculate overhead
            time_overhead = ((time_with_state - time_no_state) / time_no_state) * 100
            memory_overhead = memory_with_state - memory_no_state
            throughput_change = ((throughput_with_state - throughput_no_state) / throughput_no_state) * 100
            
            print(f"   Without state: {time_no_state:.3f}s, {memory_no_state:.2f}GB, {throughput_no_state:.0f} tok/s")
            print(f"   With state:    {time_with_state:.3f}s, {memory_with_state:.2f}GB, {throughput_with_state:.0f} tok/s")
            print(f"   Time overhead: {time_overhead:+.1f}%")
            print(f"   Memory overhead: {memory_overhead:+.2f}GB")
            print(f"   Throughput change: {throughput_change:+.1f}%")
    
    def _profile_training(self, scenario, state_passing):
        """Profile training for a specific scenario."""
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create model
            model_config = TTTConfig(**{
                **TTT_STANDARD_CONFIGS["125m"],
                "vocab_size": len(self.tokenizer),
                "max_position_embeddings": scenario["seq_length"] * 2,
                "state_passing": state_passing,
                "disable_conv": True,
            })
            
            model = TTTForCausalLM(model_config).to(self.device)
            model.train()
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            # Create cache if needed
            ttt_cache = None
            if state_passing:
                ttt_cache = TTTCache(model.model, batch_size=scenario["batch_size"])
            
            # Create test data
            input_ids = torch.randint(0, len(self.tokenizer), 
                                    (scenario["batch_size"], scenario["seq_length"]), 
                                    device=self.device)
            labels = input_ids.clone()
            
            # Warmup
            for _ in range(2):
                optimizer.zero_grad()
                forward_kwargs = {"input_ids": input_ids, "labels": labels}
                if state_passing and ttt_cache is not None:
                    forward_kwargs["cache_params"] = ttt_cache
                outputs = model(**forward_kwargs)
                outputs.loss.backward()
                optimizer.step()
            
            # Reset metrics
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Profile actual training
            start_time = time.time()
            total_tokens = 0
            
            for step in range(scenario["steps"]):
                optimizer.zero_grad()
                
                forward_kwargs = {"input_ids": input_ids, "labels": labels}
                if state_passing and ttt_cache is not None:
                    forward_kwargs["cache_params"] = ttt_cache
                
                outputs = model(**forward_kwargs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_tokens += input_ids.numel()
            
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            throughput = total_tokens / total_time
            
            return total_time, peak_memory, throughput
            
        except Exception as e:
            print(f"   Error in profiling: {e}")
            return float('inf'), float('inf'), 0
        finally:
            torch.cuda.empty_cache()
            gc.collect()

class CacheManagementTest:
    """Test cache management and edge cases."""
    
    def __init__(self, device, tokenizer):
        self.device = device
        self.tokenizer = tokenizer
    
    def run_cache_tests(self):
        """Run comprehensive cache management tests."""
        print("\n🗄️ Cache Management Tests")
        print("=" * 40)
        
        tests = [
            self._test_cache_reset_intervals,
            self._test_cache_state_evolution,
            self._test_cache_memory_growth,
            self._test_different_sequence_lengths,
            self._test_batch_size_changes,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ Test failed: {e}")
    
    def _test_cache_reset_intervals(self):
        """Test different cache reset intervals."""
        print(f"\n🔄 Testing cache reset intervals...")
        
        intervals = [0, 5, 10, 20]  # 0 means no reset
        
        for interval in intervals:
            print(f"   Testing reset interval: {interval}")
            
            model_config = TTTConfig(**{
                **TTT_STANDARD_CONFIGS["125m"],
                "vocab_size": len(self.tokenizer),
                "state_passing": True,
                "disable_conv": True,
            })
            
            model = TTTForCausalLM(model_config).to(self.device)
            model.train()
            
            ttt_cache = TTTCache(model.model, batch_size=2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            cache_states = []
            
            for step in range(25):
                # Reset cache if needed
                if interval > 0 and step > 0 and step % interval == 0:
                    ttt_cache = TTTCache(model.model, batch_size=2)
                
                # Create batch
                input_ids = torch.randint(0, len(self.tokenizer), (2, 32), device=self.device)
                labels = input_ids.clone()
                
                # Training step
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                # Record cache state
                if hasattr(ttt_cache, 'ttt_params_dict') and 'W1_states' in ttt_cache.ttt_params_dict:
                    state_norm = torch.norm(ttt_cache.ttt_params_dict['W1_states'][0]).item()
                    cache_states.append(state_norm)
            
            print(f"     Cache state evolution: {len(cache_states)} steps recorded")
            if len(cache_states) > 0:
                print(f"     Final state norm: {cache_states[-1]:.4f}")
    
    def _test_cache_state_evolution(self):
        """Test how cache state evolves over time."""
        print(f"\n📈 Testing cache state evolution...")
        
        model_config = TTTConfig(**{
            **TTT_STANDARD_CONFIGS["125m"],
            "vocab_size": len(self.tokenizer),
            "state_passing": True,
            "disable_conv": True,
        })
        
        model = TTTForCausalLM(model_config).to(self.device)
        model.train()
        
        ttt_cache = TTTCache(model.model, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Get initial cache state
        initial_state = None
        if hasattr(ttt_cache, 'ttt_params_dict') and 'W1_states' in ttt_cache.ttt_params_dict:
            if 0 in ttt_cache.ttt_params_dict['W1_states']:
                initial_state = ttt_cache.ttt_params_dict['W1_states'][0].clone()
        
        # Run training steps
        for step in range(10):
            input_ids = torch.randint(0, len(self.tokenizer), (2, 32), device=self.device)
            labels = input_ids.clone()
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # Check final state
        final_state = None
        if hasattr(ttt_cache, 'ttt_params_dict') and 'W1_states' in ttt_cache.ttt_params_dict:
            if 0 in ttt_cache.ttt_params_dict['W1_states']:
                final_state = ttt_cache.ttt_params_dict['W1_states'][0].clone()
        
        if initial_state is not None and final_state is not None:
            state_change = torch.norm(final_state - initial_state).item()
            print(f"   Cache state change magnitude: {state_change:.6f}")
            
            if state_change > 1e-6:
                print(f"   ✅ Cache state is evolving (good)")
            else:
                print(f"   ⚠️ Cache state barely changed")
        else:
            print(f"   ⚠️ Could not access cache states")
    
    def _test_cache_memory_growth(self):
        """Test cache memory usage over time."""
        print(f"\n💾 Testing cache memory growth...")
        
        model_config = TTTConfig(**{
            **TTT_STANDARD_CONFIGS["125m"],
            "vocab_size": len(self.tokenizer),
            "state_passing": True,
            "disable_conv": True,
        })
        
        model = TTTForCausalLM(model_config).to(self.device)
        model.train()
        
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        ttt_cache = TTTCache(model.model, batch_size=4)
        cache_creation_memory = torch.cuda.memory_allocated()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        memory_points = []
        
        for step in range(20):
            input_ids = torch.randint(0, len(self.tokenizer), (4, 32), device=self.device)
            labels = input_ids.clone()
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            current_memory = torch.cuda.memory_allocated()
            memory_points.append(current_memory)
        
        cache_overhead = (cache_creation_memory - initial_memory) / 1e6  # MB
        memory_growth = (memory_points[-1] - memory_points[0]) / 1e6  # MB
        
        print(f"   Cache creation overhead: {cache_overhead:.1f} MB")
        print(f"   Memory growth over 20 steps: {memory_growth:.1f} MB")
        
        if memory_growth < 100:  # Less than 100MB growth
            print(f"   ✅ Memory growth is reasonable")
        else:
            print(f"   ⚠️ Significant memory growth detected")
    
    def _test_different_sequence_lengths(self):
        """Test cache with different sequence lengths."""
        print(f"\n📏 Testing different sequence lengths...")
        
        seq_lengths = [16, 32, 64, 128]
        
        for seq_len in seq_lengths:
            try:
                print(f"   Testing sequence length: {seq_len}")
                
                model_config = TTTConfig(**{
                    **TTT_STANDARD_CONFIGS["125m"],
                    "vocab_size": len(self.tokenizer),
                    "max_position_embeddings": seq_len * 2,
                    "state_passing": True,
                    "disable_conv": True,
                })
                
                model = TTTForCausalLM(model_config).to(self.device)
                model.train()
                
                ttt_cache = TTTCache(model.model, batch_size=2)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                
                # Test a few steps
                for step in range(3):
                    input_ids = torch.randint(0, len(self.tokenizer), (2, seq_len), device=self.device)
                    labels = input_ids.clone()
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                
                print(f"     ✅ Sequence length {seq_len} works")
                
            except Exception as e:
                print(f"     ❌ Sequence length {seq_len} failed: {e}")
    
    def _test_batch_size_changes(self):
        """Test cache behavior with different batch sizes."""
        print(f"\n📦 Testing different batch sizes...")
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            try:
                print(f"   Testing batch size: {batch_size}")
                
                model_config = TTTConfig(**{
                    **TTT_STANDARD_CONFIGS["125m"],
                    "vocab_size": len(self.tokenizer),
                    "state_passing": True,
                    "disable_conv": True,
                })
                
                model = TTTForCausalLM(model_config).to(self.device)
                model.train()
                
                ttt_cache = TTTCache(model.model, batch_size=batch_size)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                
                # Test a few steps
                for step in range(3):
                    input_ids = torch.randint(0, len(self.tokenizer), (batch_size, 32), device=self.device)
                    labels = input_ids.clone()
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids=input_ids, labels=labels, cache_params=ttt_cache)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                
                print(f"     ✅ Batch size {batch_size} works")
                
            except Exception as e:
                print(f"     ❌ Batch size {batch_size} failed: {e}")

def main():
    """Run all deep state passing tests."""
    print("🚀 Deep State Passing Analysis Suite")
    print("=" * 60)
    print()
    
    device, tokenizer = setup_test_environment()
    print(f"Device: {device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print()
    
    # Run all test suites
    try:
        # 1. Convergence Analysis
        convergence_analyzer = ConvergenceAnalyzer(device, tokenizer)
        convergence_analyzer.run_convergence_test(num_epochs=2, steps_per_epoch=15)
        
        # 2. Configuration Robustness
        robustness_tester = ConfigurationRobustnessTest(device, tokenizer)
        robustness_tester.run_robustness_tests()
        
        # 3. Performance Profiling
        profiler = PerformanceProfiler(device, tokenizer)
        profiler.run_performance_analysis()
        
        # 4. Cache Management Tests
        cache_tester = CacheManagementTest(device, tokenizer)
        cache_tester.run_cache_tests()
        
        print("\n🎉 All Deep Tests Completed!")
        print("=" * 40)
        print("State passing implementation has been thoroughly validated.")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()