import pickle
import json
import os
import sys
from pprint import pprint

# === 添加这部分：导入 Isaac Lab 相关模块 ===
try:
    # 添加项目路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # 导入必要的模块（根据你的项目结构调整）
    import isaaclab
    from isaaclab.assets import *
    from isaaclab.envs import *
    from isaaclab.managers import *
    from isaaclab.scene import *
    from isaaclab.sensors import *
    from isaaclab.utils import *
    
    # 导入你的自定义环境
    import envs
    
    print("[INFO] Isaac Lab modules loaded successfully")
except ImportError as e:
    print(f"[WARNING] Failed to import some modules: {e}")
    print("[INFO] Will try to load pkl anyway...")

import torch
import numpy as np

def read_and_save_pkl(pkl_path, output_format='txt'):
    """
    读取 pkl 文件并保存为可读格式
    
    Args:
        pkl_path: pkl 文件路径
        output_format: 输出格式，可选 'txt', 'json', 'both'
    """
    # 检查文件是否存在
    if not os.path.exists(pkl_path):
        print(f"[ERROR] File not found: {pkl_path}")
        return
    
    print(f"[INFO] Reading: {pkl_path}")
    
    # 读取 pkl 文件
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"[SUCCESS] Loaded pkl file")
    except Exception as e:
        print(f"[ERROR] Failed to load pkl: {e}")
        print("\n[INFO] Trying alternative method...")
        
        # 方法2：使用自定义 unpickler 忽略缺失的模块
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except (ImportError, AttributeError):
                    print(f"[WARNING] Skipping unavailable class: {module}.{name}")
                    # 返回一个占位符类
                    return type(name, (), {'__module__': module, '__name__': name})
        
        try:
            with open(pkl_path, 'rb') as f:
                data = CustomUnpickler(f).load()
            print(f"[SUCCESS] Loaded pkl with custom unpickler")
        except Exception as e2:
            print(f"[ERROR] Alternative method also failed: {e2}")
            return None
    
    print(f"[INFO] Data type: {type(data)}")
    
    # 转换数据为可序列化的格式
    def make_serializable(obj, depth=0, max_depth=10):
        """递归转换对象为可序列化格式"""
        if depth > max_depth:
            return "<MAX_DEPTH_REACHED>"
        
        if isinstance(obj, dict):
            return {str(k): make_serializable(v, depth+1, max_depth) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item, depth+1, max_depth) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return f"<Tensor shape={list(obj.shape)} dtype={obj.dtype}>"
        elif isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            # 对于自定义对象，尝试提取其属性
            try:
                attrs = {k: make_serializable(v, depth+1, max_depth) 
                        for k, v in obj.__dict__.items() 
                        if not k.startswith('_')}
                return {
                    '__type__': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                    '__attributes__': attrs
                }
            except:
                return f"<Object {type(obj).__name__}>"
        elif callable(obj):
            try:
                return f"<Function {obj.__module__}.{obj.__name__}>"
            except:
                return f"<Callable {type(obj).__name__}>"
        else:
            try:
                json.dumps(obj)  # 测试是否可序列化
                return obj
            except:
                return str(obj)
    
    # 准备输出路径
    base_name = os.path.splitext(pkl_path)[0]
    
    # 1. 保存为 txt 格式（详细的打印输出）
    if output_format in ['txt', 'both']:
        txt_path = base_name + '_readable.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"PKL File: {pkl_path}\n")
            f.write(f"Data Type: {type(data)}\n")
            f.write("="*80 + "\n\n")
            
            # 使用 pprint 格式化输出
            try:
                f.write(pprint.pformat(data, width=120, depth=6))
            except Exception as e:
                f.write(f"Error formatting data: {e}\n")
                f.write(str(data))
        
        print(f"[SUCCESS] Saved readable text to: {txt_path}")
    
    # 2. 保存为 JSON 格式（结构化数据）
    if output_format in ['json', 'both']:
        json_path = base_name + '_readable.json'
        try:
            serializable_data = make_serializable(data)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            print(f"[SUCCESS] Saved JSON to: {json_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save JSON: {e}")
    
    # 3. 打印摘要信息到控制台
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    
    def print_summary(obj, indent=0):
        """递归打印对象摘要"""
        prefix = "  " * indent
        
        if isinstance(obj, dict):
            print(f"{prefix}Dictionary with {len(obj)} keys:")
            for key, value in list(obj.items())[:20]:  # 最多显示20个键
                value_type = type(value).__name__
                if isinstance(value, (list, tuple)):
                    print(f"{prefix}  [{key}]: {value_type} (length={len(value)})")
                elif isinstance(value, dict):
                    print(f"{prefix}  [{key}]: {value_type} (keys={len(value)})")
                    if indent < 2:  # 只递归2层
                        print_summary(value, indent+2)
                elif isinstance(value, torch.Tensor):
                    print(f"{prefix}  [{key}]: Tensor (shape={list(value.shape)}, dtype={value.dtype})")
                elif isinstance(value, np.ndarray):
                    print(f"{prefix}  [{key}]: ndarray (shape={value.shape}, dtype={value.dtype})")
                elif hasattr(value, '__dict__') and not isinstance(value, type):
                    print(f"{prefix}  [{key}]: {value.__class__.__name__}")
                    if indent < 2:
                        print_summary(value.__dict__, indent+2)
                else:
                    print(f"{prefix}  [{key}]: {value_type}")
                    if len(str(value)) < 100:
                        print(f"{prefix}      Value: {value}")
            
            if len(obj) > 20:
                print(f"{prefix}  ... and {len(obj) - 20} more keys")
        
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}{type(obj).__name__} with {len(obj)} items")
            for i, item in enumerate(list(obj)[:5]):  # 只显示前5项
                print(f"{prefix}  [{i}]: {type(item).__name__}")
        
        else:
            print(f"{prefix}Data content:\n{prefix}{str(obj)[:500]}")
    
    print_summary(data)
    print("="*80 + "\n")
    
    return data


# 使用示例
if __name__ == "__main__":
    # 读取 env.pkl
    pkl_file = "/home/wzx/Documents/DWAM/logs/skrl/PushBox/Nov29_13-13-26_PPO/params/env.pkl"
    data = read_and_save_pkl(pkl_file, output_format='both')
    
    # 如果成功加载，可以进一步探索
    if data is not None and isinstance(data, dict):
        print("\n" + "="*80)
        print("REWARD CONFIGURATION (if exists)")
        print("="*80)
        
        # 尝试查找 reward 相关的配置
        for key in data.keys():
            if 'reward' in key.lower() or 'action' in key.lower() or 'observation' in key.lower():
                print(f"\n{key}:")
                pprint(data[key], depth=4)