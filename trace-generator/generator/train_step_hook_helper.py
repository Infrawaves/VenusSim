from typing import Dict, Callable

# Train Step:   
# forward_pre_hook -> forward -> forward_hook -> backward -> backward_hook
class TrainStepHookHelper:
    def __init__(self) -> None:
        self.forward_pre_hook_map: Dict[str, Callable] = {}
        self.forward_hook_map: Dict[str, Callable] = {}
        self.backward_hook_map: Dict[str, Callable] = {}

    def set_forward_pre_hook(
        self,
        hook_name: str,
        hook: Callable
    ):
        assert hook_name not in self.forward_pre_hook_map.keys(), \
            "Duplicated hook in forward_pre_hook_map."
        self.forward_pre_hook_map[hook_name] = hook
    
    def set_forward_hook(
        self,
        hook_name: str,
        hook: Callable
    ):
        assert hook_name not in self.forward_hook_map.keys(), \
            "Duplicated hook in forward_hook_map."
        self.forward_hook_map[hook_name] = hook
    
    def set_backward_hook(
        self,
        hook_name: str,
        hook: Callable
    ):
        assert hook_name not in self.backward_hook_map.keys(), \
            "Duplicated hook in backward_hook_map."
        self.backward_hook_map[hook_name] = hook

    def delete_forward_pre_hook(
        self,
        hook_name: str
    ):
        assert hook_name in self.forward_pre_hook_map.keys(), \
            f"Hook {hook_name} not found forward_pre_hook_map."
        self.forward_pre_hook_map.pop(hook_name)
    
    def delete_forward_hook(
        self,
        hook_name: str
    ):
        assert hook_name in self.forward_hook_map.keys(), \
            f"Hook {hook_name} not found forward_hook_map."
        self.forward_hook_map.pop(hook_name)
    
    def delete_backward_hook(
        self,
        hook_name: str
    ):
        assert hook_name in self.backward_hook_map.keys(), \
            f"Hook {hook_name} not found backward_hook_map."
        self.backward_hook_map.pop(hook_name)

    def forward_pre_hook(
        self,
        hook_name: str
    ):
        assert hook_name in self.forward_pre_hook_map.keys(), \
            f"Hook {hook_name} not found forward_pre_hook_map."
        return self.forward_pre_hook_map[hook_name]
    
    def forward_hook(
        self,
        hook_name: str
    ):
        assert hook_name in self.forward_hook_map.keys(), \
            f"Hook {hook_name} not found forward_hook_map."
        return self.forward_hook_map[hook_name]
    
    def backward_hook(
        self,
        hook_name: str
    ):
        assert hook_name in self.backward_hook_map.keys(), \
            f"Hook {hook_name} not found backward_hook_map."
        return self.backward_hook_map[hook_name]

