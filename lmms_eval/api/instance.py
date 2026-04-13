from dataclasses import dataclass, field
from typing import Literal, Tuple
from typing import Optional

@dataclass
class Instance:
    request_type: Literal["loglikelihood", "generate_until", "generate_until_multi_round", "generate_until_agentic"]
    arguments: tuple
    idx: int
    metadata: Tuple[str, int, int] = field(default_factory=lambda: (None, None, None))  # TODO: better typehints here
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)
    
    # initialized after init
    task_name: str = None
    doc_id: str = None
    repeats: str = None
    doc: dict = None
    success: Optional[bool] = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata["task"], self.metadata["doc_id"], self.metadata["repeats"]

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)


@dataclass
class AgenticInstance(Instance):
    # env_step_fn is doc_to_text which acts as the environment simulator
    env_step_fn: callable = None
    max_steps: int = 10
    current_step: int = 0
    history_trace: list = field(default_factory=list)
    previous_round_info: dict = None
    is_done: bool = False
    
    def step(self, model_response: str) -> bool:
        if self.is_done:
            return True
            
        self.history_trace.append(model_response)
        
        # Invoke environment step
        # By convention in agentic tasks, doc_to_text(doc, previous_output, round_idx, previous_round_info)
        # returns (visuals, next_context, terminal_signal, updated_outputs, next_round_info)
        
        doc = self.doc
        step_payload = self.env_step_fn(
            doc,
            previous_output=self.history_trace,
            round_idx=self.current_step + 1,
            previous_round_info=self.previous_round_info,
        )
        
        if isinstance(step_payload, tuple) and len(step_payload) == 5:
            visuals, next_context, terminal_signal, updated_outputs, next_round_info = step_payload
            
            if updated_outputs is not None:
                self.history_trace = list(updated_outputs)
            
            self.previous_round_info = next_round_info
            
            if terminal_signal:
                self.is_done = True
            
            if next_context is not None:
                args_list = list(self.arguments)
                args_list[0] = next_context # update ctx
                
                # Update visuals if provided
                if visuals is not None:
                    if len(args_list) >= 3:
                        if isinstance(args_list[1], dict):
                            # It's ConfigurableTask: (ctx, gen_kwargs, doc_to_visual, ...)
                            if callable(args_list[2]):
                                args_list[2] = lambda _: visuals
                        elif isinstance(args_list[2], dict):
                            # It's ConfigurableMessagesTask: (ctx, doc_to_messages, gen_kwargs, ...)
                            if callable(args_list[1]):
                                # We need to create a new doc_to_messages that injects the new visuals
                                # but usually doc_to_messages just ignores the visual argument if we override it,
                                # wait, doc_to_messages for agentic in chat models takes the visuals from doc_to_visual usually.
                                # Let's just create a dynamic doc_to_messages
                                def _dynamic_doc_to_messages(_doc):
                                    content = []
                                    for visual in visuals:
                                        if isinstance(visual, dict):
                                            content.append({"type": "audio", "url": visual})
                                        elif isinstance(visual, str):
                                            content.append({"type": "video", "url": visual})
                                        else:
                                            content.append({"type": "image", "url": visual})
                                    content.append({"type": "text", "text": next_context})
                                    return [{"role": "user", "content": content}]
                                args_list[1] = _dynamic_doc_to_messages
                                
                self.arguments = tuple(args_list)
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.is_done = True
            
        if self.is_done:
            # Finally append the result. We use history trace as the final resp since it contains the structured dictionary
            self.resps.append(self.history_trace[-1] if self.history_trace else "")
            
        return self.is_done
